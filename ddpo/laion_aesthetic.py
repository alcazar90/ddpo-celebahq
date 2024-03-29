"""LAION Aesthetic predictor for the Deep Reinforcement Learning project."""

from __future__ import annotations

from pathlib import Path

import clip
import numpy as np
import requests
import torch
from PIL import Image
from torch import nn


class AestheticRewardModel(nn.Module):
    """A wrapper class to instantiate and use the LAION aesthetic predictor.

    Given an image, this model builds a linear model on top of the CLIP embeddings
    and returns the aesthetic score.
    The score ranges from 0 to 10, indicating the aesthetic quality of the image.

    Reference about the MLP:
    https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/6934dd81792f086e613a121dbce43082cb8be85e/train_predictor.py#L17
    More about the LAION aesthetic predictor: https://laion.ai/blog/laion-aesthetics/
    More about the CLIP model: https://github.com/openai/CLIP

    About clip features reproducibility issue: https://github.com/openai/CLIP/issues/13
    """

    def __init__(
        self,  # noqa: ANN101
        model_checkpoint: str,
        device: str = "cuda",
        cache: str = ".",
    ) -> None:
        """Initialize the AestheticRewardModel.

        Args:
        ----
            model_checkpoint (str): Path to the CLIP model checkpoint.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
            cache (str, optional): Cache directory for model weights. Defaults to ".".

        """
        super().__init__()
        self.device = torch.device(device)
        self.clip_model, self.preprocess = clip.load(model_checkpoint, device=device)
        self.aesthetic_model = self._initialize_aesthetic_model(cache)

    def _initialize_aesthetic_model(self, cache: str = ".") -> nn.Module:  # noqa: ANN101
        """Initialize the aesthetic model MLP and load pre-trained weights.

        Args:
        ----
            cache (str, optional): Cache directory for model weights. Defaults to ".".

        Returns:
        -------
            nn.Module: Initialized aesthetic model.

        """

        class MLP(nn.Module):
            def __init__(
                self,  # noqa: ANN101
                input_size: int,
                xcol: str = "emb",
                ycol: int = "avg_rating",
            ) -> None:
                super().__init__()
                self.input_size = input_size
                self.xcol = xcol
                self.ycol = ycol
                self.layers = nn.Sequential(
                    nn.Linear(self.input_size, 1024),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.Dropout(0.1),
                    nn.Linear(64, 16),
                    nn.Linear(16, 1),
                )

            def forward(self, x: torch.tensor) -> torch.tensor:  # noqa: ANN101
                return self.layers(x)

        weights_fname = "sac+logos+ava1-l14-linearMSE.pth"
        loadpath = Path(cache) / weights_fname

        if not loadpath.exists():
            url = (
                "https://github.com/christophschuhmann/"
                f"improved-aesthetic-predictor/blob/main/{weights_fname}?raw=true"
            )
            r = requests.get(url, timeout=10)  # Added timeout parameter

            with loadpath.open("wb") as f:  # Replaced open() with Path.open()
                f.write(r.content)

        weights = torch.load(loadpath, map_location=torch.device("cpu"))
        mlp = MLP(768)
        mlp.load_state_dict(weights)
        return mlp.to(self.device)

    def _normalize(self, a: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:  # noqa: ANN101
        """Normalize the input array.

        Args:
        ----
            a (np.ndarray): Input array to normalize.
            axis (int, optional): Axis along which to normalize. Defaults to -1.
            order (int, optional): Order of the normalization. Defaults to 2.

        Returns:
        -------
            np.ndarray: Normalized array.

        """
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def _from_tensor_to_numpy(self, x: torch.Tensor) -> np.ndarray:  # noqa: ANN101
        """Convert a PyTorch tensor RGB img to a numpy array representation.

        Args:
        ----
            x (torch.Tensor): Input tensor.

        Returns:
        -------
            np.ndarray: Converted numpy array.

        """
        np_img = (x * 0.5 + 0.5).clamp(0, 1)
        np_img = np_img.detach().cpu().permute(1, 2, 0).numpy()
        return (np_img * 255).round().astype(np.uint8)

    def forward(
        self,  # noqa: ANN101
        images: np.ndarray | list[np.ndarray | torch.Tensor],
    ) -> torch.Tensor:
        """Perform inference on either a single image or a batch of images.

        Args:
        ----
            images (Union[np.ndarray, List[Union[np.ndarray, torch.Tensor]]]): Input image or list of images.

        Returns:
        -------
            torch.Tensor: Aesthetic score of the image(s).

        """  # noqa: E501
        if not isinstance(images, list):
            images = [images]

        if isinstance(images[0], np.ndarray):
            check_array_dim = 4
            n_dim = images[0].shape
            images = (
                [torch.from_numpy(img) for img in images[0]]
                if len(n_dim) == check_array_dim
                else [torch.from_numpy(img) for img in images]
            )

        if isinstance(images[0], torch.Tensor):
            check_tensor_dim = 4
            n_dim = images[0].shape
            images = list(images[0]) if len(n_dim) == check_tensor_dim else images

        if images[0].device != self.device:
            images = [img.to(self.device) for img in images]

        with torch.no_grad():
            self.clip_model.eval()
            self.aesthetic_model.eval()
            if len(images) == 1:
                clip_input = (
                    self.preprocess(
                        Image.fromarray(self._from_tensor_to_numpy(images[0])),
                    )
                    .unsqueeze(0)
                    .cuda()
                )
                image_features = self.clip_model.encode_image(clip_input)
                im_emb_arr = self._normalize(image_features.cpu().detach().numpy())
                return self.aesthetic_model(
                    torch.from_numpy(im_emb_arr).float().to(self.device),
                ).squeeze(1)  # return torch tensor 1dim
            clip_input = [
                self.preprocess(
                    Image.fromarray(self._from_tensor_to_numpy(img)),
                ).cuda()
                for img in images
            ]
            clip_input = torch.stack(clip_input)
            image_features = self.clip_model.encode_image(clip_input)
            im_emb_arr = self._normalize(image_features.cpu().detach().numpy())
            return self.aesthetic_model(
                torch.from_numpy(im_emb_arr).float().to(self.device),
            ).squeeze(1)  # return torch tensor 1dim
