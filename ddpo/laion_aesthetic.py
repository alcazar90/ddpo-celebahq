import os
import requests
import numpy as np

import clip
import torch

from torch import nn
from PIL import Image
from typing import Union, List


class AestheticRewardModel(nn.Module):
    """
    A wrapper class to instantiate and use the LAION aesthetic predictor.
    Given an image, this model builds a linear model on top of the CLIP embeddings and returns the aesthetic score.
    The score ranges from 0 to 10, indicating the aesthetic quality of the image.

    Reference about the MLP: https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/6934dd81792f086e613a121dbce43082cb8be85e/train_predictor.py#L17
    More about the LAION aesthetic predictor: https://laion.ai/blog/laion-aesthetics/
    More about the CLIP model: https://github.com/mlfoundations/open_clip
    """

    def __init__(self, model_checkpoint: str, device: str = 'cuda', cache: str = "."):
        """
        Initialize the AestheticRewardModel.

        Args:
            model_checkpoint (str): Path to the CLIP model checkpoint.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
            cache (str, optional): Cache directory for model weights. Defaults to ".".
        """
        super().__init__()
        self.device = torch.device(device)
        self.clip_model, self.preprocess = clip.load(model_checkpoint, device=device)
        self.aesthetic_model = self._initialize_aesthetic_model(cache)

    def _initialize_aesthetic_model(self, cache: str = ".") -> nn.Module:
        """
        Initialize the aesthetic model MLP and load pre-trained weights.

        Args:
            cache (str, optional): Cache directory for model weights. Defaults to ".".

        Returns:
            nn.Module: Initialized aesthetic model.
        """
        class MLP(nn.Module):
            def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
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
                    nn.Linear(16, 1)
                )

            def forward(self, x):
                return self.layers(x)

        weights_fname = "sac+logos+ava1-l14-linearMSE.pth"
        loadpath = os.path.join(cache, weights_fname)

        if not os.path.exists(loadpath):
            url = (
                "https://github.com/christophschuhmann/"
                f"improved-aesthetic-predictor/blob/main/{weights_fname}?raw=true"
            )
            r = requests.get(url)

            with open(loadpath, "wb") as f:
                f.write(r.content)

        weights = torch.load(loadpath, map_location=torch.device("cpu"))
        mlp = MLP(768)
        mlp.load_state_dict(weights)
        return mlp.to(self.device)

    def _normalize(self, a: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:
        """
        Normalize the input array.

        Args:
            a (np.ndarray): Input array to normalize.
            axis (int, optional): Axis along which to normalize. Defaults to -1.
            order (int, optional): Order of the normalization. Defaults to 2.

        Returns:
            np.ndarray: Normalized array.
        """
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def _from_tensor_to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor RGB img to a numpy array representation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            np.ndarray: Converted numpy array.
        """
        return ((x.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5).numpy() * 255).astype(np.uint8)

    def forward(self, images: Union[np.ndarray, List[Union[np.ndarray, torch.Tensor]]]) -> torch.Tensor:
        """
        Perform inference on either a single image or a batch of images.

        Args:
            images (Union[np.ndarray, List[Union[np.ndarray, torch.Tensor]]]): Input image or list of images.

        Returns:
            torch.Tensor: Aesthetic score of the image(s).
        """
        if not isinstance(images, list):
          images = [images]

        if isinstance(images[0], np.ndarray):
          n_dim = images[0].shape
          images = [torch.from_numpy(img) for img in images[0]] if len(n_dim) == 4 else [torch.from_numpy(img) for img in images]

        if isinstance(images[0], torch.Tensor):
          n_dim = images[0].shape
          images = [img for img in images[0]] if len(n_dim) == 4 else images

        if images[0].device != self.device:
          images = [img.to(self.device) for img in images]

        with torch.no_grad():
          if len(images) == 1:
              image_features = self.clip_model.encode_image(self.preprocess(Image.fromarray(self._from_tensor_to_numpy(images[0]))).unsqueeze(0).to(self.device))
              im_emb_arr = self._normalize(image_features.cpu().detach().numpy())
              self.aesthetic_model.eval()
              prediction = self.aesthetic_model(torch.from_numpy(im_emb_arr).float().to(self.device))
              return prediction
          else:
              imgs = torch.stack([self.preprocess(Image.fromarray(self._from_tensor_to_numpy(img))) for img in images])
              image_features = self.clip_model.encode_image(imgs.to(self.device))
              im_emb_arr = self._normalize(image_features.cpu().detach().numpy())
              self.aesthetic_model.eval()
              prediction = self.aesthetic_model(torch.from_numpy(im_emb_arr).float().to(self.device))
              return prediction
