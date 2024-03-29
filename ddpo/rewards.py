import io
import torch
import numpy as np

from PIL import Image
from ddpo.utils import decode_tensor_to_np_img

# Following closure style for rewards functions using in https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/rewards.py

def aesthetic_score():
    from ddpo.laion_aesthetic import AestheticRewardModel

    laion_aesthetic = AestheticRewardModel(model_checkpoint="ViT-L/14",
                                           device="cuda")
    
    def _fn(images):
        aesthetic_score = laion_aesthetic(images)
        return aesthetic_score
    
    return _fn


def over50_old(threshold: float = 0.6, punishment: float = -1.0):
    from transformers import ViTImageProcessor, ViTForImageClassification
    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    transforms = ViTImageProcessor.from_pretrained('nateraw/vit-age-classifier')
    model.to("cuda")
    model.eval()

    def _fn(images):
        inputs = transforms(
            decode_tensor_to_np_img(images,
                                    melt_batch=False,),
            return_tensors="pt").pixel_values.cuda()
        with torch.no_grad():
          outputs = model(inputs).logits
        probs = outputs.softmax(dim=1)
        probs = probs[:, 6:].sum(dim=1)
        rewards = torch.where(probs > threshold, probs, punishment)
        return rewards

    return _fn


def under30_old(threshold: float = 0.6, punishment: float = -1.0):
    from transformers import ViTImageProcessor, ViTForImageClassification
    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    transforms = ViTImageProcessor.from_pretrained('nateraw/vit-age-classifier')
    model.to("cuda")
    model.eval()

    def _fn(images):
        inputs = transforms(
            decode_tensor_to_np_img(images,
                                    melt_batch=False,),
            return_tensors="pt").pixel_values.cuda()
        with torch.no_grad():
          outputs = model(inputs).logits
        probs = outputs.softmax(dim=1)
        probs = probs[:, :4].sum(dim=1)
        rewards = torch.where(probs > threshold, probs, punishment)
        return rewards

    return _fn


def jpeg_incompressibility():
    """Return the size of the images in kilobytes, after JPEG compression"""
    def _fn(images, metadata=None):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        # return np.array(sizes), {}
        return np.array(sizes)

    return _fn


def jpeg_compressibility():
    """Return the negative size of the images in kilobytes, after JPEG compression"""
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, metadata=None):
        rew, meta = jpeg_fn(images, metadata)
        # return -rew, meta
        return rew

    return _fn
