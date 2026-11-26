"""Pixel transforms shared by calibration, validation, and QAT."""
import torch
from torchvision import transforms
from .openimages_multilabel import TARGET_SIZE, Letterbox

class ToTensorNoNorm:
    """Custom transform that converts to tensor without normalization (keeps 0-255 range)."""

    def __call__(self, pic):
        import numpy as np
        from PIL import Image

        if isinstance(pic, Image.Image):
            pic = np.array(pic)
            if len(pic.shape) == 2:
                pic = pic[..., None]

        img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        return img.to(torch.float)


# The model was trained on letterboxed images
PreprocWithoutNorm = transforms.Compose(
    [
        Letterbox(TARGET_SIZE),
        ToTensorNoNorm(),
    ],
)

# QAT fine-tune: the flip of the training recipe, nothing that moves the label content
PreprocTrainWithoutNorm = transforms.Compose(
    [
        Letterbox(TARGET_SIZE),
        transforms.RandomHorizontalFlip(),
        ToTensorNoNorm(),
    ],
)

