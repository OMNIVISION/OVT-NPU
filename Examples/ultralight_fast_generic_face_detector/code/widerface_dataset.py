import math
import os

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class WiderFaceDataset(Dataset):
    """
    Exactly the images given, in the order given, with no ground truth.

    Yields (image, target), image being CHW float32 in [0, 255] at the network input
    size. Normalisation is left to the model, whose QStandardize2d folds it in, and to
    the fp32 path's preprocessor. target holds:
        path        str    the image file, which the WIDER FACE kit scores by
        orig_size   [2]    (width, height) of the source, the decode scales boxes by it
    """

    def __init__(self, files, input_shape=(120, 160)) -> None:
        """
        Args:
            files (list): Image paths, used as given
            input_shape (tuple): Network input as (height, width)
        """
        self.files = [str(f) for f in files]
        if not self.files:
            msg = "WiderFaceDataset needs at least one file!"
            raise AssertionError(msg)
        self.input_height, self.input_width = input_shape

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, position):
        path = self.files[position]
        image = cv2.imread(path)
        if image is None:
            msg = f"Failed to read image {path}!"
            raise AssertionError(msg)

        height, width = image.shape[:2]
        # A plain resize, not a letterbox, as upstream's own evaluation does
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height))

        target = {
            "path": path,
            "orig_size": torch.tensor([width, height], dtype=torch.float32),
        }
        return torch.from_numpy(image).permute(2, 0, 1).float(), target

## Use custom dataloader instead of using BaseInference's dataloader. 
def detection_collate(batch):
    images, targets = zip(*batch)
    return torch.stack(list(images), 0), list(targets)

def WiderFaceLoader(dataset, batch_size, collate_fn=detection_collate, shuffle=False):
    """Wraps a detection dataset in a DataLoader, batched with detection_collate."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=min(max(os.cpu_count() // 2, 1), math.ceil(len(dataset) / batch_size)), # No more workers than batches: a one-image dump needs one, not half the cores
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )