"""
COCO as Megvii YOLOX's own training and evaluation read it: its COCODataset and
TrainTransform, repacked into the flow's (image, target) form.

It imports Megvii's 'yolox' package, so it is only ever imported by
coco_data.coco_dataset(), which the flow calls once has_upstream() has found the clone.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from yolox.data import COCODataset, TrainTransform


class YoloxCocoDataset(Dataset):
    """
    Megvii YOLOX's COCODataset and TrainTransform, as its own training reads COCO,
    with each sample repacked into the flow's (image, target) form, target also
    holding the loss's labels: [max_labels, 5] as (cls, cx, cy, w, h) in input pixels.
    """

    def __init__(self, ann_file: Path, image_dir: Path, input_shape, max_labels: int, train: bool) -> None:
        """
        Args:
            ann_file (Path): <data_dir>/annotations/<json>, where COCODataset looks for it
            image_dir (Path): The images, somewhere under <data_dir>
            input_shape (tuple): Network input as (height, width)
            max_labels (int): Boxes per image the labels are padded to
            train (bool): Random flips and HSV jitter for training, none for validation
        """
        data_dir = ann_file.parent.parent
        self.coco = COCODataset(
            data_dir=str(data_dir),
            json_file=ann_file.name,
            name=str(image_dir.relative_to(data_dir)),
            img_size=tuple(input_shape),
            preproc=TrainTransform(
                max_labels=max_labels,
                flip_prob=0.5 if train else 0.0,
                hsv_prob=1.0 if train else 0.0,
            ),
            cache=False,
        )

    def __len__(self) -> int:
        return len(self.coco)

    def __getitem__(self, position):
        image, labels, (height, width), image_id = self.coco[position]
        target = {
            "image_id": int(image_id[0]),
            "orig_size": torch.tensor([height, width], dtype=torch.float32),
            "labels": torch.from_numpy(labels).float(),
        }
        return torch.from_numpy(image).float(), target
