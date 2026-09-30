"""
The data of the flow: what is on disk, the datasets and loaders, and the COCO scoring.

Two datasets feed the flow:
    coco_dataset()      COCO 2017 (10 classes) through Megvii YOLOX's official
                        COCODataset, when the clone and the dataset are there
    ImageListDataset    the images provided with the example: the calibration set and
                        images/test.png

Every batch is (images [B, 3, H, W], targets), targets being one dict per image:
    image_id    int     its COCO id, -1 outside COCO
    orig_size   [2]     (height, width) of the source, the decode scales boxes by it
    labels      [M, 5]  (cls, cx, cy, w, h) in input pixels, COCO only, for QAT's loss
"""

import contextlib
import io
import math
import os

import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# ----------------------------------------------------------------------------------
# Datasets and loaders
# ----------------------------------------------------------------------------------
def coco_dataset(config: dict, split: str, input_shape):
    """
    COCO's split ('train' or 'val') through Megvii YOLOX's official COCODataset, as its
    own training and evaluation read it; the first num_train_files / num_test_files
    images only, if the config limits them. Needs has_upstream().
    """
    from .yolox_coco import YoloxCocoDataset  # imports Megvii's yolox

    dataset = YoloxCocoDataset(
        config[f"{split}_ann_file"],
        config[f"{split}_image_dir"],
        input_shape,
        config["max_labels"],
        train=split == "train",
    )
    max_items = config["num_train_files" if split == "train" else "num_test_files"]
    return Subset(dataset, range(min(max_items, len(dataset)))) if max_items else dataset


def calibration_files(config: dict) -> list[Path]:
    """The calibration images, unzipped from calib_zip on first use."""
    image_dir = Path(config["calib_data_dir"])
    if not image_dir.is_dir():
        with zipfile.ZipFile(config["calib_zip"]) as archive:
            archive.extractall(image_dir)
    files = sorted(image_dir.glob("*.jpg"))
    max_items = config["num_calib_files"]
    return files[:max_items] if max_items else files


def letterbox(image: np.ndarray, input_size) -> np.ndarray:
    """
    YOLOX's preprocessing: keep the aspect ratio, place the image at the top left,
    pad with 114, CHW float32 in [0, 255]. The channel order stays BGR, as read.
    """
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    ratio = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    height, width = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
    padded[:height, :width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)


class ImageListDataset(Dataset):
    """Exactly the images given, in the order given, letterboxed, with no ground truth."""

    def __init__(self, files, input_shape=(256, 256)) -> None:
        """
        Args:
            files (list): Image paths
            input_shape (tuple): Network input as (height, width)
        """
        self.files = [str(f) for f in files]
        self.input_shape = input_shape

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, position):
        path = self.files[position]
        image = cv2.imread(path)
        if image is None:
            msg = f"Failed to read image {path}!"
            raise AssertionError(msg)

        target = {
            "image_id": -1,
            "orig_size": torch.tensor(image.shape[:2], dtype=torch.float32),
        }
        return torch.from_numpy(letterbox(image, self.input_shape)), target


def detection_collate(batch):
    """Stacks the images and keeps the targets as a list of per-image dicts."""
    images, targets = zip(*batch)
    return torch.stack(list(images), 0), list(targets)


def CocoLoader(dataset, batch_size, shuffle=False, workers=True):
    """
    Wraps a dataset in a DataLoader batched with detection_collate.

    workers=False reads in the calling process, with no pinned memory, for the
    simulator, whose heap is corrupted by a loader worker and the pin-memory thread.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # No more workers than batches
        num_workers=min(max(os.cpu_count() // 2, 1), math.ceil(len(dataset) / batch_size)) if workers else 0,
        pin_memory=workers,
        drop_last=False,
        collate_fn=detection_collate,
    )


# ----------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------
def coco_ap(detections: list, targets: list, ann_file: str | Path) -> dict[str, float]:
    """
    COCO bbox AP of the detections, over the images they came from.

    Args:
        detections (list): One {'boxes' [N, 4] xyxy in original pixels, 'scores' [N],
            'labels' [N]} per image
        targets (list): The matching targets, whose 'image_id' is the COCO id
        ann_file (str | Path): The ground truth, whose category ids are the model's
            class indices (the 10-class files number them 0..9)
    Returns:
        {'ap50_95', 'ap50'}
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rows = []
    for detection, target in zip(detections, targets):
        for (x0, y0, x1, y1), score, label in zip(
            detection["boxes"].tolist(), detection["scores"].tolist(), detection["labels"].tolist(),
        ):
            rows.append({
                "image_id": int(target["image_id"]),
                "category_id": int(label),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": score,
            })
    if not rows:
        return {"ap50_95": 0.0, "ap50": 0.0}

    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(ann_file))
        evaluation = COCOeval(ground_truth, ground_truth.loadRes(rows), "bbox")
        evaluation.params.imgIds = sorted({int(target["image_id"]) for target in targets})
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    return {"ap50_95": float(evaluation.stats[0]), "ap50": float(evaluation.stats[1])}
