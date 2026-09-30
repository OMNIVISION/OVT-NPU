"""
The official WIDER FACE Easy / Medium / Hard average precision, from the model
repository's own evaluation kit, widerface_evaluate/evaluation.py.

widerface_ap() hands the kit the flow's detections in memory and runs its scoring
unmodified, so it plugs straight into common.CorpusMetrics:

    CorpusMetrics(functools.partial(widerface_ap, evaluate_dir=...))
"""

import contextlib
import io
import re
import sys
import types
from pathlib import Path

import numpy as np


def numpy_bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """
    Numpy stand-in for the kit's Cython bbox_overlaps, same inclusive-edge convention.

    The kit's box_overlaps.pyx declares 'DTYPE = np.float', which NumPy 1.24 removed,
    so it no longer builds.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    query_boxes = np.asarray(query_boxes, dtype=np.float64)
    if boxes.size == 0 or query_boxes.size == 0:
        return np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float64)

    box_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_area = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    width = np.minimum(boxes[:, None, 2], query_boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], query_boxes[None, :, 0]) + 1
    height = np.minimum(boxes[:, None, 3], query_boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], query_boxes[None, :, 1]) + 1
    intersection = width.clip(min=0) * height.clip(min=0)

    union = box_area[:, None] + query_area[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


def widerface_ap(detections: list, targets: list, evaluate_dir: str | Path) -> dict[str, float]:
    """
    Scores detections on the WIDER FACE validation set with the official kit.

    Needs the model's repository on sys.path, which can_benchmark() sees to.

    Args:
        detections (list): One {'boxes' [N, 4] xyxy in original image pixels,
            'scores' [N]} per image
        targets (list): The matching targets, whose 'path' ends in
            <event>/<image>.jpg, the layout of WIDER_val/images
        evaluate_dir (str | Path): The repository's widerface_evaluate directory
    Returns:
        {'AP_easy', 'AP_medium', 'AP_hard'}
    """
    # The kit imports a Cython 'bbox' module that no longer builds
    sys.modules["bbox"] = types.SimpleNamespace(bbox_overlaps=numpy_bbox_overlaps)
    from widerface_evaluate import evaluation as kit

    # The kit wants an entry for every validation image; one this run did not cover
    # counts as an image with no detections
    gt_dir = str(Path(evaluate_dir, "ground_truth"))
    _, events, files, *_ = kit.get_gt_boxes(gt_dir)
    preds = {str(e[0][0]): {str(f[0][0]): np.empty((0, 5)) for f in fs[0]} for e, fs in zip(events, files)}

    # The kit's box layout: x, y, w, h, score
    for detection, target in zip(detections, targets):
        image, boxes = Path(target["path"]), detection["boxes"].cpu().double().numpy()
        scores = detection["scores"].cpu().double().numpy().reshape(-1)
        preds[image.parent.name][image.stem] = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2], scores])

    # Feed the predictions in place of the directory of .txt files it would read.
    # It prints its APs rather than returning them.
    kit.get_preds = lambda _: preds
    with contextlib.redirect_stdout(log := io.StringIO()):
        kit.evaluation(None, gt_dir)

    aps = re.findall(r"(Easy|Medium|Hard)\s+Val AP: (\S+)", log.getvalue())
    return {f"AP_{setting.lower()}": float(value) for setting, value in aps}
