from model_configs import upstream_repo_path as UPSTREAM_REPO
import sys
import importlib.util
from pathlib import Path

def has_upstream() -> bool:
    """
        git clone https://github.com/Megvii-BaseDetection/YOLOX
    """
    if not (UPSTREAM_REPO / "yolox").is_dir():
        return False
    if str(UPSTREAM_REPO) not in sys.path:
        sys.path.append(str(UPSTREAM_REPO))
    return True


def has_val_dataset(config: dict) -> bool:
    """Whether COCO val2017, its 10-class annotations and pycocotools, to score it, are there."""
    return (
        Path(config["val_image_dir"]).is_dir()
        and Path(config["val_ann_file"]).is_file()
        and importlib.util.find_spec("pycocotools") is not None
    )


def has_train_dataset(config: dict) -> bool:
    """Whether COCO train2017 and its 10-class annotations, for QAT, are there."""
    return Path(config["train_image_dir"]).is_dir() and Path(config["train_ann_file"]).is_file()