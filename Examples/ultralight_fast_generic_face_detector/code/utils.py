import sys
from pathlib import Path
from model_configs import upstream_repo_path as UPSTREAM_REPO

def has_upstream(config: dict) -> bool:
    upstream_path =UPSTREAM_REPO / "vision"
    if not upstream_path.is_dir():
        return False
    if str(UPSTREAM_REPO) not in sys.path:
        sys.path.append(str(UPSTREAM_REPO))
    return True

def has_val_dataset(config: dict) -> bool:
    val_path = Path(config["widerface_val_dir"])
    return val_path.is_dir()

def has_train_dataset(config: dict) -> bool:
    train_path = Path(config["voc_data_dir"])
    return train_path.is_dir()