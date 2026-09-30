"""
Runs the model repository's own WIDER FACE to VOC conversion.

The repository trains from a VOC tree built by data/wider_face_2_voc_add_landmark.py,
which its VOCDataset then reads. That converter is a script, not a module: relative
paths throughout, and an os.makedirs() on its output at import time, so a rerun raises
FileExistsError before doing any work. This module drives it without editing it.

    python widerface_voc.py --download      fetch WIDER FACE, then convert

The download lands in data/widerface/WIDER_<split>/images/<event>/, which calibration
and validation read directly. The conversion produces data/wider_face_add_lm_10_10/
with JPEGImages, Annotations, and ImageSets/Main/{trainval,test}.txt for the train and
val splits, which QAT trains on. It drops faces under 10 pixels, which is what the
'10_10' records; that thins the VOC ground truth only, not the official .mat that
widerface_ap() scores against.
"""

import os
import runpy
import shutil
from pathlib import Path

# The model's repository, cloned beside this file: it ships both the converter
# and the RetinaFace labels the converter reads.
from model_configs import upstream_repo_path as UPSTREAM_REPO

CONVERTER_NAME = "wider_face_2_voc_add_landmark.py"
LABELS_DIRNAME = "retinaface_labels"

VOC_DIRNAME = "wider_face_add_lm_10_10"
# Where torchvision.datasets.WIDERFace puts the archives it downloads
TORCHVISION_DIRNAME = "widerface"
# Where the repository's converter looks for them
CONVERTER_DIRNAME = "wider_face"


def link_torchvision_layout(data_dir: Path) -> None:
    """
    Points the converter's expected wider_face/ at torchvision's widerface/.

    torchvision unpacks to widerface/WIDER_<split>/images/<event>/, which is the
    same shape under a different name, so a symlink is enough and no second copy
    of the images is made.
    """
    source = data_dir / TORCHVISION_DIRNAME
    target = data_dir / CONVERTER_DIRNAME

    if target.exists() or target.is_symlink():
        return
    if not source.is_dir():
        msg = f"{source} not found. Download it first with: python widerface_voc.py --download"
        raise AssertionError(msg)

    target.symlink_to(source.name)


def link_upstream_inputs(data_dir: Path) -> None:
    """
    Links the converter and the RetinaFace labels in from the vendored repository.

    Both ship upstream under its data/, and the converter resolves every path
    relative to the working directory, so they have to sit beside the downloaded
    images. Linking rather than copying keeps one source of truth, and keeps the
    18MB of labels out of the working directory that holds the datasets.
    """
    for name in (CONVERTER_NAME, LABELS_DIRNAME):
        target = data_dir / name
        if target.exists() or target.is_symlink():
            continue

        source = UPSTREAM_REPO / "data" / name
        if not source.exists():
            msg = f"{source} not found in the vendored repository!"
            raise AssertionError(msg)

        target.symlink_to(source)


def voc_root(data_dir: Path) -> Path:
    """Where the converted tree lives, whether or not it has been built yet."""
    return data_dir / VOC_DIRNAME


def is_converted(data_dir: Path) -> bool:
    """True when a previous run left a complete looking VOC tree behind."""
    root = voc_root(data_dir)
    return all(
        (root / part).exists()
        for part in ("JPEGImages", "Annotations", "ImageSets/Main/trainval.txt", "ImageSets/Main/test.txt")
    )


def convert(data_dir: str | Path = "data", force: bool = False) -> Path:
    """
    Builds the VOC tree by running the repository's converter unmodified.

    Args:
        data_dir (str | Path): Directory holding wider_face/ and retinaface_labels/
        force (bool): Rebuild even when a converted tree is already present
    Returns:
        Path of the VOC root, the directory VOCDataset should be pointed at
    """
    data_dir = Path(data_dir).resolve()
    root = voc_root(data_dir)

    if is_converted(data_dir) and not force:
        print(f"WIDER FACE already converted at {root}")
        return root

    link_torchvision_layout(data_dir)
    link_upstream_inputs(data_dir)

    labels = data_dir / LABELS_DIRNAME
    if not labels.is_dir():
        msg = f"{labels} not found, it ships with the model repository under data/."
        raise AssertionError(msg)

    # The converter calls os.makedirs(rootdir) at import time, so a leftover tree
    # from an interrupted run would make it raise before doing any work.
    if root.exists():
        print(f"Removing incomplete conversion at {root}")
        shutil.rmtree(root)

    script = data_dir / CONVERTER_NAME
    if not script.is_file():
        msg = f"{script} not found, it ships with the model repository under data/."
        raise AssertionError(msg)

    # It resolves every path relative to the working directory, so run it from there
    previous = Path.cwd()
    try:
        os.chdir(data_dir)
        print(f"Converting WIDER FACE to VOC with {script.name}, this takes a while")
        runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(previous)

    if not is_converted(data_dir):
        msg = f"Conversion finished but {root} is missing the expected files!"
        raise AssertionError(msg)

    print(f"VOC tree ready at {root}")
    return root


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Convert WIDER FACE to the repository's VOC layout")
    parser.add_argument("--data-dir", default="data", help="Directory holding the raw dataset")
    parser.add_argument("--force", action="store_true", help="Rebuild an existing tree")
    parser.add_argument("--download", action="store_true", help="Fetch WIDER FACE first")
    args = parser.parse_args()

    if args.download:
        from torchvision.datasets import WIDERFace

        for split in ("train", "val"):
            WIDERFace(root=args.data_dir, split=split, download=True)

    convert(args.data_dir, force=args.force)
