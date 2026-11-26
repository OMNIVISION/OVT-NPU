"""ImageNet subset datasets for open-set CLS evaluation (known classes plus a catch-all "others" class).

Vendored verbatim from tinyaibench/cls/common/imagenet_subset.py so this example uses the
same dataloader as the TinyAIBench-CLS evaluate notebook.
"""

import random
import re

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def read_manifest(root):
    manifest = json.loads((Path(root) / "dataset.json").read_text())
    if manifest["format_version"] != 2 or manifest["source_dataset"] != "ILSVRC/ImageNet":
        raise ValueError("Unsupported ImageNet-11 manifest version")
    return manifest


class PreparedDataset(Dataset):
    """Load the portable ImageNet-11 export without inferring labels from folders."""

    def __init__(self, root, manifest, split, transform=None):
        self.root = Path(root).resolve()
        self.samples = []
        self.transform = transform
        self.num_known_classes = len(manifest["selected_wnids"])
        self.others_class_idx = self.num_known_classes
        self.mapping_rows = manifest["mapping_rows"]
        for row in manifest["splits"][split]:
            path = (self.root / row["file"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Missing or invalid dataset image: {row['file']}")
            label = row["label"]
            if not isinstance(label, int) or not 0 <= label <= self.others_class_idx:
                raise ValueError(f"Invalid label: {label}")
            self.samples.append((path, label))
        if not self.samples:
            raise ValueError(f"Empty dataset split: {split}")
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _normalize_class_name(name: str) -> str:
    """Normalize separators/casing so config class names can be slightly flexible."""
    return re.sub(r"[\s_-]+", " ", name.strip().lower())


def resolve_selected_wnids(
    class_to_idx, selected_wnids=None, selected_class_names=None, categories=None
):
    """Resolve a config's `selected_wnids` or `selected_class_names` to WNIDs.

    Args:
        class_to_idx: WNID -> ImageNet class index mapping (e.g. from `ImageFolder`).
        selected_wnids: Explicit WNIDs for the known classes, in local label order.
        selected_class_names: Human-readable class names to resolve to WNIDs
            (requires `categories`, e.g. from torchvision weights metadata).
        categories: Ordered class names matching `class_to_idx`'s indices,
            used to resolve `selected_class_names`.

    Returns:
        List of WNIDs in local label order 0..N-1.

    Raises:
        ValueError: If neither or both of `selected_wnids`/`selected_class_names`
            are given, if `categories` is missing when needed, or if any name
            can't be resolved (or resolves to a duplicate class).
    """
    has_wnids = bool(selected_wnids)
    has_names = bool(selected_class_names)
    if has_wnids == has_names:
        raise ValueError("Set exactly one of selected_wnids or selected_class_names")

    if has_wnids:
        return list(selected_wnids)

    if not categories:
        raise ValueError(
            "selected_class_names requires a categories list from torchvision weights metadata"
        )

    idx_to_wnid = {idx: wnid for wnid, idx in class_to_idx.items()}
    name_to_idx = {}
    for idx, name in enumerate(categories):
        key = _normalize_class_name(str(name))
        if key not in name_to_idx:
            name_to_idx[key] = idx

    resolved = []
    missing = []
    for name in selected_class_names:
        idx = name_to_idx.get(_normalize_class_name(str(name)))
        if idx is None or idx not in idx_to_wnid:
            missing.append(name)
            continue
        resolved.append(idx_to_wnid[idx])

    if missing:
        raise ValueError(f"Could not resolve class names to WNID: {missing}")
    if len(set(resolved)) != len(resolved):
        raise ValueError("selected_class_names resolved to duplicate classes")
    return resolved


def format_class_mapping(mapping_rows):
    """Render a dataset's class mapping as a tab-separated table (plain text)."""
    lines = ["local_label\twnid\timagenet_id\tname"]
    for row in mapping_rows:
        lines.append(
            f"{row['local_label']}\t{row['wnid']}\t{row['imagenet_id']}\t{row['name']}"
        )
    return "\n".join(lines)


class ImageNetSubset(Dataset):
    """ImageFolder subset containing only the given WNIDs, relabeled 0..N-1.

    Args:
        root: ImageNet folder root with wnid class subfolders.
        selected_wnids: WNIDs to keep, in local label order 0..N-1.
        transform: Image transforms.
    """

    def __init__(self, root, selected_wnids, transform=None):
        self.base = ImageFolder(root, transform=transform)
        self.selected_wnids = list(selected_wnids)

        if not self.selected_wnids:
            raise ValueError("selected_wnids must not be empty")

        missing = [
            wnid for wnid in self.selected_wnids if wnid not in self.base.class_to_idx
        ]
        if missing:
            raise ValueError(f"Unknown class folders under {root}: {missing}")
        imagenet_ids = [self.base.class_to_idx[wnid] for wnid in self.selected_wnids]

        self.selected_imagenet_ids = imagenet_ids
        self.class_to_local = {
            int(c): i for i, c in enumerate(self.selected_imagenet_ids)
        }
        idx_to_wnid = {idx: wnid for wnid, idx in self.base.class_to_idx.items()}
        self.mapping_rows = [
            {
                "local_label": i,
                "wnid": idx_to_wnid[idx],
                "imagenet_id": idx,
                "name": "",
            }
            for i, idx in enumerate(self.selected_imagenet_ids)
        ]
        self.samples = [
            (path, self.class_to_local[label])
            for path, label in self.base.samples
            if label in self.class_to_local
        ]
        if not self.samples:
            raise ValueError(
                "No samples found after filtering. Check data root and selected_wnids mapping."
            )
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.base.loader(path)
        if self.base.transform is not None:
            sample = self.base.transform(sample)
        return sample, target


class ImageNetSubset11(Dataset):
    """11-class dataset: the first N classes are the selected ones (by WNIDs),
    and class index N is an "others" catch-all for all remaining ImageNet classes.

    Args:
        root: ImageNet folder root with wnid class subfolders.
        selected_wnids: WNIDs for the N known classes, in local label order 0..N-1.
        transform: Image transforms.
        others_max_per_class: If set, cap the number of "others" samples taken
            from each non-selected class to limit extreme class imbalance.
        seed: RNG seed used when sub-sampling "others" samples.
    """

    def __init__(
        self, root, selected_wnids, transform=None, others_max_per_class=None, seed=0
    ):
        self.base = ImageFolder(root, transform=transform)
        self.selected_wnids = list(selected_wnids)

        if not self.selected_wnids:
            raise ValueError("selected_wnids must not be empty")

        missing = [w for w in self.selected_wnids if w not in self.base.class_to_idx]
        if missing:
            raise ValueError(f"Unknown class folders under {root}: {missing}")

        selected_imagenet_ids = [self.base.class_to_idx[w] for w in self.selected_wnids]
        self.selected_imagenet_ids = selected_imagenet_ids
        self.num_known_classes = len(selected_imagenet_ids)
        self.others_class_idx = self.num_known_classes  # local label for "others"

        selected_set = set(selected_imagenet_ids)
        known_local = {iid: local for local, iid in enumerate(selected_imagenet_ids)}

        # Partition samples into known and "others" groups.
        known_samples = []
        others_by_class: dict[int, list[tuple[str, int]]] = {}
        for path, label in self.base.samples:
            if label in selected_set:
                known_samples.append((path, known_local[label]))
            else:
                others_by_class.setdefault(label, []).append(
                    (path, self.others_class_idx)
                )

        # Optionally sub-sample "others" per originating class (deterministic).
        rng = random.Random(seed)
        others_samples: list[tuple[str, int]] = []
        for class_samples in others_by_class.values():
            if (
                others_max_per_class is not None
                and len(class_samples) > others_max_per_class
            ):
                class_samples = rng.sample(class_samples, others_max_per_class)
            others_samples.extend(class_samples)

        self.samples = known_samples + others_samples
        if not self.samples:
            raise ValueError("No samples found. Check data root and selected_wnids.")

        self.targets = [label for _, label in self.samples]

        idx_to_wnid = {idx: wnid for wnid, idx in self.base.class_to_idx.items()}
        self.mapping_rows = [
            {"local_label": i, "wnid": idx_to_wnid[iid], "imagenet_id": iid, "name": ""}
            for i, iid in enumerate(self.selected_imagenet_ids)
        ]
        self.mapping_rows.append(
            {
                "local_label": self.others_class_idx,
                "wnid": "others",
                "imagenet_id": -1,
                "name": f"others ({len(others_by_class)} source classes, {len(others_samples)} samples)",
            }
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.base.loader(path)
        if self.base.transform is not None:
            sample = self.base.transform(sample)
        return sample, target
