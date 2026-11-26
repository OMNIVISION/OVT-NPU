"""Datasets and scoring for multi-label MobileNetV2."""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from quartz.modelkit import BaseMetrics

from .openimages_multilabel import OpenImagesMultiLabel, map


class OpenImagesRopt(OpenImagesMultiLabel):
    """
    The label caches of prepare_openimages_multilabel.py as (image, label) pairs, the shape
    BaseInference and BaseTraining iterate. The label is a dict of the multi-hot
    target, its verified-label mask and the background-image flag.
    """

    def __getitem__(self, i):
        image, target, mask = super().__getitem__(i)
        return image, {"target": target, "mask": mask, "is_background": self.is_background[i]}

class ImageListDataset(Dataset):
    """Unlabelled images, as (image, 0) pairs."""

    def __init__(self, files, transform) -> None:
        self.files = [str(f) for f in files]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as image:
            image = image.convert("RGB")
        return self.transform(image), 0

class MultiLabelMetrics(BaseMetrics):
    """
    update() takes the model output as it is, class probabilities (the model ends in a
    sigmoid), so the same object serves the inference stages and the QAT loop, which
    hands over its outputs before any postprocessing.
    """

    def __init__(self, classes, num_object_classes, thresholds=None, verbose=False) -> None:
        super().__init__(verbose=verbose)
        self.classes = list(classes)
        self.num_object_classes = num_object_classes
        self.thresholds = thresholds
        self.reset()

    def reset(self) -> None:
        self._probs, self._target, self._mask, self._is_background = [], [], [], []
        self._result = None

    def update(self, output, target, **kwargs) -> None:
        self._probs.append(output.detach().float().reshape(output.shape[0], -1).cpu())
        self._target.append(target["target"].cpu())
        self._is_background.append(target["is_background"].cpu())
        self._result = None

    def compute(self) -> dict:
        if self._result is not None or not self._probs:
            return self._result or {}

        from sklearn.metrics import roc_auc_score

        probs, target = (torch.cat(t) for t in (self._probs, self._target))
        is_background = torch.cat(self._is_background).bool()
        k = self.num_object_classes

        _, aps = map(probs, target)
        aucs = []
        for c in range(len(self.classes)):
            y, s = target[:, c].numpy(), probs[:, c].numpy()
            p = y.mean() if len(y) else float("nan")
            both = 0 < y.sum() < len(y)
            aucs.append(roc_auc_score(y, s) if both else float("nan"))

        result = {
            "mAP": float(np.nanmean(aps[:k])),
            "mean_AUROC": float(np.nanmean(aucs[:k])),
        }
        # if is_background.any():
        #     result["bg_fpr@0.5"], _ = background_fpr(probs, [0.5] * probs.shape[1], is_background, k)
        #     if self.thresholds is not None:
        #         result["bg_fpr@tuned"], _ = background_fpr(probs, self.thresholds, is_background, k)
        for name, ap, auc in zip(self.classes, aps, aucs):
            result[f"AP/{name}"] = float(ap)
            result[f"AUROC/{name}"] = float(auc)
        self._result = result
        return result

    def report(self) -> None:
        result = self.compute()
        if not result:
            print("No samples were scored")
            return
        print(f'{"class":<12}{"AP":>8}{"AUROC":>8}')
        for name in self.classes:
            print(f"{name:<12}"
                  + "".join(f"{self.format_value(result[f'{m}/{name}']):>8}"
                            for m in ("AP", "AUROC")))
        for name in ("mAP", "mean_AUROC"):
            if name in result:
                print(f"{name}: {self.format_value(result[name])}")

