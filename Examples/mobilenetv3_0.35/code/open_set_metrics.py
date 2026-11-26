"""Open-set (known classes + "others") classification metrics for the ROPT flow.

Computes the same numbers as the TinyAIBench-CLS evaluate notebook
(tinyaibench/cls/common/loops.py, evaluate_open_set_mixed), so the FP32, calibrated,
RON-optimized and simulated models can be compared against its report.
"""
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from quartz.modelkit import BaseMetrics


class OpenSetMetrics(BaseMetrics):
    """
    Buffers the logits of a pass and scores them at once.

    Targets use local labels 0..num_known_classes-1 for the known classes and
    num_known_classes for the catch-all "others" class.
    """

    def __init__(self, num_known_classes: int, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self._num_known = num_known_classes
        self.reset()

    def reset(self) -> None:
        self._logits = []
        self._targets = []
        self._memo = None

    def update(self, output: torch.Tensor, target: torch.Tensor | int, **kwargs: Any) -> None:
        targets = torch.as_tensor(target).reshape(-1).cpu().long()
        num_classes = self._num_known + 1
        # RON and simulator outputs may retain singleton spatial dimensions:
        # (C, 1, 1) or (N, C, 1, 1), rather than the FP32 model's (N, C).
        # The configured class count, not the last dimension, defines a row.
        if output.numel() != targets.numel() * num_classes:
            raise ValueError(
                f"Expected {targets.numel()} sample(s) with {num_classes} logits each, "
                f"got output shape {tuple(output.shape)}"
            )
        logits = output.detach().float().cpu().reshape(targets.numel(), num_classes)
        self._logits.append(logits)
        self._targets.append(targets)
        self._memo = None

    def running(self) -> Dict[str, float]:
        if not self._logits:
            return {}
        logits, targets = torch.cat(self._logits), torch.cat(self._targets)
        return {"top1": 100.0 * (logits.argmax(1) == targets).float().mean().item()}

    def compute(self) -> Dict[str, float]:
        if self._memo is not None:
            return self._memo
        if not self._logits:
            return {}

        logits, targets = torch.cat(self._logits), torch.cat(self._targets)
        c = self._num_known
        preds = logits.argmax(1)
        top5 = logits.topk(min(5, logits.shape[1]), 1).indices
        known = targets < c
        others = targets == c
        n_known, n_others = int(known.sum()), int(others.sum())

        m = {
            "top1": 100.0 * (preds == targets).float().mean().item(),
            "top5": 100.0 * (top5 == targets[:, None]).any(1).float().mean().item(),
            "top1_known": 100.0 * (preds[known] == targets[known]).float().mean().item() if n_known else 0.0,
            "sample_count_total": int(targets.numel()),
            "sample_count_known": n_known,
            "sample_count_others": n_others,
            "recall_others": 100.0 * (preds[others] == c).float().mean().item() if n_others else 0.0,
            "frr_known_to_others": 100.0 * (preds[known] == c).float().mean().item() if n_known else 0.0,
        }
        m["far_others_to_known"] = 100.0 - m["recall_others"]

        if n_known and n_others:
            # Knownness = 1 - P(others); higher means "more likely a known class"
            knownness = (1.0 - torch.softmax(logits, 1)[:, c]).numpy()
            is_known = known.numpy().astype(int)
            m["auroc_known_vs_others"] = float(roc_auc_score(is_known, knownness))
            precision, recall, _ = precision_recall_curve(is_known, knownness)
            m["aupr_known_vs_others"] = float(auc(recall, precision))
            fpr, tpr, _ = roc_curve(is_known, knownness)
            m["fpr95_known_vs_others"] = 100.0 * float(fpr[(tpr >= 0.95).argmax()])
            m["oscr"] = self._oscr(knownness, known.numpy(), others.numpy(), (preds == targets).numpy())

        self._memo = m
        return m

    @staticmethod
    def _oscr(knownness, known_mask, others_mask, correct) -> float:
        """Area under correct-known-accept-rate vs others-false-accept-rate as the threshold sweeps."""
        order = knownness.argsort()[::-1]
        scores = knownness[order]
        group_end = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
        ccr = (known_mask & correct)[order].cumsum()[group_end] / known_mask.sum()
        fpr = others_mask[order].cumsum()[group_end] / others_mask.sum()
        return float(auc(np.r_[0.0, fpr], np.r_[0.0, ccr]))

    def report(self) -> None:
        metrics = self.compute()
        print(f"Open-set metrics ({metrics.get('sample_count_total', 0)} samples):")
        for name, value in metrics.items():
            print(f"  {name:<24} {self.format_value(value)}")
