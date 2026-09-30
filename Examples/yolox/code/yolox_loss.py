'''
# After ONNX -> QD conversion the YOLOX model is a flat QSequential: there is no
# .backbone, no .head, and no loss inside the forward.  The graph emits 9 raw
# tensors -- (reg, obj, cls) x strides 8/16/32 -- with sigmoid ALREADY APPLIED to
# obj and cls inside the graph, and with grid-decode / exp() / NMS all left
# outside.  This module supplies everything the graph does not:
#
#   YoloxQatLossHead  SimOTA + IoU/BCE/L1 loss over those tensors
#
# (QdYoloxOutputs, which groups the 9 tensors into 3 levels, lives in
# postprocess_yolox.py: it is plain torch, and the decode needs it without YOLOX.)
#
# On the sigmoid: YOLOX's loss assumes logits in exactly two places --
# `bcewithlog_loss` (yolo_head.py:386-393) and the .sigmoid_() calls in
# get_assignments (yolo_head.py:476).  Rather than strip the sigmoid from the
# graph (would diverge from the deployed model graph), we
# adapt the BCEProbLoss: 

# z is the pre-sigmoid activation (the logit) and p = σ(z) is the post-sigmoid probability; t is the target. In this codebase specifically:
# - p — what the graph actually hands you: the obj and cls tensors among the 9 outputs, sigmoid already applied.
# - t — from get_assignments: fg_mask for obj, one_hot(cls) * IoU for cls.

#     dBCE/dp * dp/dz = ((p - t) / (p(1-p))) * p(1-p) = (p - t)

# which is precisely what BCEWithLogitsLoss produces.  Gradients through the
# in-graph sigmoid are therefore unchanged.

# Breaking down:
# Factor 1 — dBCE/dp. With BCE(p,t) = −[t·log p + (1−t)·log(1−p)]:
# dBCE/dp = −[ t/p − (1−t)/(1−p) ]
#         = −[ t(1−p) − p(1−t) ] / [p(1−p)]
#         = −[ t − tp − p + pt ] / [p(1−p)]
#         = −(t − p) / [p(1−p)]
#         = (p − t) / (p(1−p))
# Factor 2 — dp/dz. With σ(z) = (1+e^{−z})^{−1}:
# dσ/dz = e^{−z} / (1+e^{−z})²
#       = [1/(1+e^{−z})] · [e^{−z}/(1+e^{−z})]
#       = σ(z) · (1 − σ(z))          since e^{−z}/(1+e^{−z}) = 1 − σ(z)
#       = p(1−p)
# Chain rule. The p(1−p) cancels:
# dBCE/dz = (p − t)/(p(1−p)) · p(1−p) = p − t
'''

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.models.losses import IOUloss
from yolox.models.yolo_head import YOLOXHead
from yolox.utils import meshgrid

from .postprocess_yolox import QdYoloxOutputs

class ProbBCELoss(nn.Module):
    """BCE on probabilities. Drop-in replacement for `bcewithlog_loss`.

    The quantized sigmoid outputs land exactly on the quantizer's grid, 
    so p can be exactly 0.0 or 1.0 (0.00070 -> 0.00000 at qlen=8), 
    which would make log(p) = -100 (instead of -inf, clamped by Pytorch). This would blow up the gradients, thus add clamps.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(
            p.clamp(self.eps, 1.0 - self.eps), t, reduction="none"
        )


class YoloxQatLossHead(YOLOXHead):
    """YOLOX loss adapted

    __init__ deliberately bypasses YOLOXHead.__init__: we
    want none of its convolutions (they live in the QD graph now), and building
    them would fail anyway -- the exp uses act="relu6", which the vendored
    upstream get_activation rejects (network_blocks.py:26).
    """

    def __init__(
        self,
        num_classes: int,
        strides: Sequence[int] = (8, 16, 32),
        use_l1: bool = True,
        eps: float = 1e-6,
    ):
        nn.Module.__init__(self)  # NOT YOLOXHead.__init__ -- see docstring

        self.num_classes = int(num_classes)
        self.strides = list(strides)
        self.grids = [torch.zeros(1)] * len(self.strides)
        self.use_l1 = bool(use_l1)
        self.decode_in_inference = False

        self.l1_loss = nn.L1Loss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        # Keep the upstream attribute NAME so get_losses:386-393 needs no edit.
        self.bcewithlog_loss = ProbBCELoss(eps=eps)
        self._eps = float(eps)

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):
        """SimOTA assignment, fed logits recovered from our probabilities.

        Upstream applies .sigmoid_() to cls_preds/obj_preds (yolo_head.py:476).
        Our tensors are already post-sigmoid, so we invert first and let the
        parent round-trip them back.
        """

        def to_logit(p: torch.Tensor) -> torch.Tensor:
            return torch.logit(p.clamp(self._eps, 1.0 - self._eps))

        return super().get_assignments(
            batch_idx,
            num_gt,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            to_logit(cls_preds),
            to_logit(obj_preds),
            mode,
        )

    def get_output_and_grid(self, output, k, stride, dtype):
        """Device-correct replacement for YOLOXHead.get_output_and_grid.

        Upstream (yolo_head.py:215-233) builds the grid on CPU and moves it with
        `.type(dtype)`, where dtype is the legacy type string
        "torch.cuda.FloatTensor".  That string carries no device *index*, so the
        grid lands on the current CUDA device (cuda:0) while the model may be on
        cuda:1 -- `--device` defaults to 1 (workflow_configs.py:131) and
        base_model_manager.py:80 builds f"cuda:{device}".  Upstream then caches
        the wrong-device grid in self.grids[k], and its rebuild guard is
        shape-only (yolo_head.py:221), so it never recovers.

        This version builds the grid directly on output.device and includes the
        device in the cache guard.  It also replaces the in-place slice writes at
        yolo_head.py:231-232 with a cat: `output` descends from a fake-quant
        graph here, and in-place mutation of a view is a needless autograd risk
        for no gain.

        `dtype` is accepted and ignored, to keep the parent's signature.
        """
        del dtype

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]

        grid = self.grids[k]
        if grid.shape[2:4] != output.shape[2:4] or grid.device != output.device:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = (
                torch.stack((xv, yv), 2)
                .view(1, 1, hsize, wsize, 2)
                .to(device=output.device, dtype=output.dtype)
            )
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output = torch.cat(
            [
                (output[..., :2] + grid) * stride,
                torch.exp(output[..., 2:4]) * stride,
                output[..., 4:],
            ],
            dim=-1,
        )
        return output, grid

    def loss_from_qd_outputs(
        self,
        qd_outputs: List[torch.Tensor],
        labels: torch.Tensor,
        adapter: QdYoloxOutputs,
    ) -> Dict[str, torch.Tensor]:
        """Compute YOLOX losses from the QD model's 9 output tensors.

        Mirrors the training branch of YOLOXHead.forward (yolo_head.py:150-203),
        with the per-level convolutions replaced by tensors the graph produced.

        Args:
            qd_outputs: the list returned by the QD QSequential.
            labels: [B, max_labels, 5] as (cls, cx, cy, w, h) in input pixels.
            adapter: groups qd_outputs into levels.

        Returns:
            dict with total_loss / iou_loss / conf_loss / cls_loss / l1_loss / num_fg.
        """
        levels = adapter.group(qd_outputs)

        type_str = qd_outputs[0].type()
        torch_dtype = qd_outputs[0].dtype

        outputs = []
        origin_preds = []
        x_shifts, y_shifts, expanded_strides = [], [], []

        for k, ((reg, obj, cls), stride) in enumerate(zip(levels, self.strides)):
            if self.use_l1:
                # Snapshot the RAW reg output before decode -- L1 is computed in
                # grid/log space (yolo_head.py:174-184).
                b, _, h, w = reg.shape
                origin_preds.append(
                    reg.view(b, 1, 4, h, w).permute(0, 1, 3, 4, 2).reshape(b, -1, 4).clone()
                )

            output = torch.cat([reg, obj, cls], 1)
            output, grid = self.get_output_and_grid(output, k, stride, type_str)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1]).fill_(stride).type_as(reg)
            )
            outputs.append(output)

        (
            loss,
            iou_loss,
            conf_loss,
            cls_loss,
            l1_loss,
            num_fg,
        ) = self.get_losses(
            None,  
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            torch.cat(outputs, 1),
            origin_preds,
            dtype=torch_dtype,
        )

        return {
            "total_loss": loss,
            "iou_loss": iou_loss,
            "conf_loss": conf_loss,
            "cls_loss": cls_loss,
            "l1_loss": l1_loss,
            "num_fg": num_fg,
        }
