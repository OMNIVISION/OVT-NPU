#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOX postprocess: grid decode + class-agnostic NMS.

This is the reference implementation of everything the NPU graph does NOT do.
The compiled model stops at 9 raw feature maps -- (reg, obj, cls) x strides
8/16/32 -- because grid decode needs exp() and NMS needs dynamic shapes.  This
module picks up from there, and is the Python twin of the RISC-V firmware
post-process in pp/pp_yolox.c.  Keep the two in step.

QdYoloxOutputs groups the 9 tensors by level whatever order the model returns
them in, and decode_batch() runs the whole post-process for one batch of the flow.

Feature-map order is stride 8 -> 16 -> 32 (largest -> smallest), because
demo_postprocess builds its grids in that order.

Batching
--------
`__call__` accepts a batch and returns per-image results, since NMS output is
variable-length and cannot be stacked:

    b, s, c = pp(outs, img_height=480, img_width=640)        # scalars + B == 1
    per_img = pp(outs, img_height=[480, 720],                # -> [(b,s,c), ...]
                       img_width =[640, 1280])

Scalar image dims with B == 1 return the 3-tuple of arrays; anything else
returns a length-B list of such tuples.  Scalar dims with B > 1 broadcast the
same size to every image; a sequence whose length is not B raises.

Return, per image:
  boxes   (N, 4) float32  xyxy in ORIGINAL-image coordinates (ratio already undone)
  scores  (N,)   float32  best class score, obj * cls
  cls_ids (N,)   int64    best class index
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


class PostProc_YoloX:
    def __init__(
        self,
        input_shape=(1, 3, 192, 288),
        num_classes=2,
        nms_thr=0.65,
        score_thr=0.25,
        channel_first=True,
        exec_concat=False,
        exec_sigmoid=False,
        save_fmap=False,
        class_agnostic=True,
    ):
        self.input_shape = input_shape
        self.channel_first = channel_first
        if self.channel_first:
            self.input_height = self.input_shape[2]
            self.input_width = self.input_shape[3]
        else:
            self.input_height = self.input_shape[1]
            self.input_width = self.input_shape[2]

        self.num_classes = int(num_classes)
        self.strides = [8, 16, 32]
        self.nms_thr = float(nms_thr)
        self.score_thr = float(score_thr)
        self.exec_concat = bool(exec_concat)
        self.sigmoid_flag = bool(exec_sigmoid)
        self.save_fmap = bool(save_fmap)
        # Class-agnostic NMS is what the RISC-V post-process does; per-class NMS is what
        # YOLOX's own COCO evaluator does, so COCO AP is comparable with its results.
        self.class_agnostic = bool(class_agnostic)

    def __call__(self, outputs, img_height, img_width, score_thr=None, nms_thr=None):
        # Per-call overrides stay local.  They used to be written onto self, so a
        # single call with score_thr=0.01 silently re-thresholded every later call
        # made through the same instance -- and coco_eval reuses one instance
        # across the whole dataset.
        score_thr = self.score_thr if score_thr is None else float(score_thr)
        nms_thr = self.nms_thr if nms_thr is None else float(nms_thr)

        if self.sigmoid_flag:
            # Was accepted and silently ignored: stored as self.sigmoid_flag and
            # never read.  Every caller passes False because the graph applies
            # sigmoid to obj/cls itself.  Fail loudly rather than lie.
            raise NotImplementedError(
                "exec_sigmoid=True is not implemented; obj/cls must already be "
                "probabilities (the exported graph applies sigmoid internally)"
            )

        predictions = self._as_predictions(outputs)  # (B, sum(HW), 5+K)
        batch = predictions.shape[0]

        # np.ndim treats scalars as 0-d and handles lists, ndarrays and torch
        # tensors alike (info_imgs from a torch DataLoader arrives as tensors).
        scalar_dims = np.ndim(img_height) == 0 and np.ndim(img_width) == 0
        heights = self._image_dims(img_height, batch, "img_height")
        widths = self._image_dims(img_width, batch, "img_width")

        if self.save_fmap and batch > 1:
            raise ValueError(
                "save_fmap writes fixed filenames and cannot be used with a batch "
                f"(got batch={batch})"
            )

        predictions = self.demo_postprocess(
            predictions, (self.input_height, self.input_width), p6=False
        )

        results = []
        for i in range(batch):
            ratio = min(
                self.input_height / heights[i], self.input_width / widths[i]
            )
            results.append(self._decode_one(predictions[i], ratio, score_thr, nms_thr))

        # Backwards compatible: the single-image call is unchanged.
        if batch == 1 and scalar_dims:
            return results[0]
        return results

    @staticmethod
    def _image_dims(value, batch, name):
        """Normalise img_height / img_width to a per-image list of floats."""
        if np.ndim(value) > 0:
            dims = [float(v) for v in value]
            if len(dims) != batch:
                raise ValueError(
                    f"{name} has {len(dims)} entries but the batch is {batch}"
                )
            return dims
        return [float(value)] * batch

    def _as_predictions(self, outputs):
        """Flatten the per-scale feature maps into (B, sum(HW), 5+K)."""
        if self.exec_concat:
            # outputs: per-scale (B, 5+K, H, W), already merged by the caller
            flattened = [x.reshape(x.shape[0], x.shape[1], -1) for x in outputs]
            merged = np.concatenate(flattened, axis=2)
        else:
            # outputs: 9 tensors (reg,obj,cls) * 3 scales, but order may vary.
            # Group by spatial size and identify by channel count.
            by_hw = {}
            for x in outputs:
                if x.ndim != 4:
                    raise ValueError(f"Expected 4D tensor, got shape={x.shape}")
                hw = (x.shape[2], x.shape[3])
                by_hw.setdefault(hw, []).append(x)

            grouped_outputs = []
            # IMPORTANT: process large->small feature map to match strides=[8,16,32]
            for h, w in sorted(by_hw.keys(), key=lambda t: t[0] * t[1], reverse=True):
                group = by_hw[(h, w)]
                reg = next((t for t in group if t.shape[1] == 4), None)
                obj = next((t for t in group if t.shape[1] == 1), None)
                cls = next((t for t in group if t.shape[1] not in (1, 4)), None)
                if reg is None or obj is None or cls is None:
                    shapes = [g.shape for g in group]
                    raise ValueError(
                        f"Cannot identify (reg,obj,cls) for (H,W)=({h},{w}). shapes={shapes}"
                    )
                # NOTE: self.num_classes is deliberately NOT updated from the data
                # here.  It used to be, which mutated the object as a side effect of
                # a read-only call.  Nothing in the decode reads it: scores use
                # predictions[:, 5:], i.e. whatever class channels are present.
                merged = np.concatenate([reg, obj, cls], axis=1)  # (B, 5+K, H, W)
                grouped_outputs.append(
                    merged.reshape(merged.shape[0], merged.shape[1], -1)
                )

            merged = np.concatenate(grouped_outputs, axis=2)

        return merged.transpose(0, 2, 1)

    def _decode_one(self, predictions, ratio, score_thr, nms_thr):
        """Threshold + NMS for a single image. predictions is (sum(HW), 5+K)."""
        if self.save_fmap:
            np.savetxt("predictions.csv", predictions[:, 4:5].flatten())
            np.savetxt("conf.csv", predictions[:, 5:].flatten())
            np.savetxt("box.csv", predictions[:, :4].flatten())

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]  # obj * cls_conf, (N, K)

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr, score_thr)
        if dets is None:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return (
            dets[:, :4].astype(np.float32),
            dets[:, 4].astype(np.float32),
            dets[:, 5].astype(np.int64),
        )

    def demo_postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        hsizes = [img_size[0] // stride for stride in self.strides]
        wsizes = [img_size[1] // stride for stride in self.strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, self.strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)

        # Copy first: this used to write through into the caller's array.  The
        # writes stay in-place on the copy so the result keeps the input dtype --
        # grids are int64, so an out-of-place expression would silently promote
        # float32 predictions to float64 and change the last bits.
        outputs = np.array(outputs, copy=True)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        if self.class_agnostic:
            keep = self.nms(valid_boxes, valid_scores, nms_thr)
        else:
            # A box only suppresses boxes of its own class
            keep = []
            for cls in np.unique(valid_cls_inds):
                members = np.flatnonzero(valid_cls_inds == cls)
                keep.extend(members[self.nms(valid_boxes[members], valid_scores[members], nms_thr)])
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
            return dets
        return None

    def nms(self, boxes, scores, nms_thr):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        # NOTE: the +1 box-area convention is load-bearing -- pp/pp_yolox.c:130-141
        # uses the same one, which is what lets deploy.py compare them.
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]
        return keep


class QdYoloxOutputs:
    """Groups the QD model's flat output list into per-level (reg, obj, cls).
    """

    def __init__(
        self,
        num_classes: int,
        strides: Sequence[int] = (8, 16, 32),
        channels_last: bool = False,
    ):
        self.num_classes = int(num_classes)
        self.strides = list(strides)
        # channels_last is needed for simulator output, which is NHWC after
        # QInterleave.  Unused on the torch QD path.
        self.channels_last = bool(channels_last)

    def _spatial(self, t: torch.Tensor) -> Tuple[int, int]:
        return (t.shape[1], t.shape[2]) if self.channels_last else (t.shape[2], t.shape[3])

    def _channels(self, t: torch.Tensor) -> int:
        return t.shape[-1] if self.channels_last else t.shape[1]

    def group(self, outputs: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return [(reg, obj, cls), ...] ordered large -> small feature map."""
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(
                "Expected the QD model to return a list of tensors, got "
                f"{type(outputs).__name__}. The converted model must be a "
                "multi-output QSequential."
            )
        expected = 3 * len(self.strides)
        if len(outputs) != expected:
            raise ValueError(
                f"Expected {expected} output tensors ((reg, obj, cls) x "
                f"{len(self.strides)} levels), got {len(outputs)} with shapes "
                f"{[tuple(o.shape) for o in outputs]}"
            )

        buckets: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        for t in outputs:
            if t.dim() != 4:
                raise ValueError(f"Expected a 4D tensor, got shape {tuple(t.shape)}")
            buckets.setdefault(self._spatial(t), []).append(t)

        if len(buckets) != len(self.strides):
            raise ValueError(
                f"Expected {len(self.strides)} distinct feature-map sizes, got "
                f"{sorted(buckets)}"
            )

        levels = []
        for hw in sorted(buckets, key=lambda s: s[0] * s[1], reverse=True):
            group = buckets[hw]
            reg = next((t for t in group if self._channels(t) == 4), None)
            obj = next((t for t in group if self._channels(t) == 1), None)
            cls = next((t for t in group if self._channels(t) not in (1, 4)), None)
            if reg is None or obj is None or cls is None:
                raise ValueError(
                    f"Cannot identify (reg, obj, cls) at (H, W)={hw}; "
                    f"shapes={[tuple(t.shape) for t in group]}"
                )
            if self._channels(cls) != self.num_classes:
                raise ValueError(
                    f"cls tensor at (H, W)={hw} has {self._channels(cls)} channels, "
                    f"expected num_classes={self.num_classes}"
                )
            levels.append((reg, obj, cls))
        return levels

    def merged(self, outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Return [B, 5+K, H, W] per level -- YOLOX's per-level channel layout."""
        return [torch.cat(level, 1) for level in self.group(outputs)]


def decode_batch(outputs, targets, postproc: PostProc_YoloX, adapter: QdYoloxOutputs) -> List[dict]:
    """
    Decodes one batch of raw heads into one detection dict per image.

    Args:
        outputs: The model's 9 output tensors, any order
        targets: One dict per image, whose 'orig_size' is (height, width) of the source
        postproc: Grid decode + NMS
        adapter: Groups the 9 tensors into levels
    Returns:
        [{'boxes' [N, 4] xyxy in original pixels, 'scores' [N], 'labels' [N]}, ...]
    """
    merged = [level.detach().float().cpu().numpy() for level in adapter.merged(list(outputs))]
    heights = [float(target["orig_size"][0]) for target in targets]
    widths = [float(target["orig_size"][1]) for target in targets]
    return [
        {"boxes": torch.from_numpy(boxes), "scores": torch.from_numpy(scores), "labels": torch.from_numpy(labels)}
        for boxes, scores, labels in postproc(merged, img_height=heights, img_width=widths)
    ]


def dump_python_postprocess_result(outputs, postproc: PostProc_YoloX, adapter: QdYoloxOutputs, output_folder):
    """
    The Python twin of dump_c_postprocess_result, for where the C path is off (Windows).
    Decodes one image's raw heads and saves result.csv in the same layout: x1, y1, x2,
    y2 in model-input pixels, then score and class.

    Args:
        outputs: The model's 9 output tensors, batch of one, any order
        postproc: Grid decode + NMS, class-agnostic at conf_thresh as the C post-process does
        adapter: Groups the 9 tensors into levels
        output_folder: Where result.csv goes
    """
    merged = [level.detach().float().cpu().numpy() for level in adapter.merged(list(outputs))]
    # Model-input dims make the rescale ratio 1, so boxes stay in model-input pixels
    boxes, scores, labels = postproc(merged, img_height=postproc.input_height, img_width=postproc.input_width)
    result = np.concatenate([boxes, scores.reshape(-1, 1), labels.reshape(-1, 1)], 1)
    np.savetxt(f"{output_folder}/result.csv", result, delimiter=",")
    print(np.array2string(result, precision=3, suppress_small=True))
    return result
