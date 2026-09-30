"""
The converted Quartz OVQ model exports only the model's eight raw head ouputs, and output decoding occurs outside the model.
Upstream does the decode inside the model (SSD.forward(is_test=True)) and finishes it in Predictor.predict(); 
neither is usable here. This file re-assembles the same steps outside the model, over a whole batch.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# The model's fixed properties. 
CFG = {
    "image_size": [160, 120],
    "image_mean": np.array([127, 127, 127]),
    "image_std": 128.0,
    "iou_threshold": 0.3,
    "center_variance": 0.1,
    "size_variance": 0.2,
    "min_boxes": [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    "feature_map_w_h_list": [[20, 10, 5, 3], [15, 8, 4, 2]],
}

def generate_priors(feature_map_list, image_size, min_boxes, clamp=True) -> torch.Tensor:
    shrinkage_list = []
    for i in range(0, len(image_size)):
        item_list = []
        for k in range(0, len(feature_map_list[i])):
            item_list.append(image_size[i] / feature_map_list[i][k])
        shrinkage_list.append(item_list)
    
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    print("priors nums:{}".format(len(priors)))
    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors

def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat(
        [
            locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
            torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:],
        ],
        dim=locations.dim() - 1,
    )

def postprocess(priors, outputs, prob_threshold, iou_threshold, top_k, candidate_size, device, layout="nchw"):
    """Decode raw SSD heads into per-image detections

    Args:
        priors: (num_priors, 4) center-form priors
        outputs: the model's heads, four confidence tensors then four location
            tensors, in the layout named by `layout`
        prob_threshold: minimum face probability to keep a candidate
        iou_threshold: NMS overlap above which a lower scoring box is suppressed
        top_k, candidate_size: NMS limits
        layout: "nchw" or "nhwc", the layout the heads arrive in. Ovq model before npu_optimisation() ouputs NCHW, 
        and after npu_optimization() outputs NHWC.

    Returns:
        A list of (boxes, probs) pairs, one per image. boxes is (num_detections, 4)
        xyxy normalised to [0, 1], probs is (num_detections,). An image with no
        detection contributes a pair of empty tensors.
    """
    head_count = len(outputs) // 2
    confidences = []
    locations = []

    for confidence, location in zip(outputs[:head_count], outputs[head_count:]):
        if layout == "nchw":
            #permute to NHWC for the rest of the code, which is written for NHWC
            confidence = confidence.permute(0, 2, 3, 1)
            location = location.permute(0, 2, 3, 1)

        confidence = confidence.contiguous()
        location = location.contiguous()
        confidences.append(confidence.view(confidence.size(0), -1, 2))
        locations.append(location.view(location.size(0), -1, 4))

    confidences = torch.cat(confidences, 1)
    locations = torch.cat(locations, 1)

    confidences = F.softmax(confidences, dim=2)
    boxes = convert_locations_to_boxes(
        locations, priors, CFG["center_variance"], CFG["size_variance"]
    )
    boxes = torch.cat([boxes[..., :2] - boxes[..., 2:] / 2,
                boxes[..., :2] + boxes[..., 2:] / 2], boxes.dim() - 1)

    boxes = boxes.to(device)
    confidences = confidences.to(device)

    batch_size = boxes.size(0)
    empty = (torch.zeros((0, 4), device=device), torch.zeros((0,), device=device))
    results = [empty] * batch_size

    # Gather every image's surviving candidates into one flat set, tagged with the
    # image it came from, so NMS runs once for the batch instead of once per image.
    flat_boxes = []
    flat_scores = []
    flat_image = []

    for index in range(batch_size):
        probs = confidences[index][:, 1]
        mask = probs > prob_threshold
        if not bool(mask.any()):
            continue

        image_boxes = boxes[index][mask]
        probs = probs[mask]

        # Same as the greedy implementation: only the highest scoring
        # candidate_size boxes are considered at all.
        if candidate_size and probs.numel() > candidate_size:
            probs, order = probs.topk(candidate_size)
            image_boxes = image_boxes[order]

        flat_boxes.append(image_boxes)
        flat_scores.append(probs)
        flat_image.append(torch.full((probs.numel(),), index, dtype=torch.long, device=device))

    if not flat_boxes:
        return results

    flat_boxes = torch.cat(flat_boxes)
    flat_scores = torch.cat(flat_scores)
    flat_image = torch.cat(flat_image)

    # batched_nms offsets the boxes per image so that boxes from different images
    # can never suppress one another, then runs a single fused NMS kernel. The
    # returned indices are ordered by descending score.
    keep = torchvision.ops.batched_nms(flat_boxes, flat_scores, flat_image, iou_threshold)

    kept_image = flat_image[keep]
    for index in range(batch_size):
        selected = keep[kept_image == index]
        # top_k <= 0 means keep everything, matching the greedy implementation
        if top_k and top_k > 0:
            selected = selected[:top_k]
        if selected.numel():
            results[index] = (flat_boxes[selected], flat_scores[selected])

    return results

def decode_batch(priors, outputs, labels, config, device, layout="nchw"):
    """
    Decodes one batch of raw SSD heads into per-image detections.

    Returns:
        List of {'boxes', 'scores', 'labels'} with boxes normalised to [0, 1]
    """
    decoded = postprocess(
        priors,
        outputs,
        config["conf_thresh"],
        config["nms_thresh"],
        config["top_k"],
        config["candidate_size"],
        device,
        layout=layout,
    )

    detections = []
    for boxes, probs in decoded:
        if boxes is None or boxes.numel() == 0:
            detections.append(
                {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long),
                },
            )
        else:
            detections.append(
                {
                    "boxes": boxes.detach().cpu(),
                    "scores": probs.detach().cpu().reshape(-1),
                    "labels": torch.ones(boxes.shape[0], dtype=torch.long),
                },
            )

    #scale the boxes to the original image size, as the official WIDER ground truth uses [0, 1] coordinates
    for prediction, ground_truth in zip(detections, labels):
        width, height = ground_truth["orig_size"].tolist()
        scale = torch.tensor([width, height, width, height], dtype=torch.float32)
        prediction["boxes"] = prediction["boxes"] * scale

    return detections

def postprocess_dump_result_csv(self, output_tensor, label, model, *args, **kwargs):
    """
    Writes the decoded detections to result.csv, one row per box.
    """
    detection, target = output_tensor[0], label[0]
    width, height = target["orig_size"].tolist()
    scale = np.array([width, height, width, height], dtype=np.float32)

    boxes = detection["boxes"].cpu().numpy() / scale
    scores = detection["scores"].cpu().numpy().reshape(-1, 1)
    result = np.c_[boxes, scores]
    result = result[np.lexsort((result[:, 1], result[:, 0]))]

    np.savetxt(f"{kwargs['output_folder']}/result.csv", result, delimiter=",")
    print(np.array2string(result, precision=3, suppress_small=True))
    return output_tensor

def dump_c_postprocess_result(output_tensor, config, output_folder):
    """
    Runs the firmware post-process (pp_ulfgfd_ctypes.postprocess_c) on one image's raw
    heads, as the device does, and writes its detections to result.csv in the same
    layout as postprocess_dump_result_csv. That file is what the device's output is
    compared against.

    Args:
        output_tensor: The model's eight raw NHWC heads, batch of one, confs then locs
        config (dict): Model config: input_shape, conf_thresh, nms_thresh
        output_folder: Where result.csv goes
    """
    from .pp_ulfgfd_ctypes import postprocess_c

    heads = [tensor[0].cpu().numpy() for tensor in output_tensor]
    _, _, height, width = config["input_shape"]
    # postprocess_c takes (locs, confs); the model emits four confs then four locs
    detections = postprocess_c(
        heads[4:],
        heads[:4],
        config["conf_thresh"],
        config["nms_thresh"],
        input_hw=(height, width),
    )

    # C returns model-input pixels, result.csv holds [0, 1] boxes
    scale = np.array([width, height, width, height], dtype=np.float32)
    result = np.c_[detections[:, :4] / scale, detections[:, 4]]
    result = result[np.lexsort((result[:, 1], result[:, 0]))]

    np.savetxt(f"{output_folder}/result.csv", result, delimiter=",")
    print(np.array2string(result, precision=3, suppress_small=True))
    return result
