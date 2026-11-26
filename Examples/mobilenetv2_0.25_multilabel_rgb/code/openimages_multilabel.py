"""Open Images multi-label dataset, transforms, and scoring helpers."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

try:
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:          # Pillow < 9.1
    BILINEAR = Image.BILINEAR

TARGET_SIZE = (160, 120)                  # (width, height), 4:3 landscape
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_FILL = tuple(round(m * 255) for m in IMAGENET_MEAN)   # (124, 116, 104)


def letterbox_scale(w, h, size=TARGET_SIZE):
    """Scale factor that fits a w x h image inside size=(W, H)."""
    W, H = size
    return np.minimum(W / np.asarray(w, dtype=float), H / np.asarray(h, dtype=float))

class Letterbox:
    """Resize keeping aspect ratio, then pad to size=(W, H), centered."""

    def __init__(self, size=TARGET_SIZE, fill=PAD_FILL, resample=BILINEAR):
        self.size, self.fill, self.resample = size, fill, resample

    def __call__(self, img):
        W, H = self.size
        w, h = img.size
        s = float(letterbox_scale(w, h, self.size))
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        img = img.resize((nw, nh), self.resample)
        canvas = Image.new('RGB', (W, H), self.fill)
        canvas.paste(img, ((W - nw) // 2, (H - nh) // 2))
        return canvas

def make_transform(train, size=TARGET_SIZE):
    """Letterbox -> (flip, color jitter if train) -> tensor -> normalize.
    No random cropping: a crop can cut a labeled object out of the image."""
    from torchvision import transforms as T
    tf = [Letterbox(size)]
    if train:
        tf += [T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2)]
    tf += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(tf)

class OpenImagesMultiLabel(Dataset):
    def __init__(self, image_dir, cache_path, transform=None, fast_decode=True):
        cache = torch.load(cache_path, weights_only=False)
        self.image_dir = image_dir
        self.transform = transform
        self.fast_decode = fast_decode
        self.ids = cache['image_ids']
        self.classes = cache['classes']
        self.target_size = cache.get('target_size', TARGET_SIZE)
        self.num_object_classes = cache.get('num_object_classes', len(self.classes))
        self.target = cache['target']
        self.mask = cache['mask']
        self.is_background = cache.get('is_background',
                                       torch.zeros(len(self.ids), dtype=torch.bool))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        path = os.path.join(self.image_dir, f'{self.ids[i]}.jpg')
        with Image.open(path) as img:
            if self.fast_decode:
                # JPEG DCT-domain downscale: decode at >= 2x target, much faster
                W, H = self.target_size
                img.draft('RGB', (2 * W, 2 * H))
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.target[i], self.mask[i]

    def pos_weight(self):
        """neg/pos ratio per class over KNOWN labels, for BCE pos_weight."""
        p = (self.target * self.mask).sum(0)
        n = ((1 - self.target) * self.mask).sum(0)
        return n / p.clamp(min=1)

def masked_bce(logits, target, mask, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction='none', pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp(min=1)

def masked_map(scores, target, mask, min_mask=1.0):
    """Per-class AP over labels with mask >= min_mask (default: verified only)."""
    from sklearn.metrics import average_precision_score
    scores, target, mask = (t.cpu().numpy() for t in (scores, target, mask))
    aps = []
    for c in range(scores.shape[1]):
        k = mask[:, c] >= min_mask
        y = target[k, c]
        aps.append(average_precision_score(y, scores[k, c])
                   if 0 < y.sum() < len(y) else float('nan'))
    return float(np.nanmean(aps)), aps

def map(scores, target):
    """Per-class AP over labels with mask >= min_mask (default: verified only)."""
    from sklearn.metrics import average_precision_score
    scores, target = (t.cpu().numpy() for t in (scores, target))
    aps = []
    for c in range(scores.shape[1]):
        y = target[:, c]
        aps.append(average_precision_score(y, scores[:, c])
                   if 0 < y.sum() < len(y) else float('nan'))
    return float(np.nanmean(aps)), aps

def background_fpr(probs, thresholds, is_background, num_object_classes):
    """On background images: share where ANY object class fires, and per class."""
    k = num_object_classes
    fired = probs[is_background, :k] > torch.as_tensor(thresholds)[:k]
    return fired.any(1).float().mean().item(), fired.float().mean(0)
