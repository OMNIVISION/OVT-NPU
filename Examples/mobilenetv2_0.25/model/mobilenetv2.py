'''
This script is a modified version of https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py

The original docs:
    """
    Creates a MobileNetV2 Model as defined in:
    Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    arXiv preprint arXiv:1801.04381.
    import from https://github.com/tonylins/pytorch-mobilenet-v2
    """
'''

import os
from typing import Optional, Dict, Union, Tuple
import math

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

__all__ = ["build_mobilenetv2_model"]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(
                c * width_mult, 4 if width_mult == 0.1 else 8
            )
            for i in range(n):
                layers.append(
                    block(input_channel, output_channel, s if i == 0 else 1, t)
                )
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = (
            _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8)
            if width_mult > 1.0
            else 1280
        )
        self.conv = conv_1x1_bn(input_channel, output_channel)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# def mobilenetv2(**kwargs):
#     """
#     Constructs a MobileNet V2 model
#     """
#     return MobileNetV2(**kwargs)


def build_mobilenetv2_model(
    width_mult: float = 0.25,
    dropout: float = 0.5,
    image_size: Union[int, Tuple[int, int]] = [120, 160],
    device: torch.device = torch.device("cpu"),
    with_sigmoid: bool = True,
    num_classes: int = 1,
) -> nn.Module:
    """
    Build a MobileNetV2 model with optional custom width multiplier.

    Args:
        width_mult (float): -1 for default TorchVision model, or e.g. 0.25, 1.0, etc.
        dropout (float): Dropout probability.
        image_size (int): Used for a dummy forward to infer final channels.
        device (torch.device): The device to put the model on.

    Returns:
        nn.Module: A complete MobileNetV2-based model.
    """
    pretrained_base = MobileNetV2(
        num_classes=num_classes, width_mult=width_mult
    ).features
    pretrained_base = pretrained_base.to(device)
    pretrained_base.eval()

    # Obtain final channel dimension
    with torch.no_grad():
        if isinstance(image_size, int):
            dummy_in = torch.zeros(1, 3, image_size, image_size, device=device)
        elif isinstance(image_size, tuple) or isinstance(image_size, list):
            dummy_in = torch.zeros(1, 3, image_size[0], image_size[1], device=device)
        else:
            raise NotImplementedError
        feat_out = pretrained_base(dummy_in)
        out_channels = feat_out.shape[1]

    drop = nn.Identity() if dropout < 0 else nn.Dropout(dropout)
    act = nn.Sigmoid() if with_sigmoid else nn.Identity()

    model = nn.Sequential(
        pretrained_base,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        drop,
        nn.Linear(out_channels, num_classes),
        act,
    ).to(device)

    return model


if __name__ == "__main__":
    # Configuration: set model checkpoint, input image, and compute device
    ptfile = "mobilenetv2_alpha0.25.pth"
    imgfile = "testing_image.bmp"
    device = torch.device("cpu")

    preprocess = transforms.Compose(
        [
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Input preparation: load image and create a normalized batch tensor
    print(f"Reading image {imgfile}")
    image = Image.open(imgfile)
    print(f"Processing image {imgfile}")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Model setup: construct MobileNetV2 and load trained weights
    print(f"Loading checkpoint {ptfile}")
    model = build_mobilenetv2_model(
        width_mult=0.25,
        dropout=-1.0,
        image_size=[120, 160],
        device=device,
        with_sigmoid=True,
        num_classes=1,
    ).to(device)
    state_dict_loaded = torch.load(ptfile, map_location=device)
    state_dict = state_dict_loaded["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Inference result: confidence = ", end="")
    with torch.no_grad():
        output_tensor = model(input_tensor)
        print(output_tensor.numpy()[0][0])

    # Export: trace the model with the same preprocessed input and save as TorchScript
    ts = torch.jit.trace(model, input_tensor)
    file_ts = "mobilenetv2_0.25.zip"
    print(f"Saving TorchScipt file {file_ts}")
    ts.save(file_ts)
