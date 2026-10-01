"""
Model factory — lightweight backbones optimised for binary REAL vs FAKE.

Backbones
---------
* ``efficientnet_b0`` (default) — ~5.3M params, ~0.39 GFLOPs @224px.
  Best speed/accuracy trade-off for <200 ms CPU inference.
* ``resnet50`` — ~25.6M params. Slightly more accurate on some CIFAKE
  splits, ~3-4x slower. Select with ``MODEL_BACKBONE=resnet50``.

Both start from ImageNet-1K weights and replace the classification head
with a single binary logit. ``train.py`` fine-tunes the whole network.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2 as T

import config


def build_model(backbone: str = config.MODEL_BACKBONE,
                pretrained: bool = True) -> nn.Module:
    """Build the binary classifier. Returns model in eval-ready (not eval) state."""
    backbone = backbone.lower()
    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features  # 1280
        net.classifier[1] = nn.Linear(in_features, 1)  # binary logit
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        in_features = net.fc.in_features  # 2048
        net.fc = nn.Linear(in_features, 1)
    elif backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.convnext_tiny(weights=weights)
        in_features = net.classifier[2].in_features  # 768
        net.classifier[2] = nn.Linear(in_features, 1)
    elif backbone == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_v2_s(weights=weights)
        in_features = net.classifier[1].in_features  # 1280
        net.classifier[1] = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Use efficientnet_b0 | resnet50 | convnext_tiny | efficientnet_v2_s.")
    return net


def last_conv_module(net: nn.Module, backbone: str = config.MODEL_BACKBONE):
    """Return the last convolutional feature map module (for activation hooks)."""
    backbone = backbone.lower()
    if backbone == "efficientnet_b0":
        return net.features[-1]  # last MBConv block
    if backbone == "efficientnet_v2_s":
        return net.features[-1]
    if backbone == "convnext_tiny":
        return net.features[-1][-1]  # last ConvNeXt block
    return net.layer4[-1]        # last ResNet bottleneck


def get_preprocess_transforms(image_size: int = config.IMAGE_SIZE) -> T.Compose:
    """
    Inference preprocessing — MUST mirror train.py validation transforms:
    Resize(int(IM*1.15)) -> CenterCrop(IM) -> ToImage -> ToDtype(float32, scale) -> Normalize(ImageNet).
    """
    return T.Compose([
        T.Resize(int(image_size * 1.15)),
        T.CenterCrop(image_size),
        T.ToImage(),  # PIL -> uint8 tensor (required before ToDtype/Normalize in tv v2)
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=list(config.IMAGENET_MEAN), std=list(config.IMAGENET_STD)),
    ])


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
