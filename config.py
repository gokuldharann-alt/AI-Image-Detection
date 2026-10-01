"""
Central configuration for the AI Image Detector.
Single source of truth — imported by app.py, train.py and ai_detector/*.
Override any value with an environment variable, e.g.:
    MODEL_BACKBONE=resnet50 uvicorn app:app --port 8000
"""
from __future__ import annotations

import os
import torch

# ---------------------------------------------------------------- paths ---
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR: str = os.path.join(BASE_DIR, "weights")
STATIC_DIR: str = os.path.join(BASE_DIR, "static")

TORCH_WEIGHTS: str = os.getenv("TORCH_WEIGHTS", os.path.join(WEIGHTS_DIR, "cifake_efficientnet_b0.pt"))
ONNX_WEIGHTS: str = os.getenv("ONNX_WEIGHTS", os.path.join(WEIGHTS_DIR, "cifake_efficientnet_b0.onnx"))
# HQ (multi-generator, high-res) checkpoints — train.py writes hq_<backbone>.pt
HQ_TORCH_PATTERN: str = os.path.join(WEIGHTS_DIR, "hq_{backbone}.pt")
HQ_ONNX_PATTERN: str = os.path.join(WEIGHTS_DIR, "hq_{backbone}.onnx")

# ---------------------------------------------------------------- model ---
MODEL_BACKBONE: str = os.getenv("MODEL_BACKBONE", "efficientnet_b0")  # efficientnet_b0 | resnet50 | convnext_tiny | efficientnet_v2_s
NUM_CLASSES: int = 1            # binary logit (sigmoid -> P(FAKE))
IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "384"))
CLASS_NAMES = ["REAL", "FAKE"]  # index 0 = REAL, 1 = FAKE (via sigmoid threshold 0.5)
THRESHOLD: float = float(os.getenv("THRESHOLD", "0.5"))

# ImageNet normalisation (CIFAKE models are fine-tuned from ImageNet).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ------------------------------------------------------------- inference ---
DEVICE: str = os.getenv("DEVICE", "auto")  # auto | cuda | cpu
USE_FP16: bool = os.getenv("USE_FP16", "1") == "1"      # AMP on CUDA only
USE_ONNX: bool = os.getenv("USE_ONNX", "1") == "1"      # prefer ONNX Runtime if .onnx exists
USE_COMPILE: bool = os.getenv("USE_COMPILE", "0") == "1"  # torch.compile (PyTorch 2+, CUDA)
ONNX_PROVIDERS_ORDER = ("CUDAExecutionProvider", "CPUExecutionProvider")

BATCH_SIZE: int = 1
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "15"))
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# ----------------------------------------------------------------- api ----
API_TITLE = "AI Image Detector — CIFAKE"
API_VERSION = "1.0.0"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


def resolve_device(preference: str = DEVICE) -> torch.device:
    """Resolve 'auto' -> cuda if available else cpu."""
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return f"CUDA:{device.index or 0} ({torch.cuda.get_device_name(device)})"
        except Exception:
            return "CUDA"
    return "CPU"
