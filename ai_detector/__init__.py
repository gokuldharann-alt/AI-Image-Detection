"""AI Image Detector package — model, inference pipeline and telemetry."""
from .model import build_model, get_preprocess_transforms  # noqa: F401
from .inference import AIDetector  # noqa: F401
from .telemetry import get_logger, TelemetryCollector, print_request_report  # noqa: F401

__all__ = [
    "build_model",
    "get_preprocess_transforms",
    "AIDetector",
    "get_logger",
    "TelemetryCollector",
    "print_request_report",
]
__version__ = "1.0.0"
