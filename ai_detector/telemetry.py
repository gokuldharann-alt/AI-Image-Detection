"""
Comprehensive "behind the scenes" telemetry + logging.

Two channels:
  1. Terminal  — colourised, human-readable per-request report via std logging.
  2. API/UI    — structured ``TelemetryCollector`` buffer streamed to the
                 frontend log console (returned as ``logs[]`` in /api/predict).

Each inference records: request_id, device, dtype, stage timings
(decode / preprocess / forward / postprocess / total), tensor shapes,
confidence scores, throughput (img/s) and layer-activation insights
(channel mean/std, sparsity, top-k channels of the last conv block).
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List

# ------------------------------------------------------------ base logger ---
_logger: logging.Logger | None = None

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[92m",
    "cyan": "\033[96m",
    "yellow": "\033[93m",
    "magenta": "\033[95m",
    "red": "\033[91m",
    "gray": "\033[90m",
}


def get_logger(name: str = "ai_detector") -> logging.Logger:
    """Singleton logger with a clean timestamped stdout handler (once)."""
    global _logger
    if _logger is not None:
        return _logger
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        _logger.addHandler(handler)
    _logger.propagate = False
    return _logger


# ------------------------------------------------- global recent-log ring ---
# Powers GET /api/logs/stream (Server-Sent Events tail).
RECENT_EVENTS: Deque[Dict[str, Any]] = deque(maxlen=200)


def push_global_event(event: Dict[str, Any]) -> None:
    event = {"t": time.strftime("%H:%M:%S"), **event}
    RECENT_EVENTS.append(event)


# ------------------------------------------------------- per-request buffer ---
@dataclass
class TelemetryCollector:
    """Collects timestamped log lines + numeric telemetry for one request."""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    t0: float = field(default_factory=time.perf_counter)
    lines: List[str] = field(default_factory=list)
    stages: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- logging helpers -------------------------------------------------
    def log(self, stage: str, message: str) -> None:
        dt = (time.perf_counter() - self.t0) * 1000
        line = f"[+{dt:7.1f}ms] [{stage}] {message}"
        self.lines.append(line)
        get_logger().info(f"req={self.request_id} {line}")

    def stage(self, name: str, seconds: float) -> None:
        self.stages[name] = round(seconds * 1000, 2)  # store ms

    def set(self, **kwargs: Any) -> None:
        self.extra.update(kwargs)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "stages_ms": dict(self.stages),
            "total_ms": self.total_ms,
            **self.extra,
        }


def print_request_report(tel: TelemetryCollector, result: Dict[str, Any]) -> None:
    """Pretty colourised terminal report printed after every prediction."""
    log = get_logger()
    c = ANSI
    label = result.get("label", "?")
    conf = result.get("confidence", 0.0)
    colour = c["green"] if label == "REAL" else c["magenta"]
    log.info(
        f"{c['bold']}=== req={tel.request_id} "
        f"-> {colour}{label} {conf:.2%}{c['reset']}{c['bold']} "
        f"in {tel.total_ms:.1f}ms ==={c['reset']}"
    )
    for k, v in tel.stages.items():
        log.info(f"    {c['dim']}[t] {k:<12}{c['reset']} {v:8.2f} ms")
    for k, v in tel.extra.items():
        log.info(f"    {c['cyan']}- {k:<12}{c['reset']} {v}")
    log.info(f"    {c['gray']}confidence REAL={result.get('prob_real', 0):.4f} "
             f"FAKE={result.get('prob_fake', 0):.4f}{c['reset']}")
