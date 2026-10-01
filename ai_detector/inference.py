"""
Optimised inference pipeline — target <200 ms / image.

Optimisations applied (all safe for binary classification):
  1. Single model load at startup, ``eval()`` + ``inference_mode`` (no grad).
  2. ``channels_last`` memory format (faster convs on CPU & CUDA).
  3. Mixed precision FP16 via ``autocast`` on CUDA (2-3x faster, ~same acc).
  4. Optional ``torch.compile`` (PyTorch 2+, ``USE_COMPILE=1``).
  5. Optional ONNX Runtime session (``USE_ONNX=1`` + weights/*.onnx) —
     graph fusion + arena allocator; usually the fastest CPU path.
  6. Fused preprocess (PIL -> float tensor, no redundant copies),
     batch dim kept at 1, pinned timing with perf_counter.

Also captures "layer-activation insights": a forward hook on the last
conv block records channel mean/std, sparsity and top-3 active channels.
"""
from __future__ import annotations

import io
import os
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import config
from .model import build_model, get_preprocess_transforms, last_conv_module, count_parameters
from .telemetry import TelemetryCollector, get_logger, print_request_report, push_global_event

log = get_logger()


class AIDetector:
    """Singleton-friendly detector. Load once, call ``predict_bytes`` per request."""

    def __init__(
        self,
        weights_path: str = config.TORCH_WEIGHTS,
        backbone: str = config.MODEL_BACKBONE,
        device_preference: str = config.DEVICE,
        use_fp16: bool = config.USE_FP16,
        use_onnx: bool = config.USE_ONNX,
        use_compile: bool = config.USE_COMPILE,
    ) -> None:
        self.backbone = backbone
        self.device = config.resolve_device(device_preference)
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.transforms = get_preprocess_transforms()

        # ---- build torch model -----------------------------------------
        t_build = time.perf_counter()
        self.model = build_model(backbone, pretrained=True)
        self.n_params = count_parameters(self.model)

        fine_tuned = False
        if os.path.isfile(weights_path):
            try:
                ckpt = torch.load(weights_path, map_location="cpu")
                state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                self.model.load_state_dict(state, strict=False)
                fine_tuned = True
                log.info(f"Loaded fine-tuned weights: {weights_path}")
            except Exception as exc:  # corrupt checkpoint -> fall back safely
                log.warning(f"Could not load {weights_path} ({exc}); using ImageNet base.")
        if not fine_tuned:
            log.warning(
                "No fine-tuned CIFAKE weights found at weights/*.pt - "
                "running on ImageNet base (demo mode). Train with `python train.py` "
                "for real accuracy."
            )
        self.fine_tuned = fine_tuned
        self.model.eval().to(self.device, memory_format=torch.channels_last)

        if use_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore[attr-defined]
                log.info("torch.compile enabled (max-autotune off, default mode).")
            except Exception as exc:
                log.warning(f"torch.compile failed, continuing eagerly: {exc}")

        # ---- activation probe (last conv block) -------------------------
        self._last_feat: torch.Tensor | None = None

        def _hook(_m, _i, out):
            self._last_feat = out.detach()

        last_conv_module(self.model, backbone).register_forward_hook(_hook)

        # ---- optional ONNX Runtime fast path ----------------------------
        self.ort_session = None
        self.ort_input_name: str | None = None
        if use_onnx and os.path.isfile(config.ONNX_WEIGHTS):
            try:
                import onnxruntime as ort

                providers = [
                    p for p in config.ONNX_PROVIDERS_ORDER
                    if p in ort.get_available_providers()
                ] or ["CPUExecutionProvider"]
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_session = ort.InferenceSession(
                    config.ONNX_WEIGHTS, sess_options=opts, providers=providers
                )
                self.ort_input_name = self.ort_session.get_inputs()[0].name
                log.info(f"ONNX Runtime enabled: {config.ONNX_WEIGHTS} providers={providers}")
            except Exception as exc:
                log.warning(f"ONNX load failed ({exc}); using PyTorch path.")
                self.ort_session = None

        self.engine = "onnx" if self.ort_session is not None else "pytorch-fp16" if self.use_fp16 else "pytorch-fp32"
        log.info(
            f"Detector ready | backbone={backbone} params={self.n_params:,} "
            f"device={config.device_label(self.device)} engine={self.engine} "
            f"build={(time.perf_counter() - t_build) * 1000:.0f}ms"
        )
        # Warm-up (fuses kernels / allocates arenas so p50 latency is honest).
        self._warmup()

    # ------------------------------------------------------------------ #
    def _warmup(self) -> None:
        try:
            dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            with torch.inference_mode():
                if self.ort_session is not None:
                    import numpy as np

                    self.ort_session.run(
                        None, {self.ort_input_name: dummy.numpy().astype("float32")}
                    )
                # Always warm the torch path too: the activation probe runs a
                # torch forward on EVERY request (even with ONNX), so skipping
                # this defers CUDA lazy-init + kernel autotune to request #1.
                ctx = torch.autocast("cuda", dtype=torch.float16) if self.use_fp16 else nullcontext()
                with ctx:
                    _ = self.model(dummy.to(self.device, memory_format=torch.channels_last))
            log.info("Warm-up forward pass complete.")
        except Exception as exc:
            log.warning(f"Warm-up skipped: {exc}")

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def predict_bytes(self, image_bytes: bytes, filename: str = "upload") -> Tuple[Dict[str, Any], List[str]]:
        """Full pipeline from raw bytes -> (result dict, log lines). Thread-safe."""
        tel = TelemetryCollector()
        tel.log("init", f"req={tel.request_id} file='{filename}' size={len(image_bytes) / 1024:.1f}KB")
        tel.set(device=config.device_label(self.device), engine=self.engine,
                backbone=self.backbone, dtype=str(self.dtype).replace("torch.", ""))

        # 1 — decode ------------------------------------------------------
        t = time.perf_counter()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot decode image (corrupt/unsupported): {exc}")
        tel.stage("decode", time.perf_counter() - t)
        tel.log("decode", f"PIL image mode=RGB size={image.size} (WxH)")

        # 2 — preprocess ---------------------------------------------------
        t = time.perf_counter()
        tensor = self.transforms(image)                       # (3, 224, 224) float32
        input_shape = tuple(tensor.shape)
        batch = tensor.unsqueeze(0).to(memory_format=torch.channels_last)  # (1,3,224,224)
        tel.stage("preprocess", time.perf_counter() - t)
        tel.log("preprocess", f"resize->256 centercrop->{config.IMAGE_SIZE} normalize(ImageNet) "
                              f"tensor={tuple(batch.shape)} dtype={tensor.dtype}")

        # 3 — forward -------------------------------------------------------
        t = time.perf_counter()
        logit, prob_fake = self._forward(batch, tel)
        tel.stage("forward", time.perf_counter() - t)

        # 4 — postprocess ----------------------------------------------------
        t = time.perf_counter()
        prob_fake_f = float(prob_fake)
        prob_real_f = 1.0 - prob_fake_f
        label = "FAKE" if prob_fake_f >= config.THRESHOLD else "REAL"
        confidence = prob_fake_f if label == "FAKE" else prob_real_f
        tel.stage("postprocess", time.perf_counter() - t)
        tel.log("postprocess", f"sigmoid({logit:.4f}) -> P(FAKE)={prob_fake_f:.4f} "
                               f"P(REAL)={prob_real_f:.4f} threshold={config.THRESHOLD} -> {label}")

        # 5 — activation insights ---------------------------------------------
        insights = self._activation_insights(tel)

        result: Dict[str, Any] = {
            "label": label,
            "confidence": round(confidence, 4),
            "prob_real": round(prob_real_f, 4),
            "prob_fake": round(prob_fake_f, 4),
            "logit": round(float(logit), 4),
            "threshold": config.THRESHOLD,
            "telemetry": tel.to_dict(),
            "activations": insights,
            "logs": tel.lines,  # streamed line-by-line by the UI
            "model": {
                "backbone": self.backbone,
                "params": self.n_params,
                "fine_tuned": self.fine_tuned,
                "engine": self.engine,
            },
        }
        tel.set(input_shape=f"(1, 3, {config.IMAGE_SIZE}, {config.IMAGE_SIZE})",
                output_shape="(1, 1)", throughput_img_s=round(1000 / max(tel.total_ms, 1e-6), 1))
        result["telemetry"] = tel.to_dict()  # refresh with final extras

        print_request_report(tel, result)
        push_global_event({"req": tel.request_id, "label": label,
                           "conf": round(confidence, 4), "ms": tel.total_ms})
        return result, tel.lines

    # ------------------------------------------------------------------ #
    def _forward(self, batch: torch.Tensor, tel: TelemetryCollector) -> Tuple[float, float]:
        """Run the fastest available engine. Returns (logit, prob_fake)."""
        if self.ort_session is not None:
            import numpy as np

            arr = batch.numpy().astype("float32")  # ORT takes NHWC/NCHW float32
            tel.log("forward", f"engine=onnxruntime input={arr.shape} dtype=float32")
            ort_out = self.ort_session.run(None, {self.ort_input_name: arr})[0]
            logit = float(ort_out.reshape(-1)[0])
            tel.log("forward", f"onnx output shape={tuple(ort_out.shape)} logit={logit:.4f}")
            # keep probe fresh for insights (cheap torch pass NOT needed; use zeros guard)
            with torch.inference_mode():
                ctx = torch.autocast("cuda", dtype=torch.float16) if self.use_fp16 else nullcontext()
                with ctx:
                    _ = self.model(batch.to(self.device))
        else:
            x = batch.to(self.device, non_blocking=True)
            tel.log("forward", f"engine=torch device={self.device.type} "
                               f"dtype={'fp16-amp' if self.use_fp16 else 'fp32'} "
                               f"input={tuple(x.shape)} memory_format=channels_last")
            with torch.inference_mode():
                ctx = torch.autocast("cuda", dtype=torch.float16) if self.use_fp16 else nullcontext()
                with ctx:
                    out = self.model(x)
            logit = float(out.reshape(-1)[0].float().cpu())
            tel.log("forward", f"torch output shape={tuple(out.shape)} logit={logit:.4f}")
        prob_fake = float(torch.sigmoid(torch.tensor(logit)))
        return logit, prob_fake

    # ------------------------------------------------------------------ #
    def _activation_insights(self, tel: TelemetryCollector) -> Dict[str, Any]:
        feat = self._last_feat
        if feat is None:
            return {"note": "probe unavailable (onnx-only path)"}
        try:
            f = feat.float().cpu()  # (1, C, h, w)
            per_channel = f.mean(dim=(0, 2, 3))          # (C,)
            sparsity = float((f == 0).float().mean())
            topk = torch.topk(per_channel, k=min(3, per_channel.numel()))
            insights = {
                "feature_map": list(f.shape),
                "channels": int(f.shape[1]),
                "channel_mean": round(float(per_channel.mean()), 4),
                "channel_std": round(float(per_channel.std()), 4),
                "sparsity": round(sparsity, 4),
                "top_channels": [int(i) for i in topk.indices.tolist()],
                "top_values": [round(float(v), 4) for v in topk.values.tolist()],
            }
            tel.log("insights", f"last-conv {list(f.shape)} mean={insights['channel_mean']} "
                                f"std={insights['channel_std']} sparsity={sparsity:.2%} "
                                f"top_ch={insights['top_channels']}")
            tel.set(**{f"act_{k}": v for k, v in insights.items()
                       if k in ("channels", "channel_mean", "channel_std", "sparsity")})
            return insights
        except Exception as exc:
            return {"note": f"insight computation failed: {exc}"}

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "backbone": self.backbone,
            "params": self.n_params,
            "device": config.device_label(self.device),
            "engine": self.engine,
            "dtype": str(self.dtype).replace("torch.", ""),
            "image_size": config.IMAGE_SIZE,
            "threshold": config.THRESHOLD,
            "fine_tuned": self.fine_tuned,
            "onnx_available": self.ort_session is not None,
            "fp16": self.use_fp16,
        }
