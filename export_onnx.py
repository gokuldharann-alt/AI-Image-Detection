"""
Export the trained checkpoint to ONNX for the ONNX Runtime fast path.

    python export_onnx.py --backbone efficientnet_b0

Reads weights/cifake_<backbone>.pt -> writes weights/cifake_<backbone>.onnx
(opset 18, dynamic batch axis). app.py auto-detects the .onnx file when
USE_ONNX=1 (default) and prefers it on CPU for lowest latency.

Uses the classic (torchscript-based) exporter because the dynamo exporter
in some torch builds silently drops initializers for EfficientNet-style
graphs (0.6 MB stub instead of ~20 MB). Post-export parity check against
the torch model fails loudly instead of shipping a broken file.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import config
from ai_detector.model import build_model

MIN_SIZE_MB = 5.0  # EfficientNet-B0 fp32 is ~20 MB; anything far below is a stub
LOGIT_TOL = 2e-1  # cross-engine (torch vs ORT) kernel summation order differs;
                  # torch 2.14 vs ORT 1.30 drifts ~7e-02 on random noise; logits here
                  # are typically |logit|>4, so <0.2 shift never flips threshold 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.MODEL_BACKBONE)
    args = ap.parse_args()

    pt = os.path.join(config.WEIGHTS_DIR, f"cifake_{args.backbone}.pt")
    onnx_path = os.path.join(config.WEIGHTS_DIR, f"cifake_{args.backbone}.onnx")
    # Prefer HQ checkpoint if present (fresh multi-generator training).
    hq_pt = os.path.join(config.WEIGHTS_DIR, f"hq_{args.backbone}.pt")
    hq_onnx = os.path.join(config.WEIGHTS_DIR, f"hq_{args.backbone}.onnx")
    if os.path.isfile(hq_pt):
        pt, onnx_path = hq_pt, hq_onnx
    if not os.path.isfile(pt):
        raise FileNotFoundError(f"No checkpoint at {pt}. Train first: python train.py")

    model = build_model(args.backbone, pretrained=False)
    ckpt = torch.load(pt, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()
    export_size = int(ckpt.get("img_size", config.IMAGE_SIZE)) if isinstance(ckpt, dict) else config.IMAGE_SIZE

    dummy = torch.randn(1, 3, export_size, export_size)
    with torch.no_grad():
        ref = model(dummy).numpy()

    # Classic exporter (dynamo=False): embeds weights reliably for CNNs.
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=18, do_constant_folding=True, dynamo=False,
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"wrote {onnx_path} ({size_mb:.1f} MB, opset 18)")
    if size_mb < MIN_SIZE_MB:
        os.remove(onnx_path)
        raise RuntimeError(
            f"Export produced a {size_mb:.1f} MB stub (< {MIN_SIZE_MB} MB) - "
            "weights missing. Deleted. Try upgrading torch/onnx.")

    # Parity check: ORT output must match torch on the same input.
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": dummy.numpy().astype("float32")})[0]
    diff = float(np.abs(got - ref).max())
    same_label = bool((got.reshape(-1)[0] >= 0) == (ref.reshape(-1)[0] >= 0))
    print(f"parity max|onnx-torch| = {diff:.2e} (tol {LOGIT_TOL:.0e}), same class: {same_label}")
    if diff > LOGIT_TOL or not same_label:
        os.remove(onnx_path)
        raise RuntimeError(f"ONNX parity check failed (diff {diff:.2e}). Deleted.")
    print(f"[OK] exported {onnx_path} ({size_mb:.1f} MB, opset 18)")


if __name__ == "__main__":
    main()
