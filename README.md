# Veritas — AI Image Detector

Binary image classifier distinguishing real photographs from AI-generated images,
served through a FastAPI backend with a zero-build web UI.

The default model is EfficientNet-B0 fine-tuned on CIFAKE (and optionally on a
higher-resolution multi-generator set), with an ONNX Runtime fast path for CPU
inference. Every prediction returns the verdict plus per-stage timings, device
and engine details, and a step-by-step processing log.

## Features

- Real vs. AI-generated classification with calibrated confidence scores
- Fast inference: ONNX Runtime on CPU, FP16 mixed precision on CUDA
- Single model load at startup; thread-isolated inference per request
- Per-request telemetry: decode / preprocess / forward / postprocess timings
- Activation insights from the final convolutional block
- Web UI with drag-and-drop, paste-from-clipboard, and result visualization
- User-feedback endpoint for collecting corrections and continual learning
- Server-Sent Events stream of recent predictions

## Project structure

```
.
├── app.py                  # FastAPI app: routes, startup model load, SSE log stream
├── config.py               # Central configuration (env-overridable)
├── train.py                # Fine-tuning on CIFAKE or HQ data
├── export_onnx.py          # Checkpoint export to ONNX (opset 18)
├── learn_feedback.py       # Continual learning from collected user feedback
├── setup_data.py           # Fresh-clone data setup: CIFAKE -> _raw -> NEWER -> HQ
├── dl_hq_real.py           # Download COCO val2017 real photos (~900 MB)
├── dl_newer.py             # Download newer-generator HF shards (~1.2 GB)
├── build_newer.py          # Assemble data/NEWER (CIFAKE + newer fakes)
├── build_hq.py             # Assemble data/HQ (high-resolution, balanced)
├── eval_metrics.py         # Full evaluation (HQ+AUTO test, metrics.json for /metrics)
├── requirements.txt
├── render.yaml             # Render deployment (Python runtime, free tier)
├── Dockerfile              # Container deployment fallback
├── ai_detector/
│   ├── __init__.py
│   ├── model.py            # Backbone factory and preprocessing
│   ├── inference.py        # AIDetector pipeline and activation probe
│   └── telemetry.py        # Request telemetry and log formatting
├── static/
│   ├── index.html          # Web UI
│   ├── styles.css          # Stylesheet
│   └── app.js              # Upload handling, result rendering, log streaming
└── weights/                # Checkpoints (*.pt) and ONNX graphs (*.onnx)
```

## Requirements

- Python 3.10 or newer
- Approximately 2 GB of free disk space
- A CUDA-capable GPU is optional (CPU inference is fully supported)
- Training requires a GPU for practical runtimes

## Getting started

```powershell
git clone <repository-url>
cd "AI Image Detector"

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000`, upload an image, and select Analyze.
Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

> Note: without trained weights the server runs in demo mode on ImageNet base
> weights (~20 MB, downloaded on first run). Predictions become accurate only
> after training as described below.

## Data setup

`data/` is not versioned. On a fresh clone, prepare datasets with:

```powershell
python setup_data.py --check     # report what exists and what is missing
python setup_data.py --dry-run   # preview the plan without downloading
python setup_data.py             # full run (resumable; skips completed stages)
```

The pipeline resolves, in order: `data/CIFAKE` (manual Kaggle download or
`kagglehub`), `data/_raw` shards, `data/HQ_SRC_REAL`, `data/NEWER`, and
`data/HQ`. See `python setup_data.py --help` for `--only`, `--skip`, and
`--auto-install` options. Expect roughly 2.1 GB of downloads plus built copies.

## Training

Fine-tune on CIFAKE (approximately 93–96% test accuracy, 12 epochs):

```powershell
python train.py --data ./data/CIFAKE --epochs 12 --batch 64 --backbone efficientnet_b0
python export_onnx.py --backbone efficientnet_b0
```

For the higher-resolution multi-generator set:

```powershell
python train.py --data ./data/HQ --backbone efficientnet_b0 --img-size 384
python export_onnx.py --backbone efficientnet_b0
```

Restart the server after training so the new checkpoint is loaded.

### Continual learning from user feedback

Corrections submitted through the UI are stored under
`data/feedback/{REAL,FAKE}/` with an entry in `data/feedback_log.csv`. Once at
least 20 feedback images are collected:

```powershell
python learn_feedback.py --backbone efficientnet_b0 --img-size 384 --epochs 5 --batch 32
python export_onnx.py --backbone efficientnet_b0
```

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Liveness probe with device, engine, and model metadata |
| `GET` | `/api/model/info` | Backbone, parameter count, dtype, and ONNX availability |
| `POST` | `/api/predict` | Multipart `file` field; returns label, confidence, telemetry, and log lines |
| `POST` | `/api/feedback` | Submit a correction (`file`, `true_label`, `pred_label`, `confidence`) |
| `GET` | `/api/logs/stream` | Server-Sent Events stream of recent predictions |

Example:

```powershell
curl -F "file=@sample.jpg" http://127.0.0.1:8000/api/predict
```

```json
{
  "label": "FAKE", "confidence": 0.9721,
  "prob_real": 0.0279, "prob_fake": 0.9721, "logit": 3.55,
  "telemetry": { "total_ms": 41.2, "stages_ms": {"decode": 3.1, "preprocess": 2.4, "forward": 34.9, "postprocess": 0.1},
                 "device": "CPU", "engine": "onnx", "throughput_img_s": 24.3 },
  "activations": {"channels": 1280, "channel_mean": 0.31, "sparsity": 0.42, "top_channels": [921, 44, 117]},
  "logs": ["[+   3.2ms] [decode] PIL image mode=RGB size=(512, 512) ..."]
}
```

## Configuration

All settings live in `config.py` and can be overridden with environment variables:

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `auto` | `auto` (CUDA if available, else CPU), `cuda`, or `cpu` |
| `USE_ONNX` | `1` | Prefer the ONNX Runtime graph when available |
| `USE_FP16` | `1` | FP16 mixed precision (CUDA only) |
| `USE_COMPILE` | `0` | `torch.compile` (PyTorch 2+, CUDA) |
| `IMAGE_SIZE` | `384` | Center-crop size; must match the training resolution |
| `THRESHOLD` | `0.5` | Decision threshold on P(FAKE) |
| `MODEL_BACKBONE` | `efficientnet_b0` | `efficientnet_b0`, `resnet50`, `convnext_tiny`, or `efficientnet_v2_s` |
| `MAX_UPLOAD_MB` | `15` | Maximum accepted upload size |
| `PORT` / `HOST` | `8000` / `127.0.0.1` | Server bind address |

## Performance

EfficientNet-B0 reference figures: approximately 35 ms per image on CPU via
ONNX Runtime and 12 ms on CUDA via FP16. Prefer ONNX on CPU and FP16 on GPU;
`USE_COMPILE=1` can reduce latency by a further ~15% on Ampere and newer GPUs.
Raising `IMAGE_SIZE` to 384 (the HQ default) increases accuracy on
high-resolution images at the cost of higher latency.

## Deployment

The service binds to `$PORT` and runs CPU-only when `DEVICE=cpu`, so it works
on platforms without GPU access. `render.yaml` provides a Render free-tier
configuration (Python runtime with a CPU-only PyTorch install); `Dockerfile`
is available as a container fallback. See `weights/README.txt` for checkpoint
details.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `backend offline` status in the UI | Server not running or wrong port; check the `uvicorn` process |
| `415 Unsupported type` | Convert the image to JPG, PNG, WEBP, or BMP |
| `413 File exceeds limit` | Images must be within `MAX_UPLOAD_MB` (default 15 MB) |
| Slow first prediction | One-time weight download and kernel warm-up; later requests are fast |
| Low accuracy | Server is in demo mode; train on CIFAKE or HQ data first |

## Acknowledgments

Built with PyTorch, FastAPI, and ONNX Runtime. Training data: CIFAKE (Bird and
Lotfi, 2024) — real CIFAR photographs versus Stable Diffusion generations —
supplemented with COCO photographs and openly licensed synthetic-image sets.
See per-script headers for dataset licenses and suggested citations.
