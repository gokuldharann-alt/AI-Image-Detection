"""
FastAPI backend — ultra-fast prediction API + Spotify-style static UI.

Endpoints
---------
GET  /                        → serves the Web UI (static/index.html)
GET  /api/health               → liveness + device/engine info
GET  /api/model/info           → backbone, params, engine, dtype
POST /api/predict              → multipart image → {label, confidence, telemetry, logs}
GET  /api/logs/stream          → Server-Sent Events tail of recent predictions

Run:
    uvicorn app:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from ai_detector import AIDetector, get_logger

log = get_logger()
detector: AIDetector | None = None


# ------------------------------------------------------------- lifespan ---
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model ONCE at startup (not per request) — key to <200 ms."""
    global detector
    t0 = time.perf_counter()
    log.info("[start] AI Image Detector - loading model...")
    detector = AIDetector()  # reads config.py env vars
    # Warm the exact request path (worker thread via to_thread, like /api/predict
    # uses): CUDA lazy-init is per-thread, so warming only the main thread
    # would leave a ~2-3 s stall on the first real request.
    try:
        import io as _io
        from PIL import Image as _Image

        _buf = _io.BytesIO()
        _Image.new("RGB", (256, 256), "gray").save(_buf, format="PNG")
        await asyncio.to_thread(detector.predict_bytes, _buf.getvalue(), "warmup")
        log.info("[ok] Request-path warm-up complete.")
    except Exception as exc:
        log.warning(f"Request-path warm-up skipped: {exc}")
    log.info(f"[ok] Model ready in {(time.perf_counter() - t0) * 1000:.0f}ms - serving UI + API.")
    yield
    log.info("[stop] Shutting down.")


app = FastAPI(title=config.API_TITLE, version=config.API_VERSION, lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static assets (CSS/JS) — index.html served at / for zero-config UX.
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui() -> FileResponse:
    return FileResponse(f"{config.STATIC_DIR}/index.html", media_type="text/html")


@app.get("/gallery", include_in_schema=False)
async def serve_gallery() -> FileResponse:
    return FileResponse(f"{config.STATIC_DIR}/gallery.html", media_type="text/html")


@app.get("/playground", include_in_schema=False)
async def serve_playground() -> FileResponse:
    return FileResponse(f"{config.STATIC_DIR}/playground.html", media_type="text/html")


@app.get("/metrics", include_in_schema=False)
async def serve_metrics() -> FileResponse:
    return FileResponse(f"{config.STATIC_DIR}/metrics.html", media_type="text/html")


@app.get("/api/metrics")
async def api_metrics() -> JSONResponse:
    """Precomputed test numbers (python eval_metrics.py). Regenerate offline, served static."""
    import json as _json
    path = os.path.join(config.WEIGHTS_DIR, "metrics.json")
    if not os.path.isfile(path):
        raise HTTPException(404, "No metrics yet. Run: python eval_metrics.py")
    with open(path) as f:
        return JSONResponse(_json.load(f))


# ---------------------------------------------------------- dataset gallery ---
# Google-Photos-style browsing of training data (HQ + NEWER + AUTO).
# Paginated + validated to avoid path traversal and 100k-file scans.
_GALLERY_DATASETS = ("HQ", "NEWER", "AUTO")
_GALLERY_SPLITS = ("train", "test")
_GALLERY_CLASSES = ("REAL", "FAKE")
_GALLERY_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _gallery_dir(dataset: str, split: str, cls: str) -> str:
    return os.path.join(config.BASE_DIR, "data", dataset, split, cls)


def _count_images(d: str) -> int:
    try:
        return sum(1 for f in os.listdir(d) if f.lower().endswith(_GALLERY_EXTS))
    except OSError:
        return 0


@app.get("/api/datasets")
async def datasets() -> JSONResponse:
    """Counts per dataset/split/class for the gallery UI."""
    out: dict = {}
    for ds in _GALLERY_DATASETS:
        out[ds] = {}
        for sp in _GALLERY_SPLITS:
            out[ds][sp] = {c: _count_images(_gallery_dir(ds, sp, c)) for c in _GALLERY_CLASSES}
    return JSONResponse({"datasets": out})


@app.get("/api/datasets/images")
async def dataset_images(dataset: str = "HQ", split: str = "train",
                         cls: str = "ALL", offset: int = 0, limit: int = 60) -> JSONResponse:
    """Paginated file list. cls=ALL|REAL|FAKE. Sorted for stable paging."""
    dataset = (dataset or "HQ").upper()
    if dataset not in _GALLERY_DATASETS:
        raise HTTPException(400, "dataset must be HQ, NEWER or AUTO.")
    if split not in _GALLERY_SPLITS:
        raise HTTPException(400, "split must be train or test.")
    cls = (cls or "ALL").upper()
    if cls not in (*_GALLERY_CLASSES, "ALL"):
        raise HTTPException(400, "cls must be ALL, REAL or FAKE.")
    limit = max(1, min(int(limit or 60), 120))
    offset = max(0, int(offset or 0))

    classes = _GALLERY_CLASSES if cls == "ALL" else (cls,)
    names: list[dict] = []
    for c in classes:
        d = _gallery_dir(dataset, split, c)
        try:
            files = sorted(f for f in os.listdir(d) if f.lower().endswith(_GALLERY_EXTS))
        except OSError:
            files = []
        names.extend({"cls": c, "name": f} for f in files)
    names.sort(key=lambda r: (r["cls"], r["name"]))
    total = len(names)
    page = names[offset:offset + limit]
    for r in page:
        r["url"] = f"/api/datasets/file?dataset={dataset}&split={split}&cls={r['cls']}&name={r['name']}"
    return JSONResponse({"dataset": dataset, "split": split, "cls": cls,
                         "total": total, "offset": offset, "limit": limit, "images": page})


@app.get("/api/datasets/file")
async def dataset_file(dataset: str = "HQ", split: str = "train",
                       cls: str = "REAL", name: str = "") -> FileResponse:
    """Serve a single gallery image (validated, no traversal)."""
    dataset = (dataset or "").upper()
    if dataset not in _GALLERY_DATASETS or split not in _GALLERY_SPLITS or cls not in _GALLERY_CLASSES:
        raise HTTPException(400, "Invalid dataset/split/cls.")
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "Invalid name.")
    if not name.lower().endswith(_GALLERY_EXTS):
        raise HTTPException(400, "Invalid extension.")
    path = os.path.join(_gallery_dir(dataset, split, cls), name)
    if not os.path.isfile(path):
        raise HTTPException(404, "Image not found.")
    return FileResponse(path)


# ----------------------------------------------------------------- health ---
@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "model_loaded": detector is not None,
        **(detector.info() if detector else {}),
    })


@app.get("/api/model/info")
async def model_info() -> JSONResponse:
    if detector is None:
        raise HTTPException(503, "Model still loading, retry in a second.")
    return JSONResponse(detector.info())


# --------------------------------------------------------------- predict ---
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accepts a single image (jpeg/png/webp/bmp, ≤ MAX_UPLOAD_MB).
    Returns verdict + confidence + full telemetry + log lines for UI streaming.
    """
    if detector is None:
        raise HTTPException(503, "Model still loading, retry in a second.")
    if file.content_type not in config.ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported type '{file.content_type}'. "
                                 f"Allowed: {sorted(config.ALLOWED_MIME)}")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file received.")
    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_MB}MB limit.")
    try:
        # CPU-bound torch forward inside a thread so the event loop stays snappy.
        result, _logs = await asyncio.to_thread(detector.predict_bytes, raw, file.filename or "upload")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # never leak tracebacks; log them
        log.exception(f"Prediction failed: {exc}")
        raise HTTPException(500, "Inference failed. Check server logs.")
    return JSONResponse(result)


# ---------------------------------------------------------- feedback loop ---
@app.post("/api/feedback")
async def feedback(
    file: UploadFile = File(...),
    true_label: str = Form("REAL"),
    pred_label: str = Form(""),
    confidence: float = Form(0.0),
) -> JSONResponse:
    """
    User correction: saves image + true label for continual learning.
    Form fields: file (image), true_label REAL|FAKE, pred_label, confidence.
    Stored in data/feedback/{REAL,FAKE}/ + feedback_log.csv (review before retrain).
    """
    from datetime import datetime, timezone
    import csv
    import hashlib

    true_label = (true_label or "").upper()
    if true_label not in ("REAL", "FAKE"):
        raise HTTPException(400, "true_label must be REAL or FAKE.")
    if file.content_type not in config.ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported type '{file.content_type}'.")
    raw = await file.read()
    if not raw or len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, "Empty file or exceeds size limit.")
    try:
        from PIL import Image as _PIL
        import io as _io
        _PIL.open(_io.BytesIO(raw)).verify()
    except Exception:
        raise HTTPException(400, "Cannot decode image.")

    fb_dir = os.path.join(config.BASE_DIR, "data", "feedback", true_label)
    os.makedirs(fb_dir, exist_ok=True)
    digest = hashlib.sha256(raw).hexdigest()[:12]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    fname = f"{ts}_{digest}.jpg"
    # Canonicalize to JPEG to match HQ training pipeline.
    try:
        from PIL import Image as _PIL2, ImageOps as _Ops
        import io as _io2
        im = _Ops.exif_transpose(_PIL2.open(_io2.BytesIO(raw)).convert("RGB"))
        w, h = im.size
        side = min(w, h)
        im = im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
        im = im.resize((512, 512), _PIL2.LANCZOS)
        im.save(os.path.join(fb_dir, fname), "JPEG", quality=95)
    except Exception as exc:
        raise HTTPException(400, f"Could not store feedback image: {exc}")

    log_path = os.path.join(config.BASE_DIR, "data", "feedback_log.csv")
    is_new = not os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["ts", "file", "true_label", "pred_label", "confidence"])
        w.writerow([ts, f"{true_label}/{fname}", true_label, pred_label, confidence])
    n_real = len(os.listdir(os.path.join(config.BASE_DIR, "data", "feedback", "REAL"))) if os.path.isdir(os.path.join(config.BASE_DIR, "data", "feedback", "REAL")) else 0
    n_fake = len(os.listdir(os.path.join(config.BASE_DIR, "data", "feedback", "FAKE"))) if os.path.isdir(os.path.join(config.BASE_DIR, "data", "feedback", "FAKE")) else 0
    log.info(f"[feedback] {true_label} (was {pred_label} {confidence:.2f}) -> {fname} | totals R={n_real} F={n_fake}")
    return JSONResponse({"ok": True, "saved": f"{true_label}/{fname}",
                         "totals": {"REAL": n_real, "FAKE": n_fake},
                         "hint": "Retrain with: python learn_feedback.py once you have 50+ per class."})


# ---------------------------------------------------------- SSE log tail ---
@app.get("/api/logs/stream", response_class=StreamingResponse, response_model=None)
async def logs_stream() -> StreamingResponse:
    """Server-Sent Events: streams the global recent-predictions ring buffer."""
    from ai_detector.telemetry import RECENT_EVENTS
    import json

    async def gen():
        last_n = 0
        yield "retry: 2000\n\n"
        while True:
            events = list(RECENT_EVENTS)
            for ev in events[last_n:]:
                yield f"data: {json.dumps(ev)}\n\n"
            last_n = len(events)
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


# --------------------------------------------------------------- fallback ---
@app.exception_handler(404)
async def not_found(_req, _exc):
    return JSONResponse({"error": "Not found. See /docs for API."}, status_code=404)


if __name__ == "__main__":
    import uvicorn

    _port = int(os.getenv("PORT", "8000"))
    _host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run("app:app", host=_host, port=_port, reload=True)
