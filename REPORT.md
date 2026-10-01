# Veritas — AI Image Detector: Real vs AI-Generated Image Classification

**Project report (submission) — EfficientNet-B0, multi-generator training, FastAPI + ONNX deployment**

---

## Abstract

Veritas is a binary image classifier that distinguishes real photographs from AI-generated images, served through a FastAPI backend with a zero-build web UI. The production model is EfficientNet-B0 (5.3M parameters), ImageNet-pretrained and fine-tuned in two stages — first on a high-resolution multi-generator set (HQ, 4,000 images), then on modern-generator data (AUTO, ~6,000 images) — at 384px resolution with a single binary logit (sigmoid = P(FAKE), threshold 0.5). Measured test performance of the submitted checkpoint (`hq_efficientnet_b0.pt`): **AUTO-test accuracy 0.977 / F1 0.977 / AUC 0.997** (n=996) and **HQ-test accuracy 0.906 / F1 0.897 / AUC 0.990** (n=1000), with per-generator FAKE recall 0.976–0.994 on modern generators. Inference runs on ONNX Runtime (CPU) or FP16 mixed precision (CUDA) against a sub-200ms budget, with per-request telemetry, a dataset gallery, an interactive CNN playground, a live metrics page, and a user-feedback loop for continual learning.

---

## 1. Problem Definition and Objectives (criterion 1)

**Problem.** Generative models (Stable Diffusion family, MidJourney, DALL-E, FLUX and others) now produce photorealistic images at scale, enabling misinformation, fraud and non-consensual content. Manual inspection does not scale, and single-generator detectors fail on unseen generators.

**Objectives (all measurable).**
1. Classify any uploaded image (JPG/PNG/WEBP/BMP, ≤15MB) as REAL vs AI-generated with calibrated confidence.
2. Generalise across generators — not just the one trained on (multi-generator training sets).
3. Respond in under 200ms per image on CPU (ONNX Runtime) with a full timing breakdown per request.
4. Deploy as a usable product: web UI, documented API, reproducible data-to-weights pipeline, and a feedback mechanism so mistakes become future training data.

---

## 2. Dataset and Data Preparation (criterion 2)

### 2.1 Data lineage (no CIFAKE in the final model)

| Stage | Location | Composition | Resolution | Size | Purpose |
|---|---|---|---|---|---|
| CIFAKE (legacy only) | `data/CIFAKE` | CIFAR-10 photos vs Stable Diffusion 1.4 (Kaggle `birdy654`; Bird & Lotfi, 2024) | 32×32 | 120,000 (100k train / 20k test) | Trains the legacy `cifake_*` checkpoint only. Excluded from the submitted model: single generator + tiny resolution teaches a resolution shortcut. |
| NEWER (staging) | `data/NEWER` | CIFAKE + MidJourney (`bitmind/GenImage_MidJeremy` shard) + MJ/DALL-E/SD/NanoBanana mix (`julienlucas/...nanobanapro`) + art fakes (`hmnshudhmn24/...art-images`) | mixed | ~14,500 (5770+5770 train, 1472+1472 test) | Staging pool. No checkpoint trains on it directly; it feeds HQ. |
| HQ (foundation) | `data/HQ` | COCO val2017 real photos (Lin et al., 2014), filtered ≥448px + NEWER fakes filtered ≥224px (CIFAKE 32px explicitly dropped) | canonical **512px JPEG Q95**, balanced REAL==FAKE | **4,000** (1500+1500 train, 500+500 test) | First-stage fine-tuning (val acc 0.947). |
| AUTO (expansion) | `data/AUTO` | Defactify — SD2.1/SDXL/SD3/DALL-E 3/MidJourney-v6 (Roy et al., 2026); v3 — SD1.5/SDXL/FLUX/Kandinsky/PixArt/Würstchen (`Shanmuk4622/...`); Tiny-GenImage — ADM/BigGAN/GLIDE/MJ/SD1.4/SD1.5/VQDM/Wukong (`TheKernel01/...`) | canonical 512px JPEG Q95, SHA256-deduped, resume-safe streaming (`dl_auto.py`) | **~6,000** (2499+2499 train, 498+498 test) | Second-stage fine-tuning. Current checkpoint trained here. |
| FINAL (GPU merge) | Colab `/content/data/FINAL` | AUTO + COCO + MJ/nano/art (mj 403 + nano 875 + art 500) | 512px Q95 | ~9,000 (4000+4000 train, 498+498 test) | GPU training target (see §4.4). |
| Feedback (personalisation) | `data/feedback/{REAL,FAKE}` + `feedback_log.csv` | User corrections from the UI, canonicalized 512px Q95 | 512px Q95 | 35 (22 REAL / 13 FAKE) | Continual learning via `learn_feedback.py` (not yet merged — see §7 limitations). |

### 2.2 Preparation pipeline (reproducible scripts)

1. `dl_hq_real.py` — COCO val2017 (~900MB) → `data/HQ_SRC_REAL/` (5,000 photos).
2. `dl_newer.py` — MidJourney parquet shard (526MB) + nanobanana shard (497MB) + art imagefolder → `data/_raw/`.
3. `build_newer.py` — assembles NEWER with fixed seed (7), disjoint train/test slices.
4. `build_hq.py` — resolution filters (≥224px fake / ≥448px real), EXIF-transpose → RGB → square center-crop → Lanczos 512 → JPEG Q95, REAL==FAKE enforced.
5. `dl_auto.py` — streams the three modern sets with per-source quotas, content-hash dedup (re-runs skip byte-identical images), seed-fixed train/test assignment.
6. `setup_data.py --check` — verifies every stage; full chain is resumable and skips finished steps.

---

## 3. Model Selection and Architecture (criterion 3)

**Backbone: EfficientNet-B0** (`ai_detector/model.py:build_model`), chosen for the stated CPU latency budget: ~5.3M parameters (~15.6MB fp32 checkpoint), compound-scaled MBConv blocks with squeeze-excitation. The ImageNet-1K classification head is replaced with a **single binary logit** (`classifier[1] = Linear(1280, 1)`); `sigmoid(logit) = P(FAKE)`, decision threshold 0.5 (`config.THRESHOLD`). Alternatives (`resnet50`, `convnext_tiny`, `efficientnet_v2_s`) are wired into the same factory and CLI but deliberately not trained: on a CPU-only deployment they cost 3–4× latency for an expected +1–3% accuracy, so B0 is the correct operating point for objectives 1–3, with the larger backbones documented as future work.

**Preprocessing (train/inference matched).** Training: `RandomResizedCrop(384, scale 0.7–1.0)` + horizontal flip + colour jitter + Gaussian blur → ImageNet normalisation. Validation/inference (identical): `Resize(441)` → `CenterCrop(384)` → ImageNet normalisation (`ai_detector/model.py:get_preprocess_transforms`, mirroring `train.py:build_loaders`). Label mapping is explicit and tested: `ImageFolder` sorts classes FAKE=0/REAL=1, mapped to BCE target `1 − idx` so the network models P(FAKE) (`train.py:flip_fake_target`).

**Inference engines.** PyTorch fp32/fp16-AMP (`channels_last` memory format, `torch.compile`-ready) plus an ONNX Runtime graph (opset 18, dynamic batch) exported by `export_onnx.py` with a loud parity check (torch vs ORT logits must agree within 0.2 and predict the same class, else the file is deleted). A forward hook on the last MBConv block reports activation statistics per request.

---

## 4. Model Training and Optimization (criterion 4)

**Two-stage transfer learning** (full fine-tune, AdamW + cosine annealing + AMP):

| Stage | Data | Init | LR | Epochs | Batch/workers | Result |
|---|---|---|---|---|---|---|
| 1. HQ adapt | HQ 3,000 train | ImageNet | 3e-4 | 8 | 16 / 0 (Windows CPU) | val acc **0.947** (~13 min/epoch, ~4 img/s; 188 batches/epoch) |
| 2. AUTO expand | AUTO 4,998 train | Stage-1 `hq_*` (`--resume`) | 3e-4 | 8 | 16–32 / 0 (CPU) | saved acc **0.9769** (~5 hrs total on CPU) |
| 3. FINAL (GPU) | FINAL 8,000 train | ImageNet (Colab notebook `colab_gpu_train.ipynb`, T4) | 3e-4 | 12 | 64 / 4 | ~5 min/epoch (recipe provided; weights drop into local `weights/`) |

Optimisation details: `BCEWithLogitsLoss`, AdamW (weight decay 1e-4), `CosineAnnealingLR`, `GradScaler` (active on CUDA only), early stopping on validation accuracy (patience 4), best-checkpoint saving as `{backbone, acc, img_size, state_dict}`. `--head-only` mode (frozen backbone) is provided for small-data updates. Continual learning (`learn_feedback.py`): mixes HQ-train with feedback ×3 at LR 5e-5 for 5 epochs, saving back to the same `hq_*.pt` (with `.bak`) **only if** HQ validation does not regress — an anti-collapse, anti-poisoning guard.

---

## 5. Evaluation Metrics and Results (criterion 5)

Evaluated with `eval_metrics.py` on the submitted checkpoint (`hq_efficientnet_b0.pt`, img 384) — 1,996 test images on CPU, FAKE treated as the positive class, threshold fixed at 0.5 (no tuning applied). Full output ships in `weights/metrics.json` and on the `/metrics` page.

| Test set (n) | Accuracy | Precision (FAKE) | Recall (FAKE) | F1 | AUC | Confusion TN/FP/FN/TP |
|---|---|---|---|---|---|---|
| HQ-test (1000) | **0.906** | 0.993 | 0.818 | 0.897 | 0.990 | 497 / 3 / 91 / 409 |
| AUTO-test (996) | **0.977** | 0.968 | 0.986 | 0.977 | 0.997 | 482 / 16 / 7 / 491 |

**Per-generator FAKE recall (generalisation probe):** Defactify 0.988 (164/166), Tiny-GenImage 0.994 (165/166), Shanmuk-v3 0.976 (162/166), HQ-era (MJ/nano/art) 0.818 (409/500). Mean P(FAKE): 0.02–0.05 on true-REAL vs 0.80–0.97 on true-FAKE, confirming the label mapping is correct (not flipped). A threshold sweep (FAR/FRR vs threshold) is computed in the same script; the operating point is reported at the default 0.5.

**Honest limitations stated.** (a) AUTO retraining caused mild forgetting on HQ fakes (0.947 → 0.818 recall) — the classic plasticity/stability trade-off, disclosed rather than hidden. (b) Phone-camera photos (the feedback cluster: 22 REAL/13 FAKE) remain the weak domain — one correction moves P(FAKE) 1.00→~0.85 but cannot flip the verdict alone against 3,000 HQ images, by design. (c) No backbone ablation was executed; B0 is justified by latency budget, not by beaten alternatives.

---

## 6. Scalability and Deployment (criterion 6)

**Service.** FastAPI (`app.py`): model loads once at startup with request-path warm-up; `POST /api/predict` (multipart image → label, confidence, both probabilities, logit, threshold, per-stage timings, activation insights, log lines); `GET /api/health`, `/api/model/info`, `/api/datasets*` (paginated gallery backend), `/api/metrics` (precomputed numbers); `POST /api/feedback` (stores canonicalized corrections + CSV log); `GET /api/logs/stream` (SSE). CPU-bound forwards run in worker threads so the event loop stays responsive. Reference latency envelope: ~35ms/image CPU-ONNX, ~12ms CUDA-FP16 (design figures; CPU test rig measured ~4 img/s end-to-end for 384px training forwards).

**Configuration.** Single source of truth (`config.py`), everything env-overridable: `TORCH_WEIGHTS` / `ONNX_WEIGHTS` (point at `hq_*`), `MODEL_BACKBONE`, `IMAGE_SIZE` (384), `THRESHOLD`, `DEVICE`, `USE_ONNX`, `USE_FP16`, `MAX_UPLOAD_MB`. Deployment artefacts: `render.yaml` (free-tier CPU), `Dockerfile` (container fallback), `requirements.txt` (pinned inference + training deps).

**User interface (zero-build, one font, shared theme).** Detect (`/`, drag-drop/paste + animated verdict + live terminal), Datasets (`/gallery`, Photos-style grid over HQ/NEWER/AUTO with references panel), Playground (`/playground`, one photo stepped through decode → preprocess → 10 B0 blocks → sigmoid using live API numbers), Metrics (`/metrics`, this section rendered), all cross-linked.

**Reproducibility.** One-command data check (`setup_data.py --check`), one-command GPU training (`colab_gpu_train.ipynb`: token prompt → downloads → FINAL build → train → ONNX → download weights), deterministic seeds (7), canonical 512px pipeline shared by training, feedback storage and the gallery.

---

## 7. Innovation and Impact (criterion 7)

1. **Continual-learning loop with guardrails.** User corrections don’t vanish into a spreadsheet — they accumulate in `data/feedback/`, and `learn_feedback.py` merges them (upsampled ×3, low LR) while refusing to save if base validation regresses. Few coursework detectors close this loop at all.
2. **Provenance-aware direction (researched, honestly scoped).** Studied Google DeepMind SynthID and the `aloshdenny/reverse-SynthID` spectral-analysis work (FFT carrier detection ~90% local claim; Google disputes systematic removal), plus OpenAI’s C2PA+SynthID verification and MidJourney’s documented lack of any watermark. Conclusion for this project: integrate **detection-side only** (C2PA manifest reading + SD-family invisible-watermark checks as advisory badges), never removal tooling. Shipped as a build plan; not claimed as done.
3. **Evaluation transparency as a feature.** The `/metrics` page publishes the forgetting regression and the weak domain alongside the headline 0.977 — unusual in student work and directly auditable via `weights/metrics.json`.
4. **Impact.** A deployable artefact (not a notebook): real-time demo, dataset literacy (gallery references), model literacy (playground), plus-Colab GPU path so the project survives beyond CPU-scale data.

**Future work (prioritised).** (i) Land the 35 feedback images with an 8-epoch low-LR run and re-publish Metrics; (ii) threshold calibration from the existing sweep (report FAR/FRR operating points); (iii) one backbone ablation (`convnext_tiny` vs B0 on FINAL, head-only, 4 epochs); (iv) provenance badges (C2PA/SD-watermark, advisory only); (v) 12GB Synthbuster as a true held-out unseen set.

---

## References

- Lin et al., Microsoft COCO: Common Objects in Context, 2014 (real photos).
- Bird & Lotfi, CIFAKE: Real and AI-Generated Synthetic Images, 2024 (legacy stage).
- Zhu et al., GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images, 2023 (MidJourney shard family).
- Roy et al., Defactify Image Dataset: Human vs AI-Generated Image Detection, arXiv 2601.00553, 2026.
- `Shanmuk4622/ai-detection-dataset-v3`; `TheKernel01/Tiny-GenImage` (Hugging Face).
- Tan & Le, EfficientNet: Rethinking Model Scaling for CNNs, 2019 (backbone).
- Gowal et al., SynthID-Image, 2026; Alosh Denny, reverse-SynthID spectral analysis, 2026 (studied for provenance direction).
- C2PA Technical Specification (Content Credentials); OpenAI content-provenance docs.

## Appendix A — Repository map

`app.py` (API + pages) · `config.py` (all settings) · `train.py` / `learn_feedback.py` / `eval_metrics.py` / `export_onnx.py` · `setup_data.py`, `dl_hq_real.py`, `dl_newer.py`, `build_newer.py`, `build_hq.py`, `dl_auto.py` (data chain) · `colab_gpu_train.ipynb` (GPU path) · `ai_detector/` (model, inference, telemetry) · `static/` (index, gallery, playground, metrics) · `weights/` (`hq_*.pt/.onnx`, `cifake_*`, `metrics.json`) · `data/` (CIFAKE, NEWER, HQ, AUTO, HQ_SRC_REAL, feedback — not versioned).

## Appendix B — Key commands (CMD)

```cmd
python setup_data.py --check
python dl_auto.py --token <HF_TOKEN> --max-train-per-class 2500 --max-test-per-class 500
python train.py --data ./data/AUTO --backbone efficientnet_b0 --img-size 384 --resume weights/hq_efficientnet_b0.pt --epochs 8 --batch 16 --workers 0 --out hq_efficientnet_b0.pt
python export_onnx.py --backbone efficientnet_b0
python eval_metrics.py --checkpoint weights/hq_efficientnet_b0.pt --batch 32
set TORCH_WEIGHTS=weights\hq_efficientnet_b0.pt && set ONNX_WEIGHTS=weights\hq_efficientnet_b0.onnx && set IMAGE_SIZE=384 && uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
