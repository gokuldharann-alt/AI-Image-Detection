"""Auto-download + canonicalize extra REAL/FAKE sets into data/AUTO (no manual work).

Sources (all pre-labelled, streaming so no giant download):
    1. Shanmuk4622/ai-image-detection-dataset
    2. Rajarshi-Roy-research/Defactify_Image_Dataset   (Label_A: 0=real, 1=fake)
    3. TheKernel01/Tiny-GenImage                       (label: 0=real, 1=fake)

Output (train.py-ready ImageFolder, balanced, HQ-style 512px Q95):
    data/AUTO/{train,test}/{REAL,FAKE}/auto_<src>_<i>.jpg

Usage (CMD):
    .venv\\Scripts\\activate.bat
    pip install datasets Pillow huggingface_hub
    python dl_auto.py --token hf_xxx --max-train-per-class 2500 --max-test-per-class 500
    python dl_auto.py --token hf_xxx --out ./data/AUTO --overwrite

Notes:
- --token is also read from HF_TOKEN env if omitted.
- Quota is split EVENLY across the 3 sources for generator diversity.
- Train/test assigned by seed-fixed random (no scene leak within a source beyond source split).
- Same canonicalize as build_hq.py: EXIF -> RGB -> square center-crop -> Lanczos 512 -> JPEG Q95.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

from PIL import Image, ImageOps


def fmt_time(s: float) -> str:
    s = max(0, int(s))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

DEFAULT_DATASETS = [
    "Shanmuk4622/ai-image-detection-dataset",
    "Rajarshi-Roy-research/Defactify_Image_Dataset",
    "TheKernel01/Tiny-GenImage",
]

CANON_SIZE = 512
JPEG_Q = 95
SEED = 7
SCAN_CAP_PER_DS = 60000  # max streamed rows scanned per dataset (quota usually hit first)

# candidate column names across the 3 schemas
IMAGE_COLS = ("image", "Image", "img", "picture", "file")
LABEL_COLS = ("label", "Label_A", "label_a", "veracity", "real_fake", "class", "target")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download 3 HF sets into data/AUTO (streaming).")
    ap.add_argument("--out", default="./data/AUTO")
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS,
                    help="Override HF dataset IDs (space-separated).")
    ap.add_argument("--max-train-per-class", type=int, default=2500)
    ap.add_argument("--max-test-per-class", type=int, default=500)
    ap.add_argument("--size", type=int, default=CANON_SIZE)
    ap.add_argument("--quality", type=int, default=JPEG_Q)
    ap.add_argument("--token", default=os.getenv("HF_TOKEN", ""),
                    help="HF token (or set HF_TOKEN env). Passed as token= to load_dataset.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Clear existing data/AUTO train/test before writing.")
    ap.add_argument("--no-canonicalize", action="store_true",
                    help="Save raw bytes instead of 512px Q95 canonical JPEG.")
    return ap.parse_args()


def canonicalize_pil(im: Image.Image, dst: str, size: int, quality: int) -> bool:
    data = canonicalize_bytes(im, size, quality, raw=False)
    if data is None:
        return False
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def canonicalize_bytes(im: Image.Image, size: int, quality: int, raw: bool) -> bytes | None:
    """Canonicalize to JPEG bytes in memory (for hash-dedup before writing)."""
    try:
        import io
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        if not raw:
            w, h = rgb.size
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            rgb = rgb.crop((left, top, left + side, top + side))
            rgb = rgb.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=quality)
        return buf.getvalue()
    except Exception:
        return None


def index_existing(out: str) -> tuple[dict, set]:
    """Scan data/AUTO for auto_* files. Returns (counts, sha256 set) for resume+dedup."""
    import glob
    import hashlib
    counts: dict = {}
    hashes: set = set()
    files: list[str] = []
    for sp in ("train", "test"):
        for cl in ("REAL", "FAKE"):
            files.extend(glob.glob(os.path.join(out, sp, cl, "auto_*")))
    t0 = time.perf_counter()
    for p in files:
        try:
            base = os.path.basename(p)  # auto_<tag>_<i>.jpg
            parts = base.split("_")
            tag = parts[1] if len(parts) >= 3 else "unk"
            # infer split/class from path
            sp = "train" if f"{os.sep}train{os.sep}" in p or "/train/" in p else "test"
            cl = "REAL" if f"{os.sep}REAL{os.sep}" in p or "/REAL/" in p else "FAKE"
            try:
                idx = int(parts[-1].split(".")[0])
            except ValueError:
                idx = -1
            key = (tag, sp, cl)
            n, mx = counts.get(key, (0, -1))
            counts[key] = (n + 1, max(mx, idx))
            with open(p, "rb") as f:
                hashes.add(hashlib.sha256(f.read()).hexdigest())
        except Exception:
            continue
    return counts, hashes


def to_pil(obj) -> Image.Image | None:
    """Accept PIL, {'bytes':..}, {'path':..}, raw bytes -> PIL."""
    try:
        if isinstance(obj, Image.Image):
            return obj
        if isinstance(obj, dict):
            raw = obj.get("bytes") or obj.get("data")
            if raw:
                import io
                return Image.open(io.BytesIO(raw))
            p = obj.get("path")
            if p and os.path.isfile(p):
                return Image.open(p)
            return None
        if isinstance(obj, (bytes, bytearray)):
            import io
            return Image.open(io.BytesIO(bytes(obj)))
        return None
    except Exception:
        return None


def to_label(obj) -> int | None:
    """Normalize to 0=REAL, 1=FAKE. Returns None if unknown."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (int, float)):
        v = int(obj)
        return v if v in (0, 1) else None
    s = str(obj).strip().lower()
    if s in ("0", "real", "human", "natural", "genuine", "false"):
        # note: some schemas use 0=real
        if s in ("real", "human", "natural", "genuine", "0"):
            return 0
    if s in ("1", "fake", "ai", "generated", "synthetic", "artificial", "true"):
        if s in ("fake", "ai", "generated", "synthetic", "artificial", "1"):
            return 1
    return None


def pick(cols: list[str], cands: tuple[str, ...]) -> str | None:
    low = {c.lower(): c for c in cols}
    for cand in cands:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def load_stream(ds_id: str, token: str):
    from datasets import load_dataset
    # new `token=` API, fallback to `use_auth_token=` on old versions
    try:
        return load_dataset(ds_id, split="train", streaming=True, token=token or None)
    except TypeError:
        return load_dataset(ds_id, split="train", streaming=True, use_auth_token=token or None)


def main() -> None:
    args = parse_args()
    token = (args.token or "").strip()
    if not token:
        print("WARNING: no --token / HF_TOKEN. Gated sets may 401. Continuing anonymous...",
              flush=True)

    n_src = max(1, len(args.datasets))
    q_tr = args.max_train_per_class // n_src
    q_te = args.max_test_per_class // n_src
    print(f"sources={args.datasets}\nquota per source: train {q_tr}/class, test {q_te}/class"
          f" -> target totals ~{q_tr * n_src}/class train, ~{q_te * n_src}/class test",
          flush=True)

    # prepare dirs
    for sp in ("train", "test"):
        for cl in ("REAL", "FAKE"):
            d = os.path.join(args.out, sp, cl)
            os.makedirs(d, exist_ok=True)
    if args.overwrite:
        import glob
        for p in glob.glob(os.path.join(args.out, "train", "*", "auto_*")) + \
                 glob.glob(os.path.join(args.out, "test", "*", "auto_*")):
            try:
                os.remove(p)
            except OSError:
                pass
        print("[overwrite] cleared old auto_* files.", flush=True)

    rng = random.Random(SEED)
    test_ratio = q_te / max(1, (q_tr + q_te))
    t_all = time.perf_counter()
    print(f"[t+0:00] start: {len(args.datasets)} source(s), "
          f"target ~{(q_tr + q_te) * 2 * n_src} images", flush=True)

    # resume index (no re-download): counts per (tag,split,class) + content hashes
    import hashlib
    exist_counts, exist_hashes = ({}, set()) if args.overwrite else index_existing(args.out)
    n_exist = sum(n for n, _ in exist_counts.values())
    if n_exist and not args.overwrite:
        print(f"[resume] found {n_exist} existing auto_* images, "
              f"{len(exist_hashes)} hashes indexed "
              f"in {fmt_time(time.perf_counter() - t_all)} — skipping duplicates", flush=True)
    totals = {"train": {"REAL": 0, "FAKE": 0}, "test": {"REAL": 0, "FAKE": 0}}
    for (tag, sp, cl), (n, _mx) in exist_counts.items():
        if sp in totals and cl in totals[sp]:
            totals[sp][cl] += n

    for di, ds_id in enumerate(args.datasets, 1):
        tag = ds_id.split("/")[-1].lower().replace("-", "")[:6]
        try:
            stream = load_stream(ds_id, token)
            cols = list(stream.features.keys())
        except Exception as exc:
            print(f"[SKIP] {ds_id}: cannot load ({exc}). Check ID / token / gated access.",
                  flush=True)
            continue
        img_col = pick(cols, IMAGE_COLS)
        lab_col = pick(cols, LABEL_COLS)
        if not img_col or not lab_col:
            print(f"[SKIP] {ds_id}: columns {cols} lack image/label pair.", flush=True)
            continue
        print(f"[{ds_id}] cols={cols} using image='{img_col}' label='{lab_col}'", flush=True)

        got = {"train": {"REAL": 0, "FAKE": 0}, "test": {"REAL": 0, "FAKE": 0}}
        # resume offsets: continue indices past existing files for this tag
        for sp0 in ("train", "test"):
            for cl0 in ("REAL", "FAKE"):
                n0, _mx0 = exist_counts.get((tag, sp0, cl0), (0, -1))
                got[sp0][cl0] = n0
        if (got["train"]["REAL"] >= q_tr and got["train"]["FAKE"] >= q_tr and
                got["test"]["REAL"] >= q_te and got["test"]["FAKE"] >= q_te):
            print(f"[{ds_id}] already complete "
                  f"train R/F={got['train']['REAL']}/{got['train']['FAKE']} "
                  f"test R/F={got['test']['REAL']}/{got['test']['FAKE']} — skipped", flush=True)
            continue
        scanned = kept = skipped_dup = 0
        t_ds = time.perf_counter()
        last_log = t_ds
        try:
            for row in stream:
                scanned += 1
                if scanned > SCAN_CAP_PER_DS:
                    break
                y = to_label(row.get(lab_col))
                if y is None:
                    continue
                cls = "FAKE" if y == 1 else "REAL"
                # seed-fixed train/test assignment
                sp = "test" if rng.random() < test_ratio else "train"
                quota = q_te if sp == "test" else q_tr
                if got[sp][cls] >= quota:
                    # try the other split before dropping
                    alt = "train" if sp == "test" else "test"
                    alt_q = q_tr if alt == "train" else q_te
                    if got[alt][cls] >= alt_q:
                        continue
                    sp = alt
                im = to_pil(row.get(img_col))
                if im is None:
                    continue
                data = canonicalize_bytes(im, args.size, args.quality,
                                          raw=args.no_canonicalize)
                if data is None:
                    continue
                h = hashlib.sha256(data).hexdigest()
                if h in exist_hashes:
                    skipped_dup += 1
                    continue
                i = got[sp][cls]
                dst = os.path.join(args.out, sp, cls, f"auto_{tag}_{i:05d}.jpg")
                try:
                    # collision-safe: bump index if filename taken by non-hash match
                    while os.path.isfile(dst):
                        i += 1
                        dst = os.path.join(args.out, sp, cls, f"auto_{tag}_{i:05d}.jpg")
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with open(dst, "wb") as f:
                        f.write(data)
                    ok = True
                except Exception:
                    ok = False
                if ok:
                    got[sp][cls] = i + 1
                    exist_hashes.add(h)
                    kept += 1
                    now = time.perf_counter()
                    if kept % 100 == 0 or (now - last_log) >= 15:
                        el = now - t_ds
                        rate = kept / max(el, 1e-6)
                        print(f"[{ds_id} {di}/{len(args.datasets)}] [t+{fmt_time(now - t_all)}] "
                              f"kept={kept} scanned={scanned} ({rate:.1f} img/s) "
                              f"train R/F={got['train']['REAL']}/{got['train']['FAKE']} "
                              f"test R/F={got['test']['REAL']}/{got['test']['FAKE']}", flush=True)
                        last_log = now
                if (got["train"]["REAL"] >= q_tr and got["train"]["FAKE"] >= q_tr and
                        got["test"]["REAL"] >= q_te and got["test"]["FAKE"] >= q_te):
                    break
        except Exception as exc:
            print(f"[{ds_id}] stream stopped early ({exc}). Keeping {kept}.", flush=True)
        el_ds = time.perf_counter() - t_ds
        print(f"[{ds_id}] done in {fmt_time(el_ds)} ({kept / max(el_ds, 1e-6):.1f} img/s): "
              f"scanned={scanned} kept={kept} dup_skipped={skipped_dup} "
              f"train R/F={got['train']['REAL']}/{got['train']['FAKE']} "
              f"test R/F={got['test']['REAL']}/{got['test']['FAKE']} "
              f"[t+{fmt_time(time.perf_counter() - t_all)}]", flush=True)
        for sp in ("train", "test"):
            for cl in ("REAL", "FAKE"):
                n0, _m0 = exist_counts.get((tag, sp, cl), (0, -1))
                totals[sp][cl] += got[sp][cl] - n0

    el_all = time.perf_counter() - t_all
    n_all = sum(totals[sp][cl] for sp in ("train", "test") for cl in ("REAL", "FAKE"))
    print(f"\nDONE in {fmt_time(el_all)} ({n_all / max(el_all, 1e-6):.1f} img/s overall) "
          f"-> {args.out} totals: "
          f"train REAL={totals['train']['REAL']} FAKE={totals['train']['FAKE']} | "
          f"test REAL={totals['test']['REAL']} FAKE={totals['test']['FAKE']}", flush=True)
    if min(totals["train"].values()) == 0:
        raise SystemExit("No images saved. Check IDs/token. "
                         "Tip: verify IDs on huggingface.co/datasets/<id>.")
    print("Next: python train.py --data ./data/AUTO --backbone efficientnet_b0 "
          "--img-size 384 --resume weights/hq_efficientnet_b0.pt --epochs 8 --batch 32 --workers 0")


if __name__ == "__main__":
    sys.exit(main())
