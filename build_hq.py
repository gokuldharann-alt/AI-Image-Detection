"""Build data/HQ — high-res, multi-generator, balanced (fixes CIFAKE shortcuts).

Why this exists:
- CIFAKE REAL/FAKE are 32x32 (CIFAR). NEWER fixed FAKE (256px MJ/nano/art)
  but build_newer.py still uses CIFAKE 32px for REAL -> resolution shortcut.
- HQ keeps only >=224px images, canonicalizes to 512px JPEG Q95, and
  balances REAL==FAKE with HIGH-RES real photos.

Usage:
    # 1. Put your good-quality real photos in data/HQ_SRC_REAL/ (jpg/png, >=448px)
    #    OR download COCO val2017: python dl_hq_real.py
    # 2. python build_hq.py --newer ./data/NEWER --real-src ./data/HQ_SRC_REAL --out ./data/HQ
    # 3. python train.py --data ./data/HQ --backbone efficientnet_b0 --img-size 384 --resume weights/cifake_efficientnet_b0.pt
"""
from __future__ import annotations

import argparse
import os
import random
import shutil

from PIL import Image, ImageOps

SEED = 7
MIN_FAKE_PX = 224   # drop CIFAKE 32px fakes leaking into NEWER
MIN_REAL_PX = 448   # good-quality real photos only
CANON_SIZE = 512
JPEG_Q = 95


def is_big_enough(path: str, min_px: int) -> tuple[bool, tuple[int, int]]:
    try:
        with Image.open(path) as im:
            w, h = im.size
            return (min(w, h) >= min_px, (w, h))
    except Exception:
        return (False, (0, 0))


def canonicalize(src: str, dst: str) -> bool:
    """EXIF-transpose -> RGB -> square center-crop -> Lanczos 512 -> JPEG Q95."""
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            im = im.crop((left, top, left + side, top + side))
            im = im.resize((CANON_SIZE, CANON_SIZE), Image.LANCZOS)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            im.save(dst, "JPEG", quality=JPEG_Q)
        return True
    except Exception:
        return False


def collect_fake(newer_dir: str, split: str) -> list[str]:
    d = os.path.join(newer_dir, split, "FAKE")
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            continue
        p = os.path.join(d, fn)
        ok, _ = is_big_enough(p, MIN_FAKE_PX)
        if ok:
            out.append(p)
    return out


def collect_real(real_src: str) -> list[str]:
    if not os.path.isdir(real_src):
        return []
    out = []
    for root, _, files in os.walk(real_src):
        for fn in sorted(files):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                continue
            p = os.path.join(root, fn)
            ok, _ = is_big_enough(p, MIN_REAL_PX)
            if ok:
                out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--newer", default="./data/NEWER")
    ap.add_argument("--real-src", default="./data/HQ_SRC_REAL",
                    help="Folder of YOUR good-quality real photos (or COCO val from dl_hq_real.py).")
    ap.add_argument("--out", default="./data/HQ")
    ap.add_argument("--max-train-fake", type=int, default=4000)
    ap.add_argument("--max-test-fake", type=int, default=1000)
    args = ap.parse_args()

    rng = random.Random(SEED)
    fake_train_all = collect_fake(args.newer, "train")
    fake_test_all = collect_fake(args.newer, "test")
    real_all = collect_real(args.real_src)

    print(f"FAKE candidates (>= {MIN_FAKE_PX}px): train={len(fake_train_all)} test={len(fake_test_all)}")
    print(f"REAL candidates (>= {MIN_REAL_PX}px) in {args.real_src}: {len(real_all)}")
    if not fake_train_all:
        raise SystemExit(f"No FAKE found in {args.newer}. Run build_newer.py first.")
    if not real_all:
        raise SystemExit(
            f"No high-res REAL in {args.real_src}.\n"
            "Fix: drop good-quality photos (>=448px) into data/HQ_SRC_REAL/ "
            "OR run: python dl_hq_real.py"
        )

    rng.shuffle(fake_train_all)
    rng.shuffle(fake_test_all)
    rng.shuffle(real_all)

    n_tr_fake = min(len(fake_train_all), args.max_train_fake)
    n_te_fake = min(len(fake_test_all), args.max_test_fake)
    # Balance REAL == FAKE
    if len(real_all) < n_tr_fake + n_te_fake:
        raise SystemExit(
            f"Need {n_tr_fake + n_te_fake} HQ real photos, have {len(real_all)}. "
            "Add more photos to data/HQ_SRC_REAL/ or lower --max-train-fake."
        )

    plan = [
        ("train", "FAKE", fake_train_all[:n_tr_fake], None),
        ("test", "FAKE", fake_test_all[:n_te_fake], None),
        ("train", "REAL", real_all[:n_tr_fake], "real"),
        ("test", "REAL", real_all[n_tr_fake:n_tr_fake + n_te_fake], "realt"),
    ]
    for split, cls, srcs, prefix in plan:
        dest = os.path.join(args.out, split, cls)
        os.makedirs(dest, exist_ok=True)
        # clear old canonical files
        for fn in os.listdir(dest):
            if fn.startswith(("hq_", "real_", "realt_", "fake_")):
                try:
                    os.remove(os.path.join(dest, fn))
                except OSError:
                    pass
        pre = prefix or ("fake" if cls == "FAKE" else "real")
        kept = 0
        for i, s in enumerate(srcs):
            if canonicalize(s, os.path.join(dest, f"hq_{pre}_{i:05d}.jpg")):
                kept += 1
        print(f"{split}/{cls}: {kept}/{len(srcs)} canonicalized -> {dest}")

    # Keep a couple CIFAKE 32px fakes OUT by design (they teach resolution shortcut).
    print(f"\nDone: data/HQ train {n_tr_fake}/{n_tr_fake} + test {n_te_fake}/{n_te_fake}, {CANON_SIZE}px Q{JPEG_Q}.")
    print("Next: python train.py --data ./data/HQ --backbone efficientnet_b0 --img-size 384 "
          "--resume weights/cifake_efficientnet_b0.pt --head-only")


if __name__ == "__main__":
    main()
