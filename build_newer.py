"""Assemble data/NEWER — CIFAKE + newer-generator fakes (MJ, NanoBanana-mix, art).

Layout out (ImageFolder, balanced, seed-fixed):
    data/NEWER/{train,test}/{REAL,FAKE}/

Sources (FAKE): CIFAKE-train/test + GenImage MidJourney shard + MJ/DALLE/SD/
NanoBanana shard + downloaded art fakes. Sources (REAL): CIFAKE-train/test.
CIFAKE train split feeds NEWER train, CIFAKE test feeds NEWER test (no leak).

    python build_newer.py --cifake ./data/CIFAKE
"""
from __future__ import annotations

import argparse
import io
import os
import random
import shutil

import pandas as pd
from PIL import Image

N_MJ_TRAIN, N_NANO_TRAIN, N_ART_TRAIN, N_CIFAKE_FAKE_TRAIN = 320, 950, 1500, 3000
SEED = 7


def save_parquet_images(parquet: str, dest: str, prefix: str, take: int, skip: int = 0) -> int:
    """Extract [skip:skip+take] images (original bytes) -> dest. Returns saved count."""
    os.makedirs(dest, exist_ok=True)
    df = pd.read_parquet(parquet, columns=["image"])
    n = 0
    for i in range(skip, min(skip + take, len(df))):
        cell = df["image"].iloc[i]
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        if not raw:
            continue
        ext = ".png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
        try:  # validate before keeping (skip corrupt)
            Image.open(io.BytesIO(raw)).verify()
        except Exception:  # noqa: BLE001
            continue
        with open(os.path.join(dest, f"{prefix}_{n:05d}{ext}"), "wb") as f:
            f.write(raw)
        n += 1
    return n


def copy_sample(src_dir: str, dest: str, take: int, prefix: str, rng: random.Random) -> int:
    files = sorted(f for f in os.listdir(src_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")))
    rng.shuffle(files)
    os.makedirs(dest, exist_ok=True)
    for i, fn in enumerate(files[:take]):
        shutil.copy2(os.path.join(src_dir, fn), os.path.join(dest, f"{prefix}_{i:05d}{os.path.splitext(fn)[1]}"))
    return min(take, len(files))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifake", default="./data/CIFAKE")
    ap.add_argument("--out", default="./data/NEWER")
    args = ap.parse_args()
    rng = random.Random(SEED)
    tr_f = os.path.join(args.out, "train", "FAKE")
    te_f = os.path.join(args.out, "test", "FAKE")
    tr_r = os.path.join(args.out, "train", "REAL")
    te_r = os.path.join(args.out, "test", "REAL")

    got = {}
    got["mj_train"] = save_parquet_images("data/_raw/mj/data/train-00000-of-00402.parquet",
                                          tr_f, "mj", N_MJ_TRAIN, skip=0)
    got["mj_test"] = save_parquet_images("data/_raw/mj/data/train-00000-of-00402.parquet",
                                         te_f, "mj", 83, skip=N_MJ_TRAIN)
    got["nano_train"] = save_parquet_images("data/_raw/nano/data/train-00000-of-00009.parquet",
                                            tr_f, "nano", N_NANO_TRAIN, skip=0)
    got["nano_test"] = save_parquet_images("data/_raw/nano/data/train-00000-of-00009.parquet",
                                           te_f, "nano", 239, skip=N_NANO_TRAIN)
    # art: single shuffle, disjoint slices (train 1500 / test 400)
    art_files = sorted(f for f in os.listdir("data/_raw/art/Data/FAKE")
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")))
    rng.shuffle(art_files)
    os.makedirs(tr_f, exist_ok=True)
    os.makedirs(te_f, exist_ok=True)
    got_art_train = got_art_test = skipped = 0
    for j, fn in enumerate(art_files[: N_ART_TRAIN + 400]):
        try:
            with Image.open(os.path.join("data/_raw/art/Data/FAKE", fn)) as im:
                im.verify()
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        sub = tr_f if j < N_ART_TRAIN else te_f
        pre = "art" if j < N_ART_TRAIN else "artt"
        shutil.copy2(os.path.join("data/_raw/art/Data/FAKE", fn),
                     os.path.join(sub, f"{pre}_{j:05d}{os.path.splitext(fn)[1]}"))
        if j < N_ART_TRAIN:
            got_art_train += 1
        else:
            got_art_test += 1
    got["art_train"], got["art_test"], got["art_skipped"] = got_art_train, got_art_test, skipped

    n_fake_train = N_CIFAKE_FAKE_TRAIN + got["mj_train"] + got["nano_train"] + got["art_train"]
    got["cifake_fake_train"] = copy_sample(os.path.join(args.cifake, "train", "FAKE"),
                                           tr_f, N_CIFAKE_FAKE_TRAIN, "cifake", rng)
    got["cifake_fake_test"] = copy_sample(os.path.join(args.cifake, "test", "FAKE"),
                                          te_f, 750, "cifake", rng)
    n_fake_test = 750 + got["mj_test"] + got["nano_test"] + got["art_test"]

    # REAL side mirrors FAKE counts using CIFAKE only (plentiful + clean labels).
    got["cifake_real_train"] = copy_sample(os.path.join(args.cifake, "train", "REAL"),
                                           tr_r, n_fake_train, "cifake", rng)
    got["cifake_real_test"] = copy_sample(os.path.join(args.cifake, "test", "REAL"),
                                          te_r, n_fake_test, "cifake", rng)

    print("built data/NEWER:")
    for k, v in got.items():
        print(f"  {k}: {v}")
    print(f"  TOTAL train FAKE={n_fake_train} REAL={n_fake_train} | "
          f"test FAKE={n_fake_test} REAL={n_fake_test}")


if __name__ == "__main__":
    main()
