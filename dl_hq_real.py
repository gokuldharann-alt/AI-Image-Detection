"""Download high-res REAL photos (COCO val2017, ~5k images, 640px) into data/HQ_SRC_REAL/.

COCO photos are real-world, high-quality, permissively usable for research,
and a big step up from CIFAKE 32px CIFAR images.

    python dl_hq_real.py
"""
from __future__ import annotations

import os
import zipfile

import urllib.request

URL = "http://images.cocodataset.org/zips/val2017.zip"
OUT_SRC = os.path.join("data", "HQ_SRC_REAL")
TMP = os.path.join("data", "_raw", "val2017.zip")


def main() -> None:
    os.makedirs(os.path.dirname(TMP), exist_ok=True)
    os.makedirs(OUT_SRC, exist_ok=True)
    if not os.path.isfile(TMP):
        print(f"downloading COCO val2017 (~900MB) -> {TMP} ...", flush=True)
        urllib.request.urlretrieve(URL, TMP)
    print("extracting (only .jpg into HQ_SRC_REAL/) ...", flush=True)
    with zipfile.ZipFile(TMP) as z:
        for m in z.infolist():
            if not m.filename.lower().endswith(".jpg"):
                continue
            # z path: val2017/000000000139.jpg -> OUT_SRC/...
            fn = os.path.basename(m.filename)
            target = os.path.join(OUT_SRC, f"coco_{fn}")
            if os.path.isfile(target):
                continue
            with z.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
    n = len([f for f in os.listdir(OUT_SRC) if f.lower().endswith(".jpg")])
    print(f"DONE: {n} real photos in {OUT_SRC}")
    print("Next: python build_hq.py --newer ./data/NEWER --real-src ./data/HQ_SRC_REAL --out ./data/HQ")


if __name__ == "__main__":
    main()
