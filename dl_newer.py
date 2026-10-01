"""Download newer-generator data (one-time setup, ~1.2 GB).

- bitmind/GenImage_MidJourney : 1 parquet shard (MidJourney fakes)
- julienlucas/...-nanobanapro : 1 train shard (MJ / DALL-E / SD / NanoBanana fakes)
- hmnshudhmn24/...-art-images : imagefolder (art REAL + FAKE)
"""
from __future__ import annotations

import os

from huggingface_hub import hf_hub_download, snapshot_download

RAW = os.path.join("data", "_raw")
os.makedirs(RAW, exist_ok=True)

print("downloading MidJourney shard (526 MB)...", flush=True)
mj = hf_hub_download(
    "bitmind/GenImage_MidJourney",
    filename="data/train-00000-of-00402.parquet",
    repo_type="dataset",
    local_dir=os.path.join(RAW, "mj"),
)
print("saved", mj, flush=True)

print("downloading nanobanana-mix shard (497 MB)...", flush=True)
nb = hf_hub_download(
    "julienlucas/midjourney-dalle-sd-nanobananapro-dataset",
    filename="data/train-00000-of-00009.parquet",
    repo_type="dataset",
    local_dir=os.path.join(RAW, "nano"),
)
print("saved", nb, flush=True)

print("downloading art imagefolder (small)...", flush=True)
art = snapshot_download(
    "hmnshudhmn24/real-fake-ai-generated-art-images",
    repo_type="dataset",
    local_dir=os.path.join(RAW, "art"),
    allow_patterns=["Data/*"],
)
print("saved", art, flush=True)
print("DOWNLOAD_DONE")
