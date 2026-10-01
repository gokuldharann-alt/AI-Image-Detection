"""One-command data setup for a fresh clone (stdlib only — runs before deps exist).

Chain (each step skips if its output already exists):
    1. data/CIFAKE/            <- MANUAL (Kaggle-gated) or `kagglehub`
    2. data/_raw/{mj,nano,art} <- python dl_newer.py          (~1.2 GB, needs huggingface_hub)
    3. data/HQ_SRC_REAL/       <- python dl_hq_real.py        (~900 MB COCO, stdlib only)
    4. data/NEWER/{train,test}/{REAL,FAKE} <- python build_newer.py (needs pandas/pyarrow/PIL)
    5. data/HQ/{train,test}/{REAL,FAKE}    <- python build_hq.py   (needs PIL)

Usage:
    python setup_data.py --check          # report what exists / what is missing
    python setup_data.py --dry-run        # show the plan without downloading anything
    python setup_data.py                  # full run (resumable, skips finished steps)
    python setup_data.py --only newer     # just one stage: cifake|raw|real|newer|hq
    python setup_data.py --skip raw       # skip stages (repeatable)
    python setup_data.py --auto-install   # pip-install missing helper deps

Next after HQ exists:
    python train.py --data ./data/HQ --backbone efficientnet_b0 --img-size 384
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CIFAKE = os.path.join("data", "CIFAKE")
RAW_MJ = os.path.join("data", "_raw", "mj", "data", "train-00000-of-00402.parquet")
RAW_NANO = os.path.join("data", "_raw", "nano", "data", "train-00000-of-00009.parquet")
RAW_ART = os.path.join("data", "_raw", "art", "Data")
REAL_SRC = os.path.join("data", "HQ_SRC_REAL")
NEWER = os.path.join("data", "NEWER")
HQ = os.path.join("data", "HQ")

KAGGLE_SLUG = "birdy654/cifake-real-and-ai-generated-synthetic-images"
MISSING_DEPS_MSG = "pip install huggingface_hub pandas pyarrow Pillow"

# (stage name, output probe dirs/files) — non-empty probe => stage done.
STAGES = ("cifake", "raw", "real", "newer", "hq")


def count_images(d: str) -> int:
    n = 0
    if not os.path.isdir(d):
        return 0
    for _, _, files in os.walk(d):
        n += sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")))
    return n


def split_counts(root: str) -> dict:
    return {
        f"{sp}/{cls}": count_images(os.path.join(root, sp, cls))
        for sp in ("train", "test") for cls in ("REAL", "FAKE")
    }


def status() -> dict:
    return {
        "cifake": split_counts(CIFAKE),
        "raw": {"mj": os.path.isfile(RAW_MJ), "nano": os.path.isfile(RAW_NANO), "art": os.path.isdir(RAW_ART)},
        "real_src": count_images(REAL_SRC),
        "newer": split_counts(NEWER),
        "hq": split_counts(HQ),
    }


def done(stage: str, st: dict) -> bool:
    if stage == "cifake":
        return all(v > 0 for v in st["cifake"].values())
    if stage == "raw":
        return bool(st["raw"]["mj"] and st["raw"]["nano"] and st["raw"]["art"])
    if stage == "real":
        return st["real_src"] > 0
    if stage in ("newer", "hq"):
        return all(v > 0 for v in st[stage].values())
    return False


def missing_modules() -> list[str]:
    out = []
    for mod in ("huggingface_hub", "pandas", "pyarrow", "PIL"):
        try:
            __import__(mod)
        except ImportError:
            out.append(mod)
    return out


def run(cmd: list[str], dry: bool) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    if dry:
        print("  (dry-run: skipped)", flush=True)
        return
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(cmd)}")


def ensure_cifake(args) -> bool:
    """Return True if CIFAKE usable. Kaggle is gated: try kagglehub, else instruct."""
    st = status()
    if done("cifake", st):
        print(f"[skip] CIFAKE present: {st['cifake']}", flush=True)
        return True
    if args.dry_run:
        print("[plan] CIFAKE missing -> would try kagglehub, else print manual steps.", flush=True)
        return False
    try:
        from kagglehub import dataset_download  # type: ignore

        print(f"downloading Kaggle {KAGGLE_SLUG} via kagglehub ...", flush=True)
        dataset_download(KAGGLE_SLUG, path=CIFAKE)
    except ImportError:
        print(f"""CIFAKE not found at {CIFAKE}/ and `kagglehub` is not installed.
Manual fix (Kaggle dataset is gated):
  1. Visit kaggle.com/datasets/{KAGGLE_SLUG} -> Download
  2. Extract so you have: data/CIFAKE/train/{{REAL,FAKE}} + data/CIFAKE/test/{{REAL,FAKE}}
Or auto: pip install kagglehub && python setup_data.py (needs Kaggle credentials).""")
        return False
    except Exception as exc:
        print(f"kagglehub download failed ({exc}). Fall back to manual Kaggle download.", flush=True)
        return False
    ok = done("cifake", status())
    print(f"[{'ok' if ok else 'MISSING'}] CIFAKE: {status()['cifake']}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Fresh-clone data setup: CIFAKE -> _raw -> NEWER -> HQ.")
    ap.add_argument("--check", action="store_true", help="Report data status and exit.")
    ap.add_argument("--dry-run", action="store_true", help="Show commands without running them.")
    ap.add_argument("--only", choices=STAGES, default=None, help="Run a single stage.")
    ap.add_argument("--skip", action="append", choices=STAGES, default=[], help="Skip stages (repeatable).")
    ap.add_argument("--auto-install", action="store_true", help="pip-install missing helper deps.")
    ap.add_argument("--cifake", default="./data/CIFAKE")
    ap.add_argument("--newer", default="./data/NEWER")
    ap.add_argument("--real-src", default="./data/HQ_SRC_REAL")
    ap.add_argument("--out", default="./data/HQ")
    ap.add_argument("--max-train-fake", type=int, default=4000)
    ap.add_argument("--max-test-fake", type=int, default=1000)
    args = ap.parse_args()

    st = status()
    if args.check or args.dry_run:
        print(json.dumps(st, indent=2))
        for s in STAGES:
            print(f"  {s}: {'DONE' if done(s, st) else 'missing'}")
        miss = missing_modules()
        if miss:
            print(f"missing python pkgs for dl/build: {miss}\n  -> {MISSING_DEPS_MSG}")
        if args.check:
            return

    wanted = [args.only] if args.only else [s for s in STAGES if s not in args.skip]
    print(f"stages: {wanted}{' (dry-run)' if args.dry_run else ''}", flush=True)
    print("disk note: ~2.1 GB downloads (1.2 GB HF shards + 900 MB COCO) + built copies.", flush=True)

    if "cifake" in wanted and not ensure_cifake(args):
        raise SystemExit("CIFAKE unresolved — see instructions above. Re-run after placing it.")

    if "raw" in wanted:
        if done("raw", status()):
            print("[skip] _raw shards present.", flush=True)
        else:
            miss = [m for m in missing_modules() if m in ("huggingface_hub",)]
            if miss and not args.dry_run:
                if args.auto_install:
                    run([sys.executable, "-m", "pip", "install", "huggingface_hub"], args.dry_run)
                else:
                    raise SystemExit(f"dl_newer.py needs huggingface_hub. Run: pip install huggingface_hub (or --auto-install)")
            run([sys.executable, "dl_newer.py"], args.dry_run)

    if "real" in wanted:
        if done("real", status()):
            print(f"[skip] HQ_SRC_REAL present ({status()['real_src']} imgs).", flush=True)
        else:
            run([sys.executable, "dl_hq_real.py"], args.dry_run)

    if "newer" in wanted:
        if done("newer", status()):
            print(f"[skip] NEWER present: {status()['newer']}", flush=True)
        else:
            miss = [m for m in missing_modules() if m in ("pandas", "pyarrow", "PIL")]
            if miss and not args.dry_run:
                if args.auto_install:
                    run([sys.executable, "-m", "pip", "install", "pandas", "pyarrow", "Pillow"], args.dry_run)
                else:
                    raise SystemExit(f"build_newer.py needs {miss}. Run: {MISSING_DEPS_MSG} (or --auto-install)")
            run([sys.executable, "build_newer.py", "--cifake", args.cifake, "--out", args.newer], args.dry_run)

    if "hq" in wanted:
        if done("hq", status()):
            print(f"[skip] HQ present: {status()['hq']}", flush=True)
        else:
            run([sys.executable, "build_hq.py", "--newer", args.newer, "--real-src", args.real_src,
                 "--out", args.out, "--max-train-fake", str(args.max_train_fake),
                 "--max-test-fake", str(args.max_test_fake)], args.dry_run)

    if args.dry_run:
        print("dry-run done — no changes made.", flush=True)
        return

    final = status()
    with open(os.path.join("data", "MANIFEST.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("FINAL:", json.dumps(final["hq"] if "hq" in wanted else final, indent=2))
    if done("hq", final):
        print("Next: python train.py --data ./data/HQ --backbone efficientnet_b0 --img-size 384")
    else:
        print("Note: HQ not built yet (see missing stages above).")


if __name__ == "__main__":
    main()
