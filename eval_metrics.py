"""Full evaluation for the Metrics page (no CIFAKE).

Evaluates ONE checkpoint on TWO test sets:
  1. data/HQ/test    (in-distribution for hq_* checkpoints)
  2. data/AUTO/test  (UNSEEN for checkpoints trained before dl_auto.py)

Reports per set: acc, precision, recall, F1 (FAKE=positive), AUC,
confusion [[TN,FP],[FN,TP]], mean P(FAKE) per true class, per-source-bucket
accuracy (filename-prefix probe for generator generalisation).

Writes weights/metrics.json (served by GET /api/metrics).

    python eval_metrics.py
    python eval_metrics.py --checkpoint weights/hq_efficientnet_b0.pt --batch 32
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as T
from tqdm import tqdm

import config
from ai_detector.model import build_model
from train import flip_fake_target


def evaluate_set(model, device, root: str, img_size: int, batch: int, desc: str) -> dict:
    tf = T.Compose([
        T.Resize(int(img_size * 1.15)), T.CenterCrop(img_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=list(config.IMAGENET_MEAN), std=list(config.IMAGENET_STD)),
    ])
    ds = datasets.ImageFolder(os.path.join(root, "test"),
                              transform=tf, target_transform=flip_fake_target)
    ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    # rows = true (REAL=0, FAKE=1), cols = predicted
    conf = torch.zeros(2, 2, dtype=torch.long)
    sum_fake = torch.zeros(2)
    n = torch.zeros(2)
    y_all, p_all = [], []
    # per-source bucket: filename prefix before 2nd underscore (e.g. auto_defact, hq_fake, mj)
    buckets: dict[str, list[int]] = {}
    paths = [s[0] for s in ds.samples]
    with torch.inference_mode():
        i = 0
        for x, y in tqdm(ld, desc=desc, unit="batch"):
            x = x.to(device, memory_format=torch.channels_last)
            probs = model(x).sigmoid().reshape(-1).float().cpu()
            pred = (probs >= config.THRESHOLD).long()
            for t, pp, pr in zip(y.tolist(), pred.tolist(), probs.tolist()):
                conf[t][pp] += 1
                sum_fake[t] += pr
                n[t] += 1
                y_all.append(t)
                p_all.append(pr)
                parts = os.path.basename(paths[i]).split("_")
                tag = parts[0] if len(parts) < 3 else f"{parts[0]}_{parts[1]}"
                buckets.setdefault(f"{tag}/{'FAKE' if t else 'REAL'}", [0, 0])
                buckets[f"{tag}/{'FAKE' if t else 'REAL'}"][1] += 1
                if pp == t:
                    buckets[f"{tag}/{'FAKE' if t else 'REAL'}"][0] += 1
                i += 1
    total = conf.sum().item()
    tn, fp, fn, tp = conf[0][0].item(), conf[0][1].item(), conf[1][0].item(), conf[1][1].item()
    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_all, p_all)) if len(set(y_all)) > 1 else None
    except Exception:
        auc = None
    # threshold sweep (FAR = FP/(FP+TN), FRR = FN/(FN+TP)) for calibration report
    sweep, best = [], {"thr": config.THRESHOLD, "f1": round(f1, 4)}
    import numpy as _np
    yt, pp_ = _np.array(y_all), _np.array(p_all)
    for thr in [round(x * 0.05, 2) for x in range(2, 20)]:
        pr = (pp_ >= thr).astype(int)
        tp_ = int(((pr == 1) & (yt == 1)).sum())
        fp_ = int(((pr == 1) & (yt == 0)).sum())
        fn_ = int(((pr == 0) & (yt == 1)).sum())
        tn_ = int(((pr == 0) & (yt == 0)).sum())
        p_ = tp_ / max(tp_ + fp_, 1)
        r_ = tp_ / max(tp_ + fn_, 1)
        f_ = 2 * p_ * r_ / max(p_ + r_, 1e-9)
        sweep.append({"thr": thr, "acc": round((tp_ + tn_) / max(total, 1), 4),
                      "f1": round(f_, 4),
                      "far": round(fp_ / max(fp_ + tn_, 1), 4),
                      "frr": round(fn_ / max(fn_ + tp_, 1), 4)})
        if f_ > best["f1"]:
            best = {"thr": thr, "f1": round(f_, 4)}
    return {
        "threshold_sweep": sweep,
        "best_f1_threshold": best,
        "n": total,
        "accuracy": round(acc, 4),
        "precision_fake": round(prec, 4),
        "recall_fake": round(rec, 4),
        "f1_fake": round(f1, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "mean_p_fake_on_real": round(float(sum_fake[0] / max(n[0], 1)), 4),
        "mean_p_fake_on_fake": round(float(sum_fake[1] / max(n[1], 1)), 4),
        "per_source": {k: {"correct": v[0], "total": v[1],
                           "acc": round(v[0] / max(v[1], 1), 4)} for k, v in sorted(buckets.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.getenv("TORCH_WEIGHTS",
                    os.path.join(config.WEIGHTS_DIR, "hq_efficientnet_b0.pt")))
    ap.add_argument("--hq", default="./data/HQ")
    ap.add_argument("--auto", default="./data/AUTO")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(config.WEIGHTS_DIR, "metrics.json"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone = ckpt.get("backbone", "efficientnet_b0") if isinstance(ckpt, dict) else "efficientnet_b0"
    img_size = int(ckpt.get("img_size", config.IMAGE_SIZE)) if isinstance(ckpt, dict) else config.IMAGE_SIZE
    model = build_model(backbone, pretrained=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt, strict=False)
    model.eval().to(device, memory_format=torch.channels_last)
    print(f"checkpoint: {args.checkpoint} (saved acc={ckpt.get('acc', '?')}, img={img_size})")

    out = {
        "checkpoint": os.path.basename(args.checkpoint),
        "backbone": backbone,
        "saved_val_acc": ckpt.get("acc") if isinstance(ckpt, dict) else None,
        "img_size": img_size,
        "threshold": config.THRESHOLD,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sets": {},
    }
    if os.path.isdir(os.path.join(args.hq, "test")):
        r = evaluate_set(model, device, args.hq, img_size, args.batch, "HQ-test")
        r["note"] = ("Original HQ distribution (COCO reals + MJ/nano/art fakes). "
                     "Checkpoints retrained on AUTO may show mild forgetting here "
                     "vs their original HQ val acc.")
        out["sets"]["HQ-test"] = r
    if os.path.isdir(os.path.join(args.auto, "test")):
        r = evaluate_set(model, device, args.auto, img_size, args.batch, "AUTO-test")
        r["note"] = ("Modern generators (Defactify: SD2.1/SDXL/SD3/DALLE3/MJv6; "
                     "v3: SD1.5/SDXL/FLUX/Kandinsky/PixArt/Wuerstchen; "
                     "Tiny-GenImage: ADM/BigGAN/GLIDE/MJ/SD1.4/SD1.5/VQDM/Wukong). "
                     "Unseen for pre-AUTO checkpoints; in-distribution for AUTO-trained ones.")
        out["sets"]["AUTO-test"] = r
    out["eval_seconds"] = round(time.perf_counter() - t0, 1)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out} ({out['eval_seconds']}s)")
    for name, s in out["sets"].items():
        c = s["confusion"]
        print(f"{name}: acc={s['accuracy']} P={s['precision_fake']} R={s['recall_fake']} "
              f"F1={s['f1_fake']} AUC={s['auc']} | TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")


if __name__ == "__main__":
    main()
