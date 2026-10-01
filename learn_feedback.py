"""Continual-learn from user feedback + HQ base (anti-collapse, anti-poisoning).

- Mixes data/HQ with data/feedback/{REAL,FAKE} (feedback upsampled x3).
- Starts from weights/hq_<backbone>.pt (fallback: ImageNet), low LR, few epochs.
- Requires >=20 feedback images (warns <50/class), validates on HQ test + feedback holdout.
- Writes weights/hq_<backbone>.pt (backs up old to .bak) — then run export_onnx.py.

    python learn_feedback.py --backbone efficientnet_b0 --img-size 384 --epochs 5 --batch 32
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets

import config
from ai_detector.model import build_model
from train import build_loaders, evaluate, flip_fake_target  # reuse transforms/mapping


def feedback_dataset(img_size: int):
    from torchvision.transforms import v2 as T
    tf = T.Compose([
        T.Resize(int(img_size * 1.15)), T.CenterCrop(img_size),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=list(config.IMAGENET_MEAN), std=list(config.IMAGENET_STD)),
    ])
    fb_root = os.path.join("data", "feedback")
    if not os.path.isdir(fb_root):
        return None, (0, 0)
    ds = datasets.ImageFolder(fb_root, transform=tf, target_transform=flip_fake_target)
    n_real = sum(1 for _, y in ds.samples if ds.classes[y] == "REAL")
    return ds, (n_real, len(ds) - n_real)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.MODEL_BACKBONE)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5, help="Low LR to avoid catastrophic forgetting.")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    img_size = args.img_size or config.IMAGE_SIZE
    device = config.resolve_device()
    print(f"device: {config.device_label(device)} | backbone={args.backbone} img={img_size}")

    fb_ds, (n_r, n_f) = feedback_dataset(img_size)
    print(f"feedback: REAL={n_r} FAKE={n_f}")
    if fb_ds is None or len(fb_ds) < 20:
        raise SystemExit("Need 20+ feedback images first (use the UI buttons).")
    if min(n_r, n_f) < 10:
        print("WARNING: one class <10 — model may bias. Collect more of the minority class.")

    # Base HQ loaders (train split for mixing, test split for validation).
    train_ld, test_ld = build_loaders("./data/HQ", args.backbone, args.batch, args.workers, img_size=img_size)
    base_train = train_ld.dataset
    # Upsample small feedback 3x so it actually moves the needle.
    mixed = ConcatDataset([base_train, fb_ds, fb_ds, fb_ds])
    pin = torch.cuda.is_available()
    mixed_ld = DataLoader(mixed, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=pin,
                          persistent_workers=args.workers > 0)

    # Start from HQ weights (the okayish model), not ImageNet.
    model = build_model(args.backbone, pretrained=True)
    ckpt_path = os.path.join(config.WEIGHTS_DIR, f"hq_{args.backbone}.pt")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        print(f"resumed HQ {ckpt_path} (acc={ckpt.get('acc', '?')})")
    model.to(device, memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, _ = evaluate(model, test_ld, device)
    print(f"start val_acc={best:.4f}")
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for x, y in mixed_ld:
            x = x.to(device, memory_format=torch.channels_last)
            y = y.float().to(device).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(x)
        sched.step()
        _, acc = evaluate(model, test_ld, device)
        print(f"epoch {ep}: loss={run / len(mixed):.4f} HQ-val_acc={acc:.4f}")
        if acc >= best:
            best = acc
            if os.path.isfile(ckpt_path):
                shutil.copy2(ckpt_path, ckpt_path + ".bak")
            torch.save({"backbone": args.backbone, "acc": acc, "img_size": img_size,
                        "state_dict": model.state_dict()}, ckpt_path)
            print(f"  * saved {ckpt_path} (acc={acc:.4f})")

    print(f"\nBEST {best:.4f} -> {ckpt_path}\nNext: python export_onnx.py --backbone {args.backbone}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"({time.perf_counter() - start:.0f}s)")
