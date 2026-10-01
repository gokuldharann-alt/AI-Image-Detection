"""
Train the REAL vs FAKE classifier on CIFAKE.

Dataset layout (Kaggle: birdy654/cifake-real-and-ai-generated-synthetic-images):
    data/CIFAKE/
        train/REAL/*.jpg   train/FAKE/*.jpg
        test/REAL/*.jpg    test/FAKE/*.jpg
  (ImageFolder-compatible; label 0 = FAKE? we map: REAL=0, FAKE=1 via target_transform.)

Usage:
    pip install -r requirements.txt
    python train.py --data ./data/CIFAKE --epochs 12 --backbone efficientnet_b0 --batch 64

Outputs:
    weights/cifake_efficientnet_b0.pt    (best checkpoint, for app.py)
    weights/cifake_efficientnet_b0.onnx  (optional, for ONNX Runtime fast path)

Technique: full fine-tune from ImageNet, AdamW + cosine schedule, AMP,
label smoothing-ish BCEWithLogits, early stopping on val accuracy.
Expected: ~93-96% test accuracy with EfficientNet-B0 @ 12-15 epochs.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as T
from tqdm import tqdm

import config
from ai_detector.model import build_model


def flip_fake_target(idx: int) -> int:
    """Map ImageFolder index -> BCE target. Must be top-level (picklable).

    ImageFolder sorts CIFAKE classes alphabetically: FAKE=0, REAL=1.
    We train P(FAKE), so target = 1 - idx. A ``lambda`` here breaks
    Windows/multiprocess DataLoaders (spawn requires picklable callables).
    """
    return 1 - int(idx)


def build_loaders(data_dir: str, backbone: str, batch: int, workers: int, img_size: int | None = None):
    mean, std = list(config.IMAGENET_MEAN), list(config.IMAGENET_STD)
    IM = img_size or config.IMAGE_SIZE
    train_tf = T.Compose([
        T.RandomResizedCrop(IM, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1, 0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.ToImage(),  # PIL -> uint8 tensor (required before ToDtype/Normalize in tv v2)
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std),
    ])
    eval_tf = T.Compose([
        T.Resize(int(IM * 1.15)), T.CenterCrop(IM),
        T.ToImage(),  # PIL -> uint8 tensor (required before ToDtype/Normalize in tv v2)
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std),
    ])
    # CIFAKE folders are named FAKE/REAL — map alphabetically {FAKE:0, REAL:1} → we want P(FAKE).
    # ImageFolder sorts classes: FAKE=0, REAL=1. Target for BCE (1=FAKE): 1 - idx.
    # NOTE: use the top-level flip_fake_target (a lambda is NOT picklable with num_workers>0).
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf,
                                    target_transform=flip_fake_target)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_tf,
                                   target_transform=flip_fake_target)
    print(f"classes (raw): {train_ds.classes} -> mapped: 1=FAKE, 0=REAL")
    print(f"train: {len(train_ds)} | test: {len(test_ds)}")
    # pin_memory only helps with CUDA; on CPU-only Windows it just emits a warning.
    pin = torch.cuda.is_available()
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    test_ld = DataLoader(test_ds, batch_size=batch * 2, shuffle=False,
                         num_workers=workers, pin_memory=pin)
    return train_ld, test_ld


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, memory_format=torch.channels_last)
        y = y.float().to(device).unsqueeze(1)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * len(x)
        correct += ((logits.sigmoid() >= 0.5).float() == y).sum().item()
        total += len(x)
    return loss_sum / total, correct / total


def main() -> None:
    ap = argparse.ArgumentParser(description="Train REAL vs FAKE classifier (CIFAKE or HQ multi-generator)")
    ap.add_argument("--data", default="./data/CIFAKE")
    ap.add_argument("--backbone", default=config.MODEL_BACKBONE, choices=["efficientnet_b0", "resnet50", "convnext_tiny", "efficientnet_v2_s"])
    ap.add_argument("--img-size", type=int, default=None, help="Override IMAGE_SIZE (default: config.IMAGE_SIZE, now 384 for HQ).")
    ap.add_argument("--out", default="", help="Checkpoint filename (default: hq_<backbone>.pt for HQ/NEWER data, else cifake_<backbone>.pt).")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers. Use 0 on Windows/CPU (default); raise to 4+ on Linux+GPU.")
    ap.add_argument("--resume", default="",
                    help="Path to a checkpoint .pt to fine-tune FROM (default: ImageNet). "
                         "Use your CIFAKE weights when adapting to newer generators.")
    ap.add_argument("--head-only", action="store_true",
                    help="Freeze the backbone, train only the classifier head. "
                         "Best for small new datasets (<10k images).")
    ap.add_argument("--patience", type=int, default=4, help="early-stopping patience (epochs)")
    args = ap.parse_args()

    device = config.resolve_device()
    img_size = args.img_size or config.IMAGE_SIZE
    print(f"device: {config.device_label(device)} | backbone: {args.backbone} | img_size: {img_size} | data: {args.data}")

    train_ld, test_ld = build_loaders(args.data, args.backbone, args.batch, args.workers, img_size=img_size)
    model = build_model(args.backbone, pretrained=True)
    if args.resume:
        ckpt0 = torch.load(args.resume, map_location="cpu")
        ckpt_bb = ckpt0.get("backbone", None)
        if ckpt_bb and ckpt_bb != args.backbone:
            raise ValueError(f"Resume checkpoint is '{ckpt_bb}' but --backbone is "
                             f"'{args.backbone}'. They must match.")
        model.load_state_dict(ckpt0.get("state_dict", ckpt0), strict=False)
        print(f"resumed from {args.resume} (prev acc={ckpt0.get('acc', '?')})")
    if args.head_only:
        for _n, _p in model.named_parameters():
            _p.requires_grad = False
        if args.backbone in ("efficientnet_b0", "efficientnet_v2_s"):
            head = model.classifier[1]
        elif args.backbone == "convnext_tiny":
            head = model.classifier[2]
        else:
            head = model.fc
        for _p in head.parameters():
            _p.requires_grad = True
        print("head-only mode: backbone frozen, training classifier head only.")
    model.to(device, memory_format=torch.channels_last)

    criterion = nn.BCEWithLogitsLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc, bad_epochs = 0.0, 0
    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
    if args.out:
        ckpt_path = os.path.join(config.WEIGHTS_DIR, args.out)
    else:
        prefix = "hq" if any(k in os.path.abspath(args.data).upper() for k in ("HQ", "NEWER")) else "cifake"
        ckpt_path = os.path.join(config.WEIGHTS_DIR, f"{prefix}_{args.backbone}.pt")
    # Persist img_size so inference/export can warn on mismatch.
    print(f"checkpoint -> {ckpt_path}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, t0 = 0.0, time.perf_counter()
        pbar = tqdm(train_ld, desc=f"epoch {epoch}/{args.epochs}", unit="batch")
        for x, y in pbar:
            x = x.to(device, memory_format=torch.channels_last)
            y = y.float().to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(x)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()

        train_loss = running / len(train_ld.dataset)
        val_loss, val_acc = evaluate(model, test_ld, device)
        dt = time.perf_counter() - t0
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} ({dt:.0f}s, {len(train_ld.dataset)/dt:.0f} img/s)")

        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            torch.save({"backbone": args.backbone, "acc": val_acc, "img_size": img_size,
                        "state_dict": model.state_dict()}, ckpt_path)
            print(f"  * new best -> saved {ckpt_path} (acc={val_acc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping (no improvement for {args.patience} epochs).")
                break

    print(f"\nBEST val_acc={best_acc:.4f} -> {ckpt_path}")
    print("Next: python export_onnx.py  ->  uvicorn app:app --port 8000")


if __name__ == "__main__":
    main()
