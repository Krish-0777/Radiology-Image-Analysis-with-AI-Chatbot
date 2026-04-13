"""
trainer.py - RadVision AI Model Training
=========================================
Trains ResNet-50 on a combined dataset (Cancer + Fracture + COVID-19 + Normal).
Usage:
    python trainer.py --data_dir ./data --epochs 30 --batch_size 32
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torch.amp import GradScaler, autocast
import numpy as np

CLASSES = ["Normal", "Chest Cancer", "Fracture", "COVID-19"]


# ─── Data Transforms ─────────────────────────────────────────────────────────
def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# ─── Model Builder ───────────────────────────────────────────────────────────
def build_model(num_classes=4, freeze_backbone=True):
    model = models.resnet50(weights="IMAGENET1K_V1")
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    # Unfreeze last 2 blocks
    for layer in [model.layer3, model.layer4, model.fc]:
        for param in layer.parameters():
            param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features if not isinstance(model.fc, nn.Sequential) else 2048, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


# ─── Weighted Sampler (handle class imbalance) ──────────────────────────────
def make_weighted_sampler(dataset):
    labels = [s[1] for s in dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


# ─── Trainer Class ───────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model, loaders, criterion, optimizer, scheduler, device, output_dir="outputs"):
        self.model = model.to(device)
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.scaler = GradScaler("cpu")
        os.makedirs(output_dir, exist_ok=True)
        self.best_acc = 0.0

    def _run_epoch(self, split="train"):
        is_train = split == "train"
        self.model.train(is_train)
        loader = self.loaders[split]
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(is_train):
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                if is_train:
                    self.optimizer.zero_grad()
                    with autocast("cpu"):
                        out = self.model(imgs)
                        loss = self.criterion(out, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    out = self.model(imgs)
                    loss = self.criterion(out, labels)

                total_loss += loss.item() * imgs.size(0)
                preds = out.argmax(1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)

        return total_loss / total, correct / total

    def fit(self, epochs=30, patience=7):
        no_improve = 0
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch("train")
            val_loss, val_acc = self._run_epoch("val")
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), f"{self.output_dir}/best_model.pt")
                print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"\nTraining complete. Best Val Accuracy: {self.best_acc:.4f}")
        print(f"Model saved → {self.output_dir}/best_model.pt")


# ─── Entry Point ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train RadVision AI")
    parser.add_argument("--data_dir", default="./data", help="Root data directory (with train/val/test subfolders)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    splits = {}
    loaders = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(args.data_dir, split)
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        ds = datasets.ImageFolder(path, transform=get_transforms(split))
        splits[split] = ds
        if split == "train":
            sampler = make_weighted_sampler(ds)
            loaders[split] = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=False)
        else:
            loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)
        print(f"{split}: {len(ds)} images, classes: {ds.classes}")

    num_classes = len(splits["train"].classes) if "train" in splits else 4
    model = build_model(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    trainer = Trainer(model, loaders, criterion, optimizer, scheduler, device, args.output_dir)
    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
