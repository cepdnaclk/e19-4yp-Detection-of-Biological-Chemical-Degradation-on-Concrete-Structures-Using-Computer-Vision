#!/usr/bin/env python3
"""
Train an image classifier on 3 degradation mechanisms
Usage:
    python train.py --data_root ./dataset --model efficientnetv2_s --epochs 25 --batch 32 --lr 1e-4
Supported models:
    efficientnetv2_s
    convnextv2_tiny
    swin_tiny_patch4_window7_224
"""

import argparse, os, time, json, copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
import timm
from tqdm import tqdm
import wandb  # optional; comment out if not using

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="dataset", type=str)
    p.add_argument("--model", default="efficientnetv2_s",
                   choices=["efficientnetv2_s", "convnextv2_tiny", "swin_tiny_patch4_window7_224"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=3, help="epochs with frozen backbone")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_wandb", action="store_true")
    return p.parse_args()

def seed_everything(seed):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_loaders(root, batch, workers):
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.GaussianBlur(3, sigma=(.1, 2.)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(Path(root)/"train", transform=train_tf)
    val_ds   = datasets.ImageFolder(Path(root)/"val",   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch*2, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

def build_model(name, num_classes):
    model = timm.create_model(name, pretrained=False, num_classes=num_classes)
    if hasattr(model, "classifier"):  # EfficientNet‑V2
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, "head"):  # ConvNeXt‑V2, Swin‑Tiny
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        f1.update(preds, labels)

    return running_loss / total, correct / total, f1.compute().item()

def main():
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader, classes = build_loaders(args.data_root, args.batch, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, len(classes)).to(device)

    # Freeze backbone for warm-up
    for p in model.parameters():
        p.requires_grad = False
    for p in model.parameters():
        if p.ndim == 2:
            p.requires_grad = True

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup)

    if args.log_wandb:
        wandb.init(project="concrete_degradation_cls", config=vars(args), name=f"{args.model}_{int(time.time())}")
        wandb.watch(model)

    best_model = None
    best_f1 = 0.0

    train_loss_hist, train_acc_hist = [], []
    val_loss_hist, val_acc_hist, val_f1_hist = [], [], []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup + 1:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, len(classes))
        scheduler.step()

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        val_f1_hist.append(val_f1)

        log = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                   val_loss=val_loss, val_acc=val_acc, val_f1=val_f1)
        print(json.dumps(log, indent=2))
        if args.log_wandb:
            wandb.log(log)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = copy.deepcopy(model).cpu()
            torch.save(best_model.state_dict(), f"best_{args.model}_seed{args.seed}.pt")

    if args.log_wandb:
        wandb.finish()

    # Create results directory for saving outputs
    results_dir = Path("results") / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    # ─── Evaluate Final Best Model ─────────────────────
    print("\nEvaluating the best saved model on validation set...")
    model.load_state_dict(torch.load(f"best_{args.model}_seed{args.seed}.pt"))
    model = model.to(device)

    val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, len(classes))

    print("\n===== FINAL EVALUATION RESULTS =====")
    print(f"Final Val Loss    : {val_loss:.4f}")
    print(f"Final Val Accuracy: {val_acc:.4f}")
    print(f"Final Val F1 Score: {val_f1:.4f}")
    print("=====================================\n")

    # Save final results to a text file
    results_txt_path = results_dir / "final_results.txt"
    with open(results_txt_path, "w") as f:
        f.write("FINAL EVALUATION RESULTS\n")
        f.write(f"Validation Loss    : {val_loss:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Validation F1 Score: {val_f1:.4f}\n")

    # ─── Plot Curves ─────────────────────
    epochs = range(1, args.epochs + 1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss_hist, label='Train')
    plt.plot(epochs, val_loss_hist,   label='Val')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc_hist, label='Train-Acc')
    plt.plot(epochs, val_acc_hist,   label='Val-Acc')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()
    plt.tight_layout()
    curves_path = results_dir / f"curves_{args.model}.png"
    plt.savefig(curves_path, dpi=200)
    plt.close()

    # ─── Confusion Matrix ────────────────
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Validation Confusion Matrix')
    cm_path = results_dir / f"cm_{args.model}.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print(f"Saved all results in {results_dir}")

if __name__ == "__main__":
    main()
