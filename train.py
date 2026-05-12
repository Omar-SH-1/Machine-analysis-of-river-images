

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from dataset import build_dataloaders
from model import UNet


# ─────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────

CONFIG = {
    "patches_images_dir" : "data/patches/images",
    "patches_masks_dir"  : "data/patches/masks",
    "checkpoint_dir"     : "outputs/checkpoints",
    "results_dir"        : "outputs/results",
    "features"           : [64, 128, 256, 512],
    "epochs"             : 30,
    "batch_size"         : 4,
    "learning_rate"      : 1e-4,
    "val_size"           : 0.2,
    "num_workers"        : 0,
    "save_every_n_epochs": 5,
    "early_stopping"     : 7,
}


# ─────────────────────────────────────────────
# ФУНКЦИИ ПОТЕРЬ
# ─────────────────────────────────────────────

class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2 * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
   
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.dice  = DiceLoss()
        self.bce   = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.alpha * self.bce(logits, targets) + \
               (1 - self.alpha) * self.dice(logits, targets)


# ─────────────────────────────────────────────
# МЕТРИКИ
# ─────────────────────────────────────────────

def calculate_iou(logits, targets, threshold: float = 0.5) -> float:
    
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        intersection = (preds * targets).sum().item()
        union        = (preds + targets).clamp(0, 1).sum().item()
        if union == 0:
            return 1.0
        return intersection / union


# ─────────────────────────────────────────────
# ОДИН ПРОХОД ПО ДАННЫМ
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, mode="train"):
    is_train = (mode == "train")
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_iou  = 0.0
    n_batches  = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        pbar = tqdm(loader, desc=f"  {mode.upper():>5}", leave=False,
                    bar_format="{l_bar}{bar:20}{r_bar}")

        for images, masks in pbar:
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)
            loss   = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_iou   = calculate_iou(logits, masks)
            total_loss += loss.item()
            total_iou  += batch_iou
            n_batches  += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "iou" : f"{batch_iou:.4f}"
            })

    return total_loss / n_batches, total_iou / n_batches


# ─────────────────────────────────────────────
# ГРАФИКИ
# ─────────────────────────────────────────────

def save_training_plot(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val",   markersize=4)
    axes[0].set_title("Loss по эпохам")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_iou"], "b-o", label="Train", markersize=4)
    axes[1].plot(epochs, history["val_iou"],   "r-o", label="Val",   markersize=4)
    axes[1].axhline(y=0.75, color="g", linestyle="--", alpha=0.7, label="Цель: 0.75")
    axes[1].set_title("IoU по эпохам")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Кривые обучения U-Net — Река Кусарчай", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device.type.upper()}")
    if device.type == "cpu":
        print("  (GPU не найден — обучение на CPU)")

    Path(CONFIG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["results_dir"]).mkdir(parents=True, exist_ok=True)

    print("\nЗагрузка данных...")
    train_loader, val_loader = build_dataloaders(
        patches_images_dir=CONFIG["patches_images_dir"],
        patches_masks_dir =CONFIG["patches_masks_dir"],
        val_size          =CONFIG["val_size"],
        batch_size        =CONFIG["batch_size"],
        num_workers       =CONFIG["num_workers"],
    )

    print("\nСоздание модели...")
    model = UNet(in_channels=3, features=CONFIG["features"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Параметров: {total_params:,}")

    criterion = CombinedLoss(alpha=0.5)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_iou" : [], "val_iou" : []
    }
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    # Оценка времени
    print("\nОценка скорости...")
    sample_images, _ = next(iter(train_loader))
    t0 = time.time()
    with torch.no_grad():
        _ = model(sample_images.to(device))
    batch_time = time.time() - t0
    est_epoch_min = batch_time * len(train_loader) / 60
    print(f"  ~{batch_time:.1f}с/батч → ~{est_epoch_min:.0f} мин/эпоха "
          f"→ ~{est_epoch_min * CONFIG['epochs'] / 60:.1f} часов всего")

    print(f"\n{'='*55}")
    print(f"  Начинаем обучение: {CONFIG['epochs']} эпох")
    print(f"{'='*55}")

    for epoch in range(1, CONFIG["epochs"] + 1):
        t_start = time.time()
        print(f"\nЭпоха {epoch}/{CONFIG['epochs']}")

        train_loss, train_iou = run_epoch(
            model, train_loader, criterion, optimizer, device, mode="train"
        )
        val_loss, val_iou = run_epoch(
            model, val_loader, criterion, optimizer, device, mode="val"
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        elapsed    = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  Train  — loss: {train_loss:.4f}  IoU: {train_iou:.4f}")
        print(f"  Val    — loss: {val_loss:.4f}  IoU: {val_iou:.4f}")
        print(f"  LR: {current_lr:.2e}  |  Время: {elapsed:.0f}с")

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss"       : val_loss,
                "val_iou"        : val_iou,
                "config"         : CONFIG,
            }, Path(CONFIG["checkpoint_dir"]) / "best_model.pth")
            print(f"  ✓ Лучшая модель сохранена (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  Без улучшений: {epochs_no_improve}/{CONFIG['early_stopping']}")

        # Чекпоинт каждые N эпох
        if epoch % CONFIG["save_every_n_epochs"] == 0:
            ckpt_path = Path(CONFIG["checkpoint_dir"]) / f"epoch_{epoch:03d}.pth"
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_loss"   : val_loss,
                "val_iou"    : val_iou,
            }, ckpt_path)
            print(f"  Чекпоинт: {ckpt_path.name}")

        # Обновляем график
        save_training_plot(
            history,
            str(Path(CONFIG["results_dir"]) / "training_curves.png")
        )

        
        if epochs_no_improve >= CONFIG["early_stopping"]:
            print(f"\n⚡ Early stopping после {epoch} эпох")
            break

    best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1
    print(f"\n{'='*55}")
    print(f"  Обучение завершено!")
    print(f"  Лучшая эпоха   : {best_epoch}")
    print(f"  Лучший val IoU : {max(history['val_iou']):.4f}")
    print(f"  Лучший val loss: {min(history['val_loss']):.4f}")
    print(f"  Модель : outputs/checkpoints/best_model.pth")
    print(f"  График : outputs/results/training_curves.png")
    print(f"{'='*55}")


if __name__ == "__main__":
    train()