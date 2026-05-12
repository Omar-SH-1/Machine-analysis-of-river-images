
import os
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────
# ШАГ 1: Нарезка снимков на патчи
# ─────────────────────────────────────────────

def create_patches(
    images_dir: str,
    masks_dir: str,
    output_dir: str,
    patch_size: int = 256,
    stride: int = 128,
    min_river_ratio: float = 0.01  
):
 
    out_images = Path(output_dir) / "patches" / "images"
    out_masks  = Path(output_dir) / "patches" / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    image_files = sorted(Path(images_dir).glob("*.jpg")) + \
                  sorted(Path(images_dir).glob("*.png"))

    total_saved = 0
    total_skipped = 0

    for img_path in image_files:
        # Ищем соответствующую маску (то же имя, расширение .png)
        mask_path = Path(masks_dir) / (img_path.stem + ".png")
        if not mask_path.exists():
            print(f"  [!] Маска не найдена для {img_path.name}, пропускаем")
            continue

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Бинаризация маски на случай артефактов сжатия (0 или 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        H, W = image.shape[:2]
        patch_count = 0

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                img_patch  = image[y:y+patch_size, x:x+patch_size]
                mask_patch = mask[y:y+patch_size,  x:x+patch_size]

                # Пропускаем патчи с почти нулевым содержанием реки
                river_ratio = np.sum(mask_patch > 0) / (patch_size * patch_size)
                if river_ratio < min_river_ratio:
                    total_skipped += 1
                    continue

                name = f"{img_path.stem}_y{y}_x{x}.png"
                cv2.imwrite(str(out_images / name),
                            cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out_masks / name), mask_patch)

                patch_count += 1
                total_saved += 1

        print(f"  ✓ {img_path.name}: сохранено {patch_count} патчей")

    print(f"\n{'─'*50}")
    print(f"Итого сохранено : {total_saved} патчей")
    print(f"Пропущено (пустые): {total_skipped} патчей")
    print(f"Патчи в: {out_images.parent}")
    return str(out_images), str(out_masks)


# ─────────────────────────────────────────────
# ШАГ 2: PyTorch Dataset
# ─────────────────────────────────────────────

def get_transforms(mode: str = "train"):

    if mode == "train":
        return A.Compose([

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1,
                rotate_limit=15, p=0.4
            ),

            
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20,
                val_shift_limit=10, p=0.3
            ),
            A.GaussNoise(var_limit=(10, 50), p=0.2),

            # Нормализация (ImageNet mean/std — стандарт для RGB снимков)
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ])
    else:  # val / test
        return A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ])


class RiverDataset(Dataset):
  
    def __init__(self, image_paths: list, mask_paths: list, transform=None):
        assert len(image_paths) == len(mask_paths), \
            "Количество изображений и масок должно совпадать!"
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Маска: 0.0 = фон, 1.0 = река
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # Tensor [3, H, W]
            mask  = augmented["mask"]    # Tensor [H, W]
            mask  = mask.unsqueeze(0)    # → [1, H, W] нужно для BCELoss

        return image, mask


# ─────────────────────────────────────────────
# ШАГ 3: Сборка DataLoader-ов
# ─────────────────────────────────────────────

def build_dataloaders(
    patches_images_dir: str,
    patches_masks_dir: str,
    val_size: float = 0.2,
    batch_size: int = 4,        
    num_workers: int = 0,      
    seed: int = 42
):

    images_dir = Path(patches_images_dir)
    masks_dir  = Path(patches_masks_dir)

   
    image_paths = sorted([str(p) for p in images_dir.glob("*.png")])
    mask_paths  = sorted([str(p) for p in masks_dir.glob("*.png")])

    assert len(image_paths) > 0, f"Патчи не найдены в {images_dir}"
    assert len(image_paths) == len(mask_paths), \
        f"Не совпадает кол-во изображений ({len(image_paths)}) и масок ({len(mask_paths)})"

    print(f"Всего патчей: {len(image_paths)}")

   
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths,
        test_size=val_size,
        random_state=seed
    )

    print(f"  Train: {len(train_imgs)} патчей")
    print(f"  Val  : {len(val_imgs)} патчей")

    train_dataset = RiverDataset(
        train_imgs, train_masks,
        transform=get_transforms("train")
    )
    val_dataset = RiverDataset(
        val_imgs, val_masks,
        transform=get_transforms("val")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False    # pin_memory=True только для GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────
# ПРОВЕРКА — запусти этот файл напрямую
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ── Настрой эти пути под свою структуру ──
    IMAGES_DIR = "dataset/images"
    MASKS_DIR  = "dataset/masks"
    OUTPUT_DIR = "data"
    # ──────────────────────────────────────────

    print("=" * 50)
    print("ШАГ 1: Нарезка снимков на патчи...")
    print("=" * 50)

    patches_img_dir, patches_mask_dir = create_patches(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        output_dir=OUTPUT_DIR,
        patch_size=256,
        stride=128,
        min_river_ratio=0.01
    )

    print("\n" + "=" * 50)
    print("ШАГ 2: Создание DataLoader-ов...")
    print("=" * 50)

    train_loader, val_loader = build_dataloaders(
        patches_images_dir=patches_img_dir,
        patches_masks_dir=patches_mask_dir,
        val_size=0.2,
        batch_size=4,
        num_workers=0
    )

    print("\n" + "=" * 50)
    print("ШАГ 3: Визуализация одного батча...")
    print("=" * 50)

    # Берём первый батч 
    images, masks = next(iter(train_loader))
    print(f"Размер батча изображений : {images.shape}")  # [4, 3, 256, 256]
    print(f"Размер батча масок       : {masks.shape}")   # [4, 1, 256, 256]
    print(f"Диапазон значений масок  : [{masks.min():.1f}, {masks.max():.1f}]")

    # Визуализация 
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i in range(min(4, len(images))):
        # Денормализация 
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * std + mean).clip(0, 1)

        msk = masks[i].squeeze().numpy()

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Снимок #{i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(msk, cmap="gray")
        axes[1, i].set_title(f"Маска #{i+1}")
        axes[1, i].axis("off")

    plt.suptitle("Примеры патчей из train_loader", fontsize=14)
    plt.tight_layout()
    plt.savefig("data/sample_patches.png", dpi=100)
    plt.show()
    print("\n✓ Всё работает! График сохранён: data/sample_patches.png")
