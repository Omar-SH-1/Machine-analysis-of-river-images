

import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(str(Path(__file__).parent))
from model import UNet


# ─────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────

CONFIG = {
    "checkpoint_path" : "outputs/checkpoints/best_model.pth",
    "output_dir"      : "outputs/results/predictions",
    "patch_size"      : 256,
    "stride"          : 256,      # перекрытие как при нарезке
    "threshold"       : 0.5,      # порог бинаризации (0.5 = 50% уверенности)
    "features"        : [64, 128, 256, 512],
    "overlay_alpha"   : 0.4,      # прозрачность маски на overlay (0=невидима, 1=непрозрачна)
    "overlay_color"   : [0, 100, 255],  # цвет русла на overlay (синий BGR)
}


# ─────────────────────────────────────────────
# ТРАНСФОРМАЦИЯ (без аугментаций — только нормализация)
# ─────────────────────────────────────────────

def get_transform():
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
# ЗАГРУЗКА МОДЕЛИ
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> UNet:
   
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Берём features из сохранённого конфига если есть
    features = ckpt.get("config", {}).get("features", CONFIG["features"])

    model = UNet(in_channels=3, features=features).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch   = ckpt.get("epoch", "?")
    val_iou = ckpt.get("val_iou", 0)
    print(f"  Модель загружена (эпоха {epoch}, val IoU: {val_iou:.4f})")
    return model


# ─────────────────────────────────────────────
# ПРЕДСКАЗАНИЕ НА БОЛЬШОМ СНИМКЕ
# ─────────────────────────────────────────────

def predict_full_image(
    model     : UNet,
    image     : np.ndarray,
    device    : torch.device,
    patch_size: int = 256,
    stride    : int = 128,
    threshold : float = 0.5,
) -> np.ndarray:
   
    transform = get_transform()
    H, W = image.shape[:2]

    # Накопительные массивы для усреднения
    prob_map   = np.zeros((H, W), dtype=np.float32)  # сумма вероятностей
    count_map  = np.zeros((H, W), dtype=np.float32)  # сколько раз пиксель покрыт

    model.eval()
    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]

                # Нормализуем и конвертируем в тензор
                tensor = transform(image=patch)["image"]
                tensor = tensor.unsqueeze(0).to(device)  # [1, 3, 256, 256]

                # Прогоняем через модель
                logit = model(tensor)                      # [1, 1, 256, 256]
                prob  = torch.sigmoid(logit).squeeze().cpu().numpy()  # [256, 256]

                # Добавляем в накопительные карты
                prob_map[y:y+patch_size, x:x+patch_size]  += prob
                count_map[y:y+patch_size, x:x+patch_size] += 1

  
    count_map = np.maximum(count_map, 1)
    avg_prob  = prob_map / count_map

    # Бинаризуем по порогу → 0 или 255
    binary_mask = (avg_prob > threshold).astype(np.uint8) * 255
    return binary_mask

def clean_mask(mask, min_area=5000):
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255
    return clean
# ─────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ
# ─────────────────────────────────────────────

def create_visualization(
    image       : np.ndarray,
    pred_mask   : np.ndarray,
    real_mask   : np.ndarray = None,
    title       : str = "",
    save_path   : str = None,
    overlay_alpha: float = 0.4,
    overlay_color: list = None,
):
    
    if overlay_color is None:
        overlay_color = [0, 100, 255]

    has_real = real_mask is not None
    n_cols   = 4 if has_real else 3

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 8))

    # ── Оригинал ──────────────────────────────
    axes[0].imshow(image)
    axes[0].set_title("Оригинальный снимок", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    col = 1

    # ── Реальная маска (если есть) ────────────
    if has_real:
        axes[col].imshow(real_mask, cmap="gray")
        axes[col].set_title("Реальная маска\n(из Label Studio)", fontsize=12)
        axes[col].axis("off")
        col += 1

    # ── Предсказанная маска ───────────────────
    axes[col].imshow(pred_mask, cmap="gray")
    axes[col].set_title("Предсказание модели", fontsize=12, fontweight="bold")
    axes[col].axis("off")
    col += 1

    # ── Overlay (маска поверх снимка) ─────────
    overlay = image.copy()
    river_pixels = pred_mask > 127
    overlay[river_pixels] = (
        np.array(overlay_color[::-1]) * overlay_alpha +         
        overlay[river_pixels] * (1 - overlay_alpha)            
    ).astype(np.uint8)

    axes[col].imshow(overlay)
    patch_legend = mpatches.Patch(
        color=[c/255 for c in overlay_color[::-1]],
        label="Русло реки"
    )
    axes[col].legend(handles=[patch_legend], loc="upper right", fontsize=10)
    axes[col].set_title("Наложение маски", fontsize=12, fontweight="bold")
    axes[col].axis("off")

    # ── Статистика ────────────────────────────
    total_pixels = pred_mask.size
    river_pixels_count = np.sum(pred_mask > 127)
    river_percent = river_pixels_count / total_pixels * 100

    if has_real:
        # Считаем IoU с реальной маской
        real_bin = (real_mask > 127).astype(np.uint8)
        pred_bin = (pred_mask > 127).astype(np.uint8)
        intersection = np.logical_and(real_bin, pred_bin).sum()
        union        = np.logical_or(real_bin, pred_bin).sum()
        iou = intersection / union if union > 0 else 0
        stats = f"Площадь русла: {river_percent:.1f}%  |  IoU с маской: {iou:.4f}"
    else:
        stats = f"Площадь русла: {river_percent:.1f}% от снимка"

    plt.suptitle(
        f"{title}\n{stats}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Сохранено: {save_path}")

    plt.show()
    return fig


# ─────────────────────────────────────────────
# АНАЛИЗ ИЗМЕНЕНИЙ ПО ГОДАМ
# ─────────────────────────────────────────────

def analyze_changes(predictions: dict, save_path: str = None):
    
    years  = sorted(predictions.keys())
    areas  = []

    for year in years:
        mask  = predictions[year]
        total = mask.size
        river = np.sum(mask > 127)
        areas.append(river / total * 100)
        print(f"  {year}: площадь русла {river/total*100:.2f}%")

  
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    
    axes[0].plot(years, areas, "b-o", markersize=8, linewidth=2)
    axes[0].fill_between(range(len(years)), areas, alpha=0.2)
    axes[0].set_xticks(range(len(years)))
    axes[0].set_xticklabels(years, rotation=45)
    axes[0].set_title("Изменение площади русла по годам", fontsize=13)
    axes[0].set_ylabel("Площадь русла (% от снимка)")
    axes[0].set_xlabel("Год")
    axes[0].grid(True, alpha=0.3)

    # Столбчатый график
    colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(years)))
    bars = axes[1].bar(range(len(years)), areas, color=colors)
    axes[1].set_xticks(range(len(years)))
    axes[1].set_xticklabels(years, rotation=45)
    axes[1].set_title("Площадь русла по годам", fontsize=13)
    axes[1].set_ylabel("Площадь русла (% от снимка)")
    axes[1].set_xlabel("Год")
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, area in zip(bars, areas):
        axes[1].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f"{area:.1f}%", ha="center", va="bottom", fontsize=9
        )

    plt.suptitle("Анализ изменений русла реки Кусарчай", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  График сохранён: {save_path}")

    plt.show()

    # Выводим выводы
    max_year = years[np.argmax(areas)]
    min_year = years[np.argmin(areas)]
    change   = areas[-1] - areas[0]
    print(f"\n  Максимальная площадь: {max_year} ({max(areas):.2f}%)")
    print(f"  Минимальная площадь : {min_year} ({min(areas):.2f}%)")
    print(f"  Изменение за период : {change:+.2f}% "
          f"({'увеличение' if change > 0 else 'уменьшение'})")


# ─────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────

def predict(image_path: str, mask_path: str = None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device.type.upper()}")

    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Загрузка модели ───────────────────────
    print("\nЗагрузка модели...")
    model = load_model(CONFIG["checkpoint_path"], device)

    # ── Загрузка снимка ───────────────────────
    print(f"\nЗагрузка снимка: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Снимок не найден: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"  Размер: {image.shape[1]}×{image.shape[0]} пикселей")

    # ── Загрузка реальной маски (если есть) ───
    real_mask = None
    if mask_path and Path(mask_path).exists():
        real_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, real_mask = cv2.threshold(real_mask, 127, 255, cv2.THRESH_BINARY)
        print(f"  Реальная маска загружена: {mask_path}")

   
    print("\nПредсказание маски...")
    pred_mask = predict_full_image(
        model=model,
        image=image,
        device=device,
        patch_size=CONFIG["patch_size"],
        stride=CONFIG["stride"],
        threshold=CONFIG["threshold"],
    )
    pred_mask = clean_mask(pred_mask, min_area=100000)
    print(f"  Площадь русла: {np.sum(pred_mask > 127) / pred_mask.size * 100:.2f}%")

   
    img_name  = Path(image_path).stem
    save_path = Path(CONFIG["output_dir"]) / f"{img_name}_prediction.png"

    print("\nВизуализация...")
    create_visualization(
        image=image,
        pred_mask=pred_mask,
        real_mask=real_mask,
        title=f"Река Кусарчай — {img_name}",
        save_path=str(save_path),
        overlay_alpha=CONFIG["overlay_alpha"],
        overlay_color=CONFIG["overlay_color"],
    )

    
    mask_save = Path(CONFIG["output_dir"]) / f"{img_name}_mask.png"
    cv2.imwrite(str(mask_save), pred_mask)
    print(f"  Маска сохранена: {mask_save}")

    return pred_mask


# ─────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────

if __name__ == "__main__":

  
    IMAGE_PATH = "data/patches/new_im/05.2026(v).jpg"   
    MASK_PATH  = None #"data/masks/04.2023(f).png"   
    # ──────────────────────────────────────────

    print("=" * 55)
    print("  Предсказание русла реки Кусарчай")
    print("=" * 55)

    pred_mask = predict(
        image_path=IMAGE_PATH,
        mask_path=MASK_PATH,
    )

    print("\n" + "=" * 55)
    print("  Готово!")
    print(f"  Результаты в: outputs/results/predictions/")
    print("=" * 55)