
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from model import UNet
from predict import load_model, predict_full_image, clean_mask


# ─────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────

CONFIG = {
    "checkpoint_path" : "outputs/checkpoints/best_model.pth",
    "images_dir"      : "dataset/images",
    "masks_dir"       : "dataset/masks",
    "output_dir"      : "outputs/results/analysis",
    "patch_size"      : 256,
    "stride"          : 256,        # без перекрытия — быстрее
    "threshold"       : 0.6,
    "min_area"        : 150000,     # убираем артефакты
    "features"        : [64, 128, 256, 512],
}

# ─────────────────────────────────────────────
# СНИМКИ ДЛЯ АНАЛИЗА ПО РАКУРСАМ
# ─────────────────────────────────────────────
ANALYSIS_GROUPS = {
    "village": [
        "04.2003(v)", "03.2018(v)", "04.2017(v)", "04.2018(v)",
        "04.2020(v)", "04.2021(v)", "05.2021(v)", "06.2021(v)",
        "06.2022(v)", "07.2020(v)", "08.2022(v)", "08.2023(v)",
        "10.2013(v)", "10.2018(v)", "10.2019(v)", "12.2022(v)",
    ],
    "city": [
        "03.2017(c)", "04.2017(c)", "04.2018(c)", "04.2021(c)",
        "05.2021(c)", "06.2004(c)", "06.2012(c)", "06.2021(c)",
        "06.2022(c)", "07.2020(c)", "08.2022(c)", "08.2023(c)" ,
        "10.2014(c)", "10.2018(c)", "10.2019(c)", "11.2019(c)",
        "12.2022(c)",
    ],
    "field": [
        "04.2005(f)", "04.2018(f)", "04.2023(f)", "06.2012(f)",
        "07.2017(f)", "08.2021(f)", "08.2023(f)", "09.2021(f)",
        "10.2013(f)", "12.2022(1)(f)", "12.2022(f)",
    ],
}


# ─────────────────────────────────────────────
# ОБРАБОТКА ОДНОГО СНИМКА
# ─────────────────────────────────────────────

def process_image(image_name, model, device):
    
    img_path = Path(CONFIG["images_dir"]) / f"{image_name}.jpg"
    if not img_path.exists():
        print(f"  [!] Не найден: {image_name}.jpg — пропускаем")
        return None

    # Загрузка
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Предсказание
    pred_mask = predict_full_image(
        model=model, image=image, device=device,
        patch_size=CONFIG["patch_size"],
        stride=CONFIG["stride"],
        threshold=CONFIG["threshold"],
    )

    
    pred_mask = clean_mask(pred_mask, min_area=CONFIG["min_area"])

    # Статистика
    total_pixels = pred_mask.size
    river_pixels = np.sum(pred_mask > 127)
    river_percent = river_pixels / total_pixels * 100

    year = int(image_name.split(".")[1][:4])
    month = int(image_name.split(".")[0])

    return {
        "name"          : image_name,
        "year"          : year,
        "month"         : month,
        "river_pixels"  : river_pixels,
        "total_pixels"  : total_pixels,
        "river_percent" : river_percent,
        "mask"          : pred_mask,
        "image"         : image,
    }


# ─────────────────────────────────────────────
# ГРАФИК ИЗМЕНЕНИЙ
# ─────────────────────────────────────────────

def plot_analysis(results, group_name, save_dir):
   
    results = sorted(results, key=lambda x: (x["year"], x["month"]))

    years        = [r["year"] for r in results]
    areas        = [r["river_percent"] for r in results]
    labels       = [r["name"] for r in results]
    unique_years = sorted(set(years))

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Анализ изменений русла реки Кусарчай\n"
        f"Ракурс: {group_name.upper()} | {len(results)} снимков ({min(years)}–{max(years)})",
        fontsize=15, fontweight="bold"
    )

    # ── График 1: Площадь по времени ──────────
    ax1 = fig.add_subplot(2, 2, 1)
    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(results)))
    ax1.plot(range(len(results)), areas, "b-o", markersize=7, linewidth=2, zorder=3)
    ax1.fill_between(range(len(results)), areas, alpha=0.15, color="blue")
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(
        [f"{r['month']:02d}.{r['year']}" for r in results],
        rotation=45, ha="right", fontsize=8
    )
    ax1.set_title("Площадь русла по времени", fontsize=12)
    ax1.set_ylabel("Площадь русла (% от снимка)")
    ax1.grid(True, alpha=0.3)

    # Линия среднего
    mean_area = np.mean(areas)
    ax1.axhline(y=mean_area, color="r", linestyle="--",
                alpha=0.7, label=f"Среднее: {mean_area:.1f}%")
    ax1.legend(fontsize=9)

    # ── График 2: Среднее по годам ────────────
    ax2 = fig.add_subplot(2, 2, 2)
    year_means = {}
    for r in results:
        if r["year"] not in year_means:
            year_means[r["year"]] = []
        year_means[r["year"]].append(r["river_percent"])
    year_means = {y: np.mean(v) for y, v in year_means.items()}

    years_sorted = sorted(year_means.keys())
    means_sorted = [year_means[y] for y in years_sorted]

    bar_colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(years_sorted)))
    bars = ax2.bar(range(len(years_sorted)), means_sorted, color=bar_colors)
    ax2.set_xticks(range(len(years_sorted)))
    ax2.set_xticklabels(years_sorted, rotation=45)
    ax2.set_title("Средняя площадь по годам", fontsize=12)
    ax2.set_ylabel("Средняя площадь (%)")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, means_sorted):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8
        )

    # ── График 3: Лучший и худший снимок ──────
    ax3 = fig.add_subplot(2, 2, 3)
    max_idx = np.argmax(areas)
    min_idx = np.argmin(areas)

    ax3.imshow(results[max_idx]["mask"], cmap="gray")
    ax3.set_title(
        f"Макс. площадь: {results[max_idx]['name']}\n"
        f"({areas[max_idx]:.1f}%)",
        fontsize=10
    )
    ax3.axis("off")

    # ── График 4: Минимальный снимок ──────────
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(results[min_idx]["mask"], cmap="gray")
    ax4.set_title(
        f"Мин. площадь: {results[min_idx]['name']}\n"
        f"({areas[min_idx]:.1f}%)",
        fontsize=10
    )
    ax4.axis("off")

    plt.tight_layout()
    save_path = Path(save_dir) / f"analysis_{group_name}.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  График сохранён: {save_path}")


# ─────────────────────────────────────────────
# СВОДНЫЙ ГРАФИК ВСЕХ РАКУРСОВ
# ─────────────────────────────────────────────

def plot_summary(all_results, save_dir):
 
     fig, ax = plt.subplots(figsize=(14, 6))

    colors = {"village": "blue", "city": "red", "field": "green"}
    labels_ru = {"village": "Деревня", "city": "Город", "field": "Поле"}

    for group_name, results in all_results.items():
        if not results:
            continue
        results_sorted = sorted(results, key=lambda x: (x["year"], x["month"]))

        
        year_means = {}
        for r in results_sorted:
            if r["year"] not in year_means:
                year_means[r["year"]] = []
            year_means[r["year"]].append(r["river_percent"])
        year_means = {y: np.mean(v) for y, v in year_means.items()}

        years  = sorted(year_means.keys())
        means  = [year_means[y] for y in years]

        ax.plot(years, means, "-o", color=colors[group_name],
                label=labels_ru[group_name], markersize=7, linewidth=2)

    ax.set_title("Изменение площади русла реки Кусарчай по годам\n(все ракурсы)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Год", fontsize=12)
    ax.set_ylabel("Средняя площадь русла (% от снимка)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_dir) / "analysis_summary.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Сводный график сохранён: {save_path}")


# ─────────────────────────────────────────────
# НАУЧНЫЕ ВЫВОДЫ
# ─────────────────────────────────────────────

def print_conclusions(all_results):
    print("\n" + "=" * 55)
    print("  НАУЧНЫЕ ВЫВОДЫ")
    print("=" * 55)

    for group_name, results in all_results.items():
        if not results:
            continue
        results_sorted = sorted(results, key=lambda x: (x["year"], x["month"]))
        areas  = [r["river_percent"] for r in results_sorted]
        years  = [r["year"] for r in results_sorted]

        max_idx = np.argmax(areas)
        min_idx = np.argmin(areas)
        change  = areas[-1] - areas[0]

        labels_ru = {"village": "Деревня", "city": "Город", "field": "Поле"}
        print(f"\n  {labels_ru[group_name]}:")
        print(f"    Период анализа  : {min(years)}–{max(years)}")
        print(f"    Максимум        : {results_sorted[max_idx]['name']} "
              f"({areas[max_idx]:.2f}%)")
        print(f"    Минимум         : {results_sorted[min_idx]['name']} "
              f"({areas[min_idx]:.2f}%)")
        print(f"    Среднее         : {np.mean(areas):.2f}%")
        print(f"    Изменение       : {change:+.2f}% "
              f"({'расширение' if change > 0 else 'сужение'} русла)")


# ─────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────

def analyze():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device.type.upper()}")

    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

    print("\nЗагрузка модели...")
    model = load_model(CONFIG["checkpoint_path"], device)

    all_results = {}

    for group_name, image_list in ANALYSIS_GROUPS.items():
        labels_ru = {"village": "Деревня", "city": "Город", "field": "Поле"}
        print(f"\n{'='*55}")
        print(f"  Обработка: {labels_ru[group_name]} ({len(image_list)} снимков)")
        print(f"{'='*55}")

        results = []
        for image_name in image_list:
            print(f"  → {image_name}...")
            result = process_image(image_name, model, device)
            if result:
                results.append(result)
                print(f"     Площадь русла: {result['river_percent']:.2f}%")

        if results:
            all_results[group_name] = results
            plot_analysis(results, group_name, CONFIG["output_dir"])
        else:
            print(f"  Нет данных для {group_name}")

    print(f"\n{'='*55}")
    print("  Сводный график всех ракурсов...")
    print(f"{'='*55}")
    plot_summary(all_results, CONFIG["output_dir"])

    # Выводы
    print_conclusions(all_results)

    print(f"\n{'='*55}")
    print(f"  Анализ завершён!")
    print(f"  Результаты в: outputs/results/analysis/")
    print(f"{'='*55}")


if __name__ == "__main__":
    analyze()