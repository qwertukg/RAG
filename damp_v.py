#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация итоговой раскладки DAMP из файла mnist_damp_detectors_torch.npz.

Режимы:
  - index:  цвет — по индексу прототипа (только NPZ).
  - label:  цвет — по метке класса (понадобится MNIST, чтобы прочитать метки у proto_idx).

Примеры:
  # 1) Окраска по индексу прототипа, сохранить PNG 8x апскейлом
  python viz_damp_layout.py --npz mnist_damp_detectors_torch.npz --mode index --out damp_layout_index.png --scale 8

  # 2) Окраска по меткам классов (0..9), скачать/прочитать MNIST из ./data
  python viz_damp_layout.py --npz mnist_damp_detectors_torch.npz --mode label --data-dir ./data --out damp_layout_label.png --scale 12 --show
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# === опционально: для режима 'label' (чтобы узнать метки по proto_idx) ===
from torchvision import datasets
from torchvision.transforms import ToTensor
_HAS_TORCHVISION = True


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str,
                    help="Путь к mnist_damp_detectors_torch.npz (из основного пайплайна).", default="data.npz")
    ap.add_argument("--mode", type=str, choices=["index", "label"], default="label",
                    help="Способ окраски: 'index' — по индексу прототипа; 'label' — по метке цифры (нужно прочитать MNIST).")
    ap.add_argument("--data-dir", type=str, default="./data",
                    help="Каталог с MNIST (используется только для --mode label).")
    ap.add_argument("--download", action="store_true",
                    help="Если указан --mode label: разрешить скачивание MNIST, если его нет локально.")
    ap.add_argument("--scale", type=int, default=8,
                    help="Целочисленный апскейл при сохранении/показе (1 пиксель клетки → scale пикселей).")
    ap.add_argument("--out", type=str, default="damp_layout.png",
                    help="Путь для сохранения PNG.")
    ap.add_argument("--show", action="store_true",
                    help="Показать окно с результатом после сохранения.")
    return ap.parse_args()


def load_npz_grid(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    if "damp_grid" not in z or "proto_idx" not in z:
        raise KeyError("В NPZ отсутствуют ключи 'damp_grid' и/или 'proto_idx'. Проверьте файл.")
    grid = z["damp_grid"].astype(int)   # [H,W], значения 0..P-1 (индексы прототипов)
    proto_idx = z["proto_idx"].astype(int)  # [P], индексы в тренировочном MNIST
    return grid, proto_idx


def build_image_index(grid: np.ndarray) -> np.ndarray:
    """
    Картинка HxW в «режиме индекса»: значения = индексы прототипов (0..P-1).
    Для отображения применим colormap в imshow.
    """
    return grid.copy()


def build_image_label(grid: np.ndarray, proto_idx: np.ndarray, data_dir: str, allow_download: bool) -> np.ndarray:
    """
    Картинка HxW в «режиме метки»: значения = {0..9} — истинные классы прототипов.
    Для этого читаем train MNIST и берём метку для каждого элемента из proto_idx.
    """
    if not _HAS_TORCHVISION:
        raise RuntimeError("Требуется torchvision для режима 'label'. Установите torchvision или используйте --mode index.")

    # Загружаем тренировочный MNIST
    tr = datasets.MNIST(data_dir, train=True, transform=ToTensor(), download=allow_download)
    # Составляем массив меток только для выбранных прототипов (ускоряет индексацию)
    labels_for_protos = np.array([int(tr[i][1]) for i in proto_idx], dtype=np.int16)  # [P]
    # Теперь каждая клетка grid содержит индекс прототипа 0..P-1 — берём соответствующую метку
    label_grid = labels_for_protos[grid]  # [H,W] значений 0..9
    return label_grid


def save_pixel_image(arr2d: np.ndarray, out_path: str, scale: int, mode: str):
    """
    Сохранить arr2d (HxW) как PNG без сглаживания (каждая клетка = 1 пиксель * scale).
    Для 'index' используем непрерывную карта-цветов; для 'label' — дискретную (10 цветов).
    """
    H, W = arr2d.shape
    fig = plt.figure(figsize=(W*scale/100.0, H*scale/100.0), dpi=100)  # даёт ровно scale апскейл
    ax = plt.axes([0, 0, 1, 1], frameon=False)  # плотное полотно без рамок
    ax.set_axis_off()

    if mode == "label":
        # 10 чётких цветов для цифр
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap([
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ])
        norm = BoundaryNorm(boundaries=np.arange(-0.5, 10.5, 1.0), ncolors=10)
        im = ax.imshow(arr2d, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        # Непрерывный градиент: нормируем на [0,1] по диапазону индексов
        vmin = float(arr2d.min()) if arr2d.size else 0.0
        vmax = float(arr2d.max()) if arr2d.size else 1.0
        im = ax.imshow(arr2d, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")

    # Без осей и полей, каждый «пиксель» — это ровно одна клетка решётки
    plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    args = parse_args()

    grid, proto_idx = load_npz_grid(args.npz)
    H, W = grid.shape
    P = proto_idx.shape[0]

    if args.mode == "label":
        try:
            arr = build_image_label(grid, proto_idx, data_dir=args.data_dir, allow_download=args.download)
        except Exception as e:
            print(f"[WARN] Не удалось построить 'label' визуализацию: {e}")
            print("       Использую режим 'index' вместо него.")
            arr = build_image_index(grid)
            args.mode = "index"
    else:
        arr = build_image_index(grid)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_pixel_image(arr, args.out, scale=max(1, int(args.scale)), mode=args.mode)

    print(f"[OK] Раскладка сохранена: {args.out}")
    print(f"    Размер решётки: {H}×{W} (P={P} прототипов) | Режим цвета: {args.mode}")
    if args.mode == "label":
        print("    Цвета соответствуют классам 0..9.")
    else:
        print("    Цвет — по индексу прототипа (градиент).")

    if args.show:
        # Быстрый предпросмотр того же изображения (без повторной отрисовки)
        try:
            import PIL.Image as Image
            im = Image.open(args.out)
            im.show()
        except Exception:
            # fallback: показ через matplotlib напрямую
            plt.figure()
            plt.imshow(arr, interpolation="nearest", cmap=("viridis" if args.mode=="index" else None))
            plt.axis("off")
            plt.title(f"DAMP layout ({H}x{W}), mode={args.mode}")
            plt.show()


if __name__ == "__main__":
    main()
