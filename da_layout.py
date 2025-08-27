#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация косинусных расстояний между кодами, полученными методом main_da.

Берёт изображения указанных цифр из набора MNIST, кодирует их по правилам
approach main_da и строит 2D-раскладку методом UMAP по косинусному
расстоянию. Точки раскрашиваются по цифрам.
Все параметры считываются из mnist_memory.meta.json.
"""

import json
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from umap import UMAP

# Параметры конфигурации из мета-файла
META_JSON = "mnist_memory.meta.json"
with open(META_JSON, "r", encoding="utf-8") as f:
    meta = json.load(f)
GRID = int(meta["GRID"])
LEVELS = int(meta["LEVELS"])
BITS_PER_CELL = int(meta["BITS_PER_CELL"])
K_BITS_PER_LEVEL = int(meta["K_BITS_PER_LEVEL"])
SEED = int(meta["SEED"])

DIGIT_COLORS = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "orange", 5: "purple", 6: "pink", 7: "brown", 8: "gray", 9: "black"}
# DIGIT_COLORS = {0: "red", 1: "blue"}
# DIGIT_COLORS = {6: "green", 9: "yellow"}

LIMIT = None  # установите None, чтобы брать все изображения каждой цифры
OUT_LAYOUT_PATH = "da_layout.png"
POINT_SIZE = 1  # размер маркера точки при визуализации

# Параметры UMAP
UMAP_METRIC = "cosine"
N_NEIGHBORS = 15
MIN_DIST = 0.1
OUT_DATA_PATH = "da_layout_coords.npz"


rng = np.random.default_rng(SEED)


def const_weight_128(bits: int = 128, k: int = 6) -> np.ndarray:
    """Возвращает k-единичный код длины bits (в виде массива uint64)."""
    assert bits % 64 == 0 and bits >= 64
    words = bits // 64
    idx = rng.choice(bits, size=k, replace=False)
    val = np.zeros(words, dtype=np.uint64)
    for i in idx:
        w = i // 64
        b = i % 64
        val[w] |= np.uint64(1) << np.uint64(b)
    return val


LEVEL_CODE = [np.zeros(BITS_PER_CELL // 64, dtype=np.uint64)] + [
    const_weight_128(BITS_PER_CELL, K_BITS_PER_LEVEL) for _ in range(LEVELS)
]


def avgpool_28_to_7(x28: np.ndarray) -> np.ndarray:
    return x28.reshape(GRID, 28 // GRID, GRID, 28 // GRID).mean(axis=(1, 3))


def quantize_levels(x7: np.ndarray) -> np.ndarray:
    q = np.floor(x7 * LEVELS).astype(np.int32)
    q[q > LEVELS] = LEVELS
    return q


def encode_image(img28: np.ndarray) -> np.ndarray:
    x7 = avgpool_28_to_7(img28)
    q = quantize_levels(x7)
    code = np.empty((GRID * GRID, LEVEL_CODE[0].shape[0]), dtype=np.uint64)
    t = 0
    for r in range(GRID):
        for c in range(GRID):
            lvl = int(q[r, c])
            code[t] = LEVEL_CODE[lvl]
            t += 1
    return code


def extract_codes(digits) -> tuple[np.ndarray, np.ndarray]:
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST("./data", train=True, transform=tfm, download=True)
    codes = []
    labels = []
    for digit in digits:
        idx = torch.where(ds.targets == digit)[0]
        if LIMIT is not None:
            idx = idx[:LIMIT]
        if len(idx) == 0:
            raise ValueError(f"В датасете нет изображений цифры {digit}")
        for i in idx:
            img, _ = ds[i]
            code = encode_image(img.squeeze(0).numpy())
            codes.append(code)
            labels.append(digit)
    return np.stack(codes), np.array(labels, dtype=np.int16)


def codes_to_bits(codes: np.ndarray) -> np.ndarray:
    """Преобразует набор кодов в массив бит для последующей обработки."""
    return np.unpackbits(
        codes.view(np.uint8).reshape(len(codes), -1), axis=1
    ).astype(np.float32)


def main() -> None:
    digits = list(DIGIT_COLORS.keys())
    codes, labels = extract_codes(digits)
    bits = codes_to_bits(codes)
    umap_model = UMAP(
        densmap=True,
        output_dens=True,
        random_state=SEED,
        metric=UMAP_METRIC,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
    )
    coords, dens_orig, dens_emb = umap_model.fit_transform(bits)
    dens = (dens_orig, dens_emb)
    np.savez_compressed(OUT_DATA_PATH, coords=coords, dens_orig=dens_orig, dens_emb=dens_emb)

    fig, ax = plt.subplots(figsize=(6, 6))
    for digit, color in DIGIT_COLORS.items():
        mask = labels == digit
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=POINT_SIZE, label=str(digit),
                   marker=',', antialiased=True, linewidths=0, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "datalim")
    ax.legend(title="digit")
    ax.set_title("Korvin approach UMAP")
    fig.tight_layout()
    fig.savefig(OUT_LAYOUT_PATH, dpi=150)
    print(f"Сохранено в {OUT_LAYOUT_PATH}")
    print(f"Координаты и плотности сохранены в {OUT_DATA_PATH}")


if __name__ == "__main__":
    main()
