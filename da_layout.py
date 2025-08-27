#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация косинусных расстояний между кодами, полученными методом main_da.

Берёт изображения указанных цифр из набора MNIST, кодирует их по правилам
approach main_da и строит 2D-раскладку методом классического MDS по
косинусным расстояниям. Точки раскрашиваются по цифрам.
Все параметры задаются константами ниже, CLI не поддерживается.
"""

import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from numpy.linalg import eigh

# Константы конфигурации
GRID = 7
LEVELS = 4
BITS_PER_CELL = 128
K_BITS_PER_LEVEL = 16
SEED = 42
DIGIT_COLORS = {0: "red", 1: "blue"}
# DIGIT_COLORS = {6: "green", 9: "yellow"}
LIMIT = 256  # установите None, чтобы брать все изображения каждой цифры
OUT_LAYOUT_PATH = "da_layout.png"
POINT_SIZE = 1  # размер маркера точки при визуализации


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


def pairwise_cosine(codes: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(
        codes.view(np.uint8).reshape(len(codes), -1), axis=1
    ).astype(np.float32)
    norms = np.linalg.norm(bits, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sim = bits @ bits.T / (norms * norms.T)
    return 1.0 - sim


def classical_mds(dist: np.ndarray, n_components: int = 2) -> np.ndarray:
    n = dist.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist ** 2) @ H
    evals, evecs = eigh(B)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    w = np.maximum(evals[:n_components], 0)
    return evecs[:, :n_components] * np.sqrt(w)


def main() -> None:
    digits = list(DIGIT_COLORS.keys())
    codes, labels = extract_codes(digits)
    dist = pairwise_cosine(codes)
    coords = classical_mds(dist)

    fig, ax = plt.subplots(figsize=(6, 6))
    for digit, color in DIGIT_COLORS.items():
        mask = labels == digit
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=POINT_SIZE, label=str(digit))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "datalim")
    ax.legend(title="digit")
    ax.set_title("DA codes cosine MDS")
    fig.tight_layout()
    fig.savefig(OUT_LAYOUT_PATH, dpi=150)
    print(f"Сохранено в {OUT_LAYOUT_PATH}")


if __name__ == "__main__":
    main()
