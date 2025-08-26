#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация попарных косинусных расстояний для кодов нескольких цифр.

Берёт коды указанных цифр из ``mnist_memory.npz``, вычисляет матрицу
косинусных расстояний и строит 2D‑раскладку методом классического MDS.
Точки раскрашиваются в соответствии с заданными цветами для каждой цифры.
Все параметры задаются константами ниже, без поддержки CLI.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

# Константы конфигурации
MEMORY_FILE = "mnist_memory.npz"
DIGIT_COLORS = {0: "red", 1: "blue"}
LIMIT = 256  # установите None, чтобы брать все коды для каждой цифры
OUT_LAYOUT_PATH = "cosine_layout.png"
POINT_SIZE = 1  # размер маркера точки при визуализации


def pairwise_cosine(codes: np.ndarray) -> np.ndarray:
    """Возвращает матрицу попарных косинусных расстояний (1 - cos)."""
    bits = np.unpackbits(
        codes.view(np.uint8).reshape(len(codes), -1), axis=1
    ).astype(np.float32)
    norms = np.linalg.norm(bits, axis=1, keepdims=True)
    # предотвращаем деление на ноль для нулевых векторов
    norms[norms == 0] = 1.0
    sim = bits @ bits.T / (norms * norms.T)
    return 1.0 - sim


def classical_mds(dist: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Классическое MDS (метод Торгерсона) для раскладки по расстояниям."""
    n = dist.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist ** 2) @ H
    evals, evecs = eigh(B)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    w = np.maximum(evals[:n_components], 0)
    return evecs[:, :n_components] * np.sqrt(w)


def main():
    data = np.load(MEMORY_FILE)
    labels_all = data["train_labels"]
    codes_all = data["train_codes"]

    sel_codes = []
    sel_labels = []
    for digit in DIGIT_COLORS:
        mask = labels_all == digit
        codes = codes_all[mask]
        if LIMIT is not None:
            codes = codes[: LIMIT]
        if len(codes) == 0:
            raise ValueError(f"В памяти нет кодов для цифры {digit}")
        sel_codes.append(codes)
        sel_labels.append(np.full(len(codes), digit, dtype=labels_all.dtype))

    codes = np.concatenate(sel_codes, axis=0)
    labels = np.concatenate(sel_labels, axis=0)

    mat = pairwise_cosine(codes)

    coords = classical_mds(mat)
    fig, ax = plt.subplots(figsize=(6, 6))
    for digit, color in DIGIT_COLORS.items():
        mask = labels == digit
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=POINT_SIZE, label=str(digit))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', 'datalim')
    ax.legend(title="digit")
    fig.tight_layout()
    fig.savefig(OUT_LAYOUT_PATH, dpi=150)
    print(f"Сохранено в {OUT_LAYOUT_PATH}")


if __name__ == "__main__":
    main()
