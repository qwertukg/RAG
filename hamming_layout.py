#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация попарных расстояний Хэмминга для кодов выбранной цифры.

Берёт все коды заданной цифры из ``mnist_memory.npz``, вычисляет матрицу
попарных расстояний Хэмминга и сохраняет две картинки:

* ``hamming_matrix.png`` — та же матрица как градиент значений 0..1 в
  равномерной решётке точек (#000..#fff);
* ``hamming_layout.png`` — 2D‑раскладка кодов, в которой евклидовы
  расстояния между точками приближённо соответствуют хэмминговым.

Все параметры задаются константами ниже, без поддержки CLI.
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

from map import draw_points_matrix


# Константы конфигурации
MEMORY_FILE = "mnist_memory.npz"
DIGIT = 2
LIMIT = 256  # установите None, чтобы брать все коды
OUT_MATRIX_PATH = "hamming_matrix.png"
OUT_LAYOUT_PATH = "hamming_layout.png"
POINT_SIZE = 1  # размер маркера точки при визуализации


def pairwise_hamming(codes: np.ndarray) -> np.ndarray:
    """Возвращает нормированную матрицу попарных расстояний Хэмминга."""
    bits = np.unpackbits(
        codes.view(np.uint8).reshape(len(codes), -1), axis=1
    ).astype(np.int32)
    pop = bits.sum(axis=1, keepdims=True)
    inter = bits @ bits.T  # пересечение
    dist = pop + pop.T - 2 * inter
    return dist.astype(np.float32) / bits.shape[1]


def classical_mds(dist: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Классическое MDS (метод Торгерсона) для раскладки по расстояниям."""
    n = dist.shape[0]
    # матрица удвоенного центрирования
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist ** 2) @ H
    evals, evecs = eigh(B)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    w = np.maximum(evals[:n_components], 0)
    return evecs[:, :n_components] * np.sqrt(w)


def main():
    data = np.load(MEMORY_FILE)
    labels = data["train_labels"]
    mask = labels == DIGIT
    codes = data["train_codes"][mask]
    if LIMIT is not None:
        codes = codes[: LIMIT]
    if len(codes) == 0:
        raise ValueError(f"В памяти нет кодов для цифры {DIGIT}")

    mat = pairwise_hamming(codes)

    # ---- 2D-раскладка по расстояниям ----
    coords = classical_mds(mat)
    avg_dist = mat.mean(axis=1)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(coords[:, 0], coords[:, 1], c=avg_dist, cmap="gray", s=POINT_SIZE)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_aspect('equal', 'datalim')
    fig2.tight_layout()
    fig2.savefig(OUT_LAYOUT_PATH, dpi=150)
    print(f"Сохранено в {OUT_LAYOUT_PATH}")


if __name__ == "__main__":
    main()
