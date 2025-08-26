#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Матрица попарных расстояний Хэмминга для кодов выбранной цифры.


Берёт все коды заданной цифры из ``mnist_memory.npz``, вычисляет
матрицу попарных расстояний Хэмминга и визуализирует её как градиент
значений 0..1 в матрице точек (#000..#fff).

Все параметры задаются константами ниже, без поддержки CLI.
"""


import numpy as np
import matplotlib.pyplot as plt
from map import draw_points_matrix


# Константы конфигурации
MEMORY_FILE = "mnist_memory.npz"
DIGIT = 3
LIMIT = 256  # установите None, чтобы брать все коды
OUT_PATH = "hamming_matrix.png"
POINT_SIZE = 100  # размер маркера точки при визуализации


def pairwise_hamming(codes: np.ndarray) -> np.ndarray:
    """Возвращает нормированную матрицу попарных расстояний Хэмминга."""
    bits = np.unpackbits(
        codes.view(np.uint8).reshape(len(codes), -1), axis=1
    ).astype(np.int32)
    pop = bits.sum(axis=1, keepdims=True)
    inter = bits @ bits.T  # пересечение
    dist = pop + pop.T - 2 * inter
    return dist.astype(np.float32) / bits.shape[1]


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

    n = mat.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, n / 5), max(4, n / 5)))
    draw_points_matrix(ax, mat, square_marker=True, point_size=POINT_SIZE)
    fig.tight_layout()

    fig.savefig(OUT_PATH, dpi=150)
    print(f"Сохранено в {OUT_PATH}")


if __name__ == "__main__":
    main()
