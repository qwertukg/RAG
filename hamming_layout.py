#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Матрица попарных расстояний Хэмминга для кодов выбранной цифры.

Скрипт берёт все коды указанной цифры из mnist_memory.npz, вычисляет
матрицу попарных расстояний Хэмминга и визуализирует её как градиент
значений 0..1 в матрице точек (#000..#fff).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from map import draw_points_matrix


def pairwise_hamming(codes: np.ndarray) -> np.ndarray:
    """Возвращает нормированную матрицу попарных расстояний Хэмминга."""
    bits = np.unpackbits(codes.view(np.uint8).reshape(len(codes), -1), axis=1)
    pop = bits.sum(axis=1, keepdims=True)
    inter = bits @ bits.T
    dist = pop + pop.T - 2 * inter
    return dist / bits.shape[1]


def main():
    p = argparse.ArgumentParser(description="Матрица расстояний Хэмминга")
    p.add_argument("--memory", default="mnist_memory.npz", help="файл с кодами памяти")
    p.add_argument("--digit", type=int, default=3, help="какую цифру визуализировать (0-9)")
    p.add_argument("--limit", type=int, default=None, help="ограничить количество кодов (для отладки)")
    p.add_argument("--out", default="hamming_matrix.png", help="куда сохранить PNG")
    args = p.parse_args()

    data = np.load(args.memory)
    labels = data["train_labels"]
    mask = labels == args.digit
    codes = data["train_codes"][mask]
    if args.limit is not None:
        codes = codes[: args.limit]
    if len(codes) == 0:
        raise ValueError(f"В памяти нет кодов для цифры {args.digit}")

    mat = pairwise_hamming(codes)

    n = mat.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, n / 5), max(4, n / 5)))
    draw_points_matrix(ax, mat, square_marker=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Сохранено в {args.out}")


if __name__ == "__main__":
    main()
