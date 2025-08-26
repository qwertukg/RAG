#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация попарных косинусных расстояний для кодов выбранной цифры.

Берёт все коды заданной цифры из ``mnist_memory.npz``, вычисляет матрицу
косинусных расстояний и строит 2D‑раскладку методом классического MDS.

Все параметры задаются константами ниже, без поддержки CLI.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

# Константы конфигурации
MEMORY_FILE = "mnist_memory.npz"
DIGIT = 0
LIMIT = 256  # установите None, чтобы брать все коды
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
    labels = data["train_labels"]
    mask = labels == DIGIT
    codes = data["train_codes"][mask]
    if LIMIT is not None:
        codes = codes[: LIMIT]
    if len(codes) == 0:
        raise ValueError(f"В памяти нет кодов для цифры {DIGIT}")

    mat = pairwise_cosine(codes)

    coords = classical_mds(mat)
    avg_dist = mat.mean(axis=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=avg_dist, cmap="gray", s=POINT_SIZE)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', 'datalim')
    fig.tight_layout()
    fig.savefig(OUT_LAYOUT_PATH, dpi=150)
    print(f"Сохранено в {OUT_LAYOUT_PATH}")


if __name__ == "__main__":
    main()
