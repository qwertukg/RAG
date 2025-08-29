#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMAP-раскладка ПЕРВИЧНЫХ кодов (после конкатенации 49 блоков по 128 бит).
Тянет параметры/функции из main_da.py и использует артефакт OUT_NPZ.
Зависимости: numpy, umap-learn, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import umap
import main_da as md  # берем SEED, OUT_NPZ, blocks_to_boolean
from main_da import mnist_range

# --------- ХАРДКОД ТОЛЬКО ДЛЯ ВИЗУАЛИЗАЦИИ ---------
PER_CLASS  = None          # сколько примеров на класс (None = все)
NEIGHBORS  = 2            # UMAP n_neighbors
MIN_DIST   = 0.10          # UMAP min_dist
METRIC     = "cosine"      # можно "hamming" или "jaccard" для bool
SAVE_PATH  = "umap_primary_codes.png"

DIGIT_COLORS = {
    0: '#ff0000',
    1: '#ff8d00',
    2: '#e3ff00',
    3: '#56ff00',
    4: '#00ff36',
    5: '#00ffc3',
    6: '#00adff',
    7: '#0020ff',
    8: '#6c00ff',
    9: '#f900ff',
}
# оставляем только разрешённые цифры
DIGIT_COLORS = {d: c for d, c in DIGIT_COLORS.items() if d in mnist_range}

def sample_indices_per_class(labels: np.ndarray, per_class: int | None,
                             classes=None, rng=np.random.default_rng(md.SEED)) -> np.ndarray:
    if classes is None:
        classes = sorted(DIGIT_COLORS.keys())
    idx = []
    for c in classes:
        w = np.flatnonzero(labels == c)
        if w.size == 0:
            continue
        if per_class is None or w.size <= per_class:
            idx.append(w)
        else:
            idx.append(rng.choice(w, size=per_class, replace=False))
    if not idx:
        return np.array([], dtype=int)
    return np.sort(np.concatenate(idx))

def main():
    # --- 1) загрузка первичных кодов и меток из артефакта пайплайна
    data = np.load(md.OUT_NPZ)
    train_codes  = data["train_codes"]     # (N,49,2) uint64
    train_labels = data["train_labels"]    # (N,)

    # --- 2) подвыборка для скорости UMAP
    sel = sample_indices_per_class(train_labels, PER_CLASS)
    X_codes = train_codes[sel]             # (M,49,2)
    y = train_labels[sel]
    M = X_codes.shape[0]
    D = 49 * 128

    # --- 3) разворачиваем первичные коды в плоские bool-векторы длины 6272
    X_bool = np.empty((M, D), dtype=bool)
    for i in range(M):
        X_bool[i] = md.blocks_to_boolean(X_codes[i])

    # --- 4) UMAP
    reducer = umap.UMAP(n_neighbors=NEIGHBORS, min_dist=MIN_DIST,
                        metric=METRIC, random_state=md.SEED)
    Z = reducer.fit_transform(X_bool.astype(np.uint8))

    # --- 5) отрисовка по DIGIT_COLORS
    # массив цветов для каждой точки
    point_colors = np.array([DIGIT_COLORS[int(lbl)] for lbl in y])

    plt.figure(figsize=(8, 7), dpi=150)
    plt.scatter(
        Z[:, 0], Z[:, 1],
        c=point_colors, s=1, alpha=1,
        linewidths=0, marker=",", antialiased=True
    )

    # легенда только для реально присутствующих цифр
    present_digits = sorted(int(d) for d in np.unique(y) if int(d) in DIGIT_COLORS)
    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='None',
                   markersize=6, markerfacecolor=DIGIT_COLORS[d],
                   markeredgecolor='none', label=str(d))
        for d in present_digits
    ]
    if handles:
        plt.legend(handles=handles, title="digit", frameon=True)

    plt.title(f"UMAP: первичные коды (49×128), metric={METRIC}")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.tight_layout()
    if SAVE_PATH:
        plt.savefig(SAVE_PATH, bbox_inches="tight")
        print(f"[Saved] {SAVE_PATH}")
    plt.show()

if __name__ == "__main__":
    main()