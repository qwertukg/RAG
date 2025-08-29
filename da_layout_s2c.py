#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMAP-раскладка первичных кодов MNIST → детекторы → "память классов".
Все ключевые параметры и функции берём из main_da.py.
Зависимости: numpy, umap-learn, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import umap
import main_da as md  # берём GRID2, DET_THR, SEED, пути к артефактам и функции

# ----- ХАРДКОД ТОЛЬКО ДЛЯ ВИЗУАЛИЗАЦИИ (не из пайплайна) -----
PER_CLASS  = None          # сколько примеров на класс брать для UMAP (None = все)
NEIGHBORS  = 5           # UMAP n_neighbors
MIN_DIST   = 0.10         # UMAP min_dist
SAVE_PATH  = "umap_primary_to_classmemory.png"

def sample_indices_per_class(labels: np.ndarray, per_class: int | None, classes=range(10),
                             rng=np.random.default_rng(md.SEED)) -> np.ndarray:
    idx = []
    for c in classes:
        where = np.flatnonzero(labels == c)
        if where.size == 0:
            continue
        if per_class is None or where.size <= per_class:
            idx.append(where)
        else:
            idx.append(rng.choice(where, size=per_class, replace=False))
    if not idx:
        return np.array([], dtype=int)
    return np.sort(np.concatenate(idx))

def main():
    # ---- загружаем артефакты из main_da.py
    data = np.load(md.OUT_NPZ)
    train_codes = data["train_codes"]            # (N,49,2) uint64
    train_labels = data["train_labels"]          # (N,)
    masks = data["masks"].astype(np.uint8)       # (K,win,win)
    class_hv = data["class_hv"].astype(bool)     # (10,K)

    # ---- отбор поднабора (для скорости UMAP)
    sel = sample_indices_per_class(train_labels, PER_CLASS, classes=range(10))
    X_codes = train_codes[sel]                   # (M,49,2)
    y = train_labels[sel]
    M = X_codes.shape[0]
    K = masks.shape[0]

    # ---- первичный код -> 2D сетка -> детекторный код (используем функции из main_da.py)
    det_mat = np.zeros((M, K), dtype=bool)
    for i in range(M):
        hv = md.blocks_to_boolean(X_codes[i])          # (6272,)
        grid = md.boolean_to_grid(hv, md.GRID2)        # (GRID2,GRID2), использует HASH_A/HASH_B из main_da.py
        det = md.grid_to_detector_code(grid, masks, md.DET_THR)  # (K,)
        det_mat[i] = det

    # ---- дополняем 10-ю памятью классов и делаем UMAP (Hamming)
    X_umap = np.vstack([det_mat.astype(np.uint8), class_hv.astype(np.uint8)])  # (M+10, K)
    reducer = umap.UMAP(n_neighbors=NEIGHBORS, min_dist=MIN_DIST,
                        metric="cosine", random_state=md.SEED)
    Z = reducer.fit_transform(X_umap)     # (M+10, 2)
    Z_img, Z_cls = Z[:-10], Z[-10:]

    # ---- рисуем
    plt.figure(figsize=(8, 7), dpi=120)
    sc = plt.scatter(Z_img[:, 0], Z_img[:, 1], c=y, s=8, cmap="tab10", alpha=0.65, linewidths=0)
    cb = plt.colorbar(sc, ticks=range(10)); cb.set_label("digit")

    # Векторы памяти классов — звёзды
    plt.scatter(Z_cls[:, 0], Z_cls[:, 1], marker="*", s=400,
                edgecolors="k", linewidths=1.2, facecolors="none")
    for c in range(10):
        plt.annotate(f"C{c}", (Z_cls[c, 0], Z_cls[c, 1]),
                     xytext=(6, 6), textcoords="offset points",
                     fontsize=10, weight="bold")

    plt.title("UMAP: первичные коды → детекторы → память классов (Hamming)")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.tight_layout()
    if SAVE_PATH:
        plt.savefig(SAVE_PATH, bbox_inches="tight")
        print(f"[Saved] {SAVE_PATH}")
    plt.show()

if __name__ == "__main__":
    main()