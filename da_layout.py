#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UMAP layout of precomputed DA codes.

Reads stored codes and meta parameters produced by ``main_da.py``. Only
performs 2D layout of existing codes using UMAP without any additional
computation or re-encoding.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import umap


# ---- Files produced by ``main_da.py`` ----
OUT_NPZ = "mnist_full_discrete_pipeline_parallel.npz"
OUT_META = "mnist_full_discrete_pipeline_parallel.meta.json"


# ---- Visualisation parameters ----
DIGIT_COLORS = {
    # 0: '#ff0000',
    # 1: '#ff8d00',
    # 2: '#e3ff00',
    3: '#56ff00',
    # 4: '#00ff36',
    # 5: '#00ffc3',
    # 6: '#00adff',
    # 7: '#0020ff',
    # 8: '#6c00ff',
    # 9: '#f900ff',
}
LIMIT = None          # None = use all available codes for each digit
OUT_LAYOUT_PATH = "da_layout3.png"
POINT_SIZE = 1
ALPHA = 0.5


# ---- Load meta parameters ----
with open(OUT_META, "r", encoding="utf-8") as f:
    meta = json.load(f)

SEED = int(meta["SEED"])


def codes_to_bits(codes: np.ndarray) -> np.ndarray:
    """Convert codes uint64 -> flat float32 bit vectors."""
    bits = np.unpackbits(
        codes.view(np.uint8).reshape(len(codes), -1),
        axis=1,
    ).astype(np.float32)
    return bits


def main() -> None:
    digits = list(DIGIT_COLORS.keys())

    data = np.load(OUT_NPZ)
    codes_all = data["train_codes"]
    labels_all = data["train_labels"]

    selected_codes: list[np.ndarray] = []
    selected_labels: list[np.ndarray] = []
    for d in digits:
        idx = np.where(labels_all == d)[0]
        if LIMIT is not None:
            idx = idx[:LIMIT]
        if idx.size == 0:
            continue
        selected_codes.append(codes_all[idx])
        selected_labels.append(labels_all[idx])

    if not selected_codes:
        raise RuntimeError("No codes found for selected digits")

    codes = np.concatenate(selected_codes, axis=0)
    labels = np.concatenate(selected_labels, axis=0)

    bits = codes_to_bits(codes)

    umap_model = umap.UMAP(n_components=2, metric="cosine", random_state=SEED, n_jobs=1)
    coords = umap_model.fit_transform(bits)

    fig, ax = plt.subplots(figsize=(6, 6))
    for digit, color in DIGIT_COLORS.items():
        mask = labels == digit
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, s=POINT_SIZE, label=str(digit),
            marker=',', antialiased=True, linewidths=0, alpha=ALPHA,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")
    ax.set_aspect("equal", "box")
    s = 50
    ax.set_xlim(-s, s)
    ax.set_ylim(-s, s)
    ax.legend(title="digit", markerscale=8, frameon=True)
    ax.set_title("UMAP по косинусу над сохранёнными бит-кодами (main_da)")
    fig.tight_layout()
    fig.savefig(OUT_LAYOUT_PATH, dpi=150)
    print(f"Сохранено в {OUT_LAYOUT_PATH}")


if __name__ == "__main__":
    main()

