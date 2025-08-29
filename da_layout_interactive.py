#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive 2D visualization of DA layout with color toggling.

The layout is computed once from precomputed DA codes stored in
``mnist_full_discrete_pipeline_parallel.npz`` and can be interactively
explored with per-digit toggling (keys 0-9)."""

import json
import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import umap
from main_da import mnist_range


# ---- Files produced by ``main_da.py`` ----
OUT_NPZ = "mnist_full_discrete_pipeline_parallel.npz"
OUT_META = "mnist_full_discrete_pipeline_parallel.meta.json"


# ---- Visualisation parameters ----
LIMIT = None  # None = use all available codes for each digit
POINT_SIZE = 1
ALPHA = 0.5


# ---- Load meta parameters ----
with open(OUT_META, "r", encoding="utf-8") as f:
    meta = json.load(f)

SEED = int(meta["SEED"])

CACHE_PATH = "da_layout_interactive_cache.npz"

# colors for digits
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
DIGIT_COLORS = {d: c for d, c in DIGIT_COLORS.items() if d in mnist_range}


def codes_to_bits(codes: np.ndarray) -> np.ndarray:
    """Convert codes uint64 -> flat float32 bit vectors."""
    bits = np.unpackbits(
        codes.view(np.uint8).reshape(len(codes), -1),
        axis=1,
    ).astype(np.float32)
    return bits


def compute_layout():
    """Compute or load cached 2D layout for all digits."""
    if os.path.exists(CACHE_PATH):
        data = np.load(CACHE_PATH)
        return data["coords"], data["labels"]

    digits = list(DIGIT_COLORS.keys())
    data = np.load(OUT_NPZ)
    codes_all = data["train_codes"]
    labels_all = data["train_labels"]

    selected_codes = []
    selected_labels = []
    for d in digits:
        idx = np.where(labels_all == d)[0]
        if LIMIT is not None:
            idx = idx[:LIMIT]
        if idx.size == 0:
            continue
        selected_codes.append(codes_all[idx])
        selected_labels.append(labels_all[idx])

    codes = np.concatenate(selected_codes, axis=0)
    labels = np.concatenate(selected_labels, axis=0)

    bits = codes_to_bits(codes)

    umap_model = umap.UMAP(n_components=2, metric="cosine", random_state=SEED)
    coords = umap_model.fit_transform(bits)

    np.savez(CACHE_PATH, coords=coords, labels=labels)
    return coords, labels


def plot_interactive(coords: np.ndarray, labels: np.ndarray) -> None:
    """Create an interactive 2D plot with digit overlays that can be toggled."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    scatters: dict[int, Any] = {}
    digit_coords: dict[int, np.ndarray] = {}
    for digit, color in DIGIT_COLORS.items():
        mask = labels == digit
        pts = coords[mask]
        sc = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=color,
            s=POINT_SIZE,
            label=str(digit),
            marker=",",
            antialiased=True,
            linewidths=0,
            alpha=ALPHA,
        )
        scatters[digit] = sc
        digit_coords[digit] = pts

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("2D UMAP (cosine) on raw DA bit codes")
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    def refresh_legend():
        handles = []
        labels_txt = []
        for d, sc in scatters.items():
            label = f"[{d}]" if sc.get_visible() else f"{d}"
            sc.set_label(label)
            handles.append(sc)
            labels_txt.append(label)
        ax.legend(handles, labels_txt, title="digit")
        ax.set_aspect("equal")

    def autoscale():
        """Adjust axes limits to fit currently visible digits."""
        visible = [digit_coords[d] for d, sc in scatters.items() if sc.get_visible()]
        if not visible:
            return
        pts = np.vstack(visible)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        pad_x = (x_max - x_min) * 0.05
        pad_y = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect("equal")

    refresh_legend()
    autoscale()

    def on_key(event):
        if event.key and event.key.isdigit():
            d = int(event.key)
            if d in scatters:
                sc = scatters[d]
                sc.set_visible(not sc.get_visible())
                refresh_legend()
                autoscale()
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    coords, labels = compute_layout()
    plot_interactive(coords, labels)

