#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive visualization of DA layout with color toggling.

The layout is computed once from precomputed DA codes stored in
``mnist_full_discrete_pipeline_parallel.npz`` and can be interactively
explored with per-digit toggling (keys 0-9).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import umap

# reuse utility functions and constants from the static layout script
import da_layout as base

CACHE_PATH = "da_layout_interactive_cache.npz"

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


def compute_layout():
    """Compute or load cached 2D layout for all digits."""
    if os.path.exists(CACHE_PATH):
        data = np.load(CACHE_PATH)
        return data["coords"], data["labels"]

    digits = list(DIGIT_COLORS.keys())
    data = np.load(base.OUT_NPZ)
    codes_all = data["train_codes"]
    labels_all = data["train_labels"]

    selected_codes = []
    selected_labels = []
    for d in digits:
        idx = np.where(labels_all == d)[0]
        if base.LIMIT is not None:
            idx = idx[:base.LIMIT]
        if idx.size == 0:
            continue
        selected_codes.append(codes_all[idx])
        selected_labels.append(labels_all[idx])

    codes = np.concatenate(selected_codes, axis=0)
    labels = np.concatenate(selected_labels, axis=0)

    bits = base.codes_to_bits(codes)

    umap_model = umap.UMAP(n_components=2, metric="cosine", random_state=base.SEED)
    coords = umap_model.fit_transform(bits)

    np.savez(CACHE_PATH, coords=coords, labels=labels)
    return coords, labels


def plot_interactive(coords: np.ndarray, labels: np.ndarray) -> None:
    """Create an interactive plot with digit overlays that can be toggled."""
    fig, ax = plt.subplots(figsize=(6, 6))
    scatters = {}
    for digit, color in DIGIT_COLORS.items():
        mask = labels == digit
        sc = ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            s=base.POINT_SIZE,
            label=str(digit),
            marker=',',
            antialiased=True,
            linewidths=0,
            alpha=0.5,
        )
        scatters[digit] = sc

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "datalim")
    ax.set_title("UMAP (cosine) on raw DA bit codes")
    ax.set_facecolor("black")

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
        visible = [sc.get_offsets() for sc in scatters.values() if sc.get_visible()]
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

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


if __name__ == "__main__":
    coords, labels = compute_layout()
    plot_interactive(coords, labels)
