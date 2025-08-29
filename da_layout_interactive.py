#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive visualization of DA layout with color toggling.

This script precomputes the UMAP layout for all digits (0-9) using the
same encoding as ``mnist_da_layout.py`` and allows enabling or disabling
individual digit overlays in real time by pressing the corresponding key
(0-9).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import umap

# переиспользуем функции и параметры из фиксированного скрипта
import da_layout as base
from main_da import mnist_range

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
    9: '#f900ff'
}
DIGIT_COLORS = {d: c for d, c in DIGIT_COLORS.items() if d in mnist_range}


def compute_layout():
    """Compute or load cached 2D layout for all digits using DA encoding."""
    if os.path.exists(CACHE_PATH):
        data = np.load(CACHE_PATH)
        return data["coords"], data["labels"]

    digits = list(DIGIT_COLORS.keys())
    codes, labels = base.extract_codes(digits)   # кодируем ровно как в mnist_da_layout.py
    bits = base.codes_to_bits(codes)             # плоские 0/1 без SVD

    # UMAP по косинусной метрике на сырых битах
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