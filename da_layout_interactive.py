#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive visualization of DA layout with color toggling.

This script precomputes the UMAP layout for all digits (0-9) using the
same encoding as ``da_layout.py`` and allows enabling or disabling
individual digit overlays in real time by pressing the corresponding key
(0-9).
"""

import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import TruncatedSVD
import numpy as np
import da_layout as base

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


def compute_layout():
    """Compute 2D layout for all digits using DA encoding."""
    digits = list(DIGIT_COLORS.keys())
    codes, labels = base.extract_codes(digits)
    bits = base.codes_to_bits(codes)
    svd = TruncatedSVD(n_components=256, random_state=base.SEED)
    bits = svd.fit_transform(bits)
    umap_model = umap.UMAP(n_components=2, metric="cosine", random_state=base.SEED)
    coords = umap_model.fit_transform(bits)
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
    ax.legend(title="digit")
    ax.set_title("Korvin approach UMAP")

    def on_key(event):
        if event.key and event.key.isdigit():
            d = int(event.key)
            if d in scatters:
                sc = scatters[d]
                sc.set_visible(not sc.get_visible())
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


if __name__ == "__main__":
    coords, labels = compute_layout()
    plot_interactive(coords, labels)
