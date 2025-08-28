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

# Precompute layouts for different sparsity levels of the DA codes.  The
# ``K_BITS_PER_LEVEL`` parameter controls the number of 1-bits in each cell
# code.  We will generate layouts for values from 4 to 32 inclusive with a
# step of 4.
K_VALUES = list(range(4, 33, 4))

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


def compute_layout(k_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2D layout for all digits using DA encoding.

    Parameters
    ----------
    k_bits:
        Value for ``K_BITS_PER_LEVEL`` controlling sparsity of the codes.

    Returns
    -------
    coords, labels:
        The 2D coordinates obtained with UMAP and the corresponding digit
        labels.
    """

    # Update the base module parameters for the requested sparsity level.
    base.K_BITS_PER_LEVEL = k_bits
    base.rng = np.random.default_rng(base.SEED)
    base.LEVEL_CODE = [
        np.zeros(base.BITS_PER_CELL // 64, dtype=np.uint64)
    ] + [
        base.const_weight_128(base.BITS_PER_CELL, k_bits)
        for _ in range(base.LEVELS)
    ]

    digits = list(DIGIT_COLORS.keys())
    codes, labels = base.extract_codes(digits)
    bits = base.codes_to_bits(codes)
    svd = TruncatedSVD(n_components=256, random_state=base.SEED)
    bits = svd.fit_transform(bits)
    umap_model = umap.UMAP(n_components=2, metric="cosine", random_state=base.SEED)
    coords = umap_model.fit_transform(bits)
    return coords, labels


def plot_interactive(layouts: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    """Create an interactive plot with digit overlays that can be toggled.

    Parameters
    ----------
    layouts:
        Dictionary mapping ``K_BITS_PER_LEVEL`` values to precomputed
        ``(coords, labels)`` pairs.
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    # Digit visibility shared across all layouts.
    digit_visibility = {d: True for d in DIGIT_COLORS}

    # Prepare scatter plots for every combination of K and digit.  They are
    # initially hidden and only made visible for the currently selected K.
    scatters = {k: {} for k in layouts}
    for k, (coords, labels) in layouts.items():
        for digit, color in DIGIT_COLORS.items():
            mask = labels == digit
            sc = ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                s=base.POINT_SIZE,
                label=str(digit),
                marker=",",
                antialiased=True,
                linewidths=0,
                alpha=0.5,
                visible=False,
            )
            scatters[k][digit] = sc

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "datalim")
    ax.legend(title="digit")

    k_index = 0

    def show_current_k() -> None:
        k = K_VALUES[k_index]
        ax.set_title(f"Korvin approach UMAP K_BITS_PER_LEVEL={k}")
        for digit, sc in scatters[k].items():
            sc.set_visible(digit_visibility[digit])

    def hide_k(k: int) -> None:
        for sc in scatters[k].values():
            sc.set_visible(False)

    show_current_k()

    def on_key(event):
        nonlocal k_index
        if event.key and event.key.isdigit():
            d = int(event.key)
            if d in digit_visibility:
                digit_visibility[d] = not digit_visibility[d]
                scatters[K_VALUES[k_index]][d].set_visible(digit_visibility[d])
                fig.canvas.draw_idle()
        elif event.key == "left":
            hide_k(K_VALUES[k_index])
            k_index = (k_index - 1) % len(K_VALUES)
            show_current_k()
            fig.canvas.draw_idle()
        elif event.key == "right":
            hide_k(K_VALUES[k_index])
            k_index = (k_index + 1) % len(K_VALUES)
            show_current_k()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    # Precompute layouts for all requested ``K_BITS_PER_LEVEL`` values and cache
    # them in memory so that switching between them is instant.
    cached_layouts = {k: compute_layout(k) for k in K_VALUES}
    plot_interactive(cached_layouts)
