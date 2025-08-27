#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация расстояний между скрытыми признаками CNN с помощью UMAP.

Берёт изображения указанных цифр из набора MNIST, пропускает их через обученную
сверточную сеть (как в main_cnn.py) и строит 2D‑раскладку с помощью алгоритма
UMAP. Точки раскрашиваются по цифрам. Все параметры задаются константами ниже,
CLI не поддерживается.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import umap

try:
    from main_cnn import Net
except Exception as e:  # pragma: no cover - описание ошибки помогает пользователю
    raise SystemExit("Не удалось импортировать Net из main_cnn.py")

# Константы конфигурации
WEIGHTS_PATH = "mnist_cnn.pt"
DIGIT_COLORS = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "orange", 5: "purple", 6: "pink", 7: "brown", 8: "gray", 9: "black"}
# DIGIT_COLORS = {0: "red", 1: "blue"}
# DIGIT_COLORS = {6: "green", 9: "yellow"}
LIMIT = None  # установите None, чтобы брать все изображения каждой цифры
OUT_LAYOUT_PATH = "cnn_layout.png"
POINT_SIZE = 1  # размер маркера точки при визуализации


def extract_f1_features(net: Net, digits) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает матрицу признаков f1 и соответствующие метки."""
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ds = datasets.MNIST("./data", train=True, transform=tfm, download=True)

    feats = []
    labels = []
    for digit in digits:
        idx = torch.where(ds.targets == digit)[0]
        if LIMIT is not None:
            idx = idx[:LIMIT]
        if len(idx) == 0:
            raise ValueError(f"В датасете нет изображений цифры {digit}")
        for i in idx:
            x, _ = ds[i]
            x = x.unsqueeze(0)
            with torch.no_grad():
                z = F.relu(net.c1(x))
                z = F.relu(net.c2(z))
                z = net.p(z)
                z = net.d(z)
                z = z.view(z.size(0), -1)
                z = F.relu(net.f1(z))
            feats.append(z.squeeze(0).cpu().numpy())
            labels.append(digit)
    return np.stack(feats), np.array(labels, dtype=np.int16)


def umap_2d(X: np.ndarray) -> np.ndarray:
    """Строит двумерную раскладку признаков X с помощью UMAP."""
    reducer = umap.UMAP(n_components=2, random_state=42)
    return reducer.fit_transform(X)


def main() -> None:
    net = Net()
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    net.eval()

    digits = list(DIGIT_COLORS.keys())
    feats, labels = extract_f1_features(net, digits)
    coords = umap_2d(feats)

    fig, ax = plt.subplots(figsize=(6, 6))
    for digit, color in DIGIT_COLORS.items():
        mask = labels == digit
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=POINT_SIZE, label=str(digit),
                   marker=',', antialiased=True, linewidths=0, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "datalim")
    ax.legend(title="digit")
    ax.set_title("CNN approach UMAP")
    fig.tight_layout()
    fig.savefig(OUT_LAYOUT_PATH, dpi=150)
    print(f"Сохранено в {OUT_LAYOUT_PATH}")


if __name__ == "__main__":
    main()
