#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация попарных косинусных расстояний для признаковых векторов CNN.

Берёт признаки указанных цифр из обученного ``mnist_cnn.pt`` и строит
тепловую карту их попарных косинусных расстояний. Все параметры задаются
константами ниже, без поддержки CLI.
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Константы конфигурации
WEIGHTS_PATH = "mnist_cnn.pt"
DIGIT_COLORS = {0: "red", 1: "blue"}
LIMIT = None  # установите None, чтобы брать все коды для каждой цифры
OUT_LAYOUT_PATH = "cnn_layout"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """CNN как в main_cnn.py, возвращает логиты и признаки."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.p = nn.MaxPool2d(2)
        self.d = nn.Dropout(0.25)
        self.f1 = nn.Linear(64 * 12 * 12, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.p(x)
        x = self.d(x)
        x = x.view(x.size(0), -1)
        feats = F.relu(self.f1(x))
        logits = self.f2(feats)
        return logits, feats


def pairwise_cosine(vecs: torch.Tensor) -> torch.Tensor:
    """Матрица попарных косинусных расстояний (1 - cos) для тензора."""
    vecs = F.normalize(vecs, p=2, dim=1)
    return 1.0 - vecs @ vecs.T


def main():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(root="./data", train=True, transform=tfm, download=True)

    net = Net().to(device)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    net.eval()

    sel_codes = []
    sel_labels = []
    for digit in DIGIT_COLORS:
        mask = (ds.targets == digit).nonzero(as_tuple=True)[0]
        if LIMIT is not None:
            mask = mask[:LIMIT]
        if len(mask) == 0:
            raise ValueError(f"В памяти нет примеров для цифры {digit}")
        loader = DataLoader(Subset(ds, mask.tolist()), batch_size=256, shuffle=False)
        feats_list = []
        for x, _ in loader:
            x = x.to(device)
            _, feats = net(x)
            feats_list.append(feats.detach().cpu())
        feats_arr = torch.cat(feats_list, dim=0)
        sel_codes.append(feats_arr)
        sel_labels.append(torch.full((feats_arr.size(0),), digit, dtype=torch.int64))

    codes = torch.cat(sel_codes, dim=0)
    labels = torch.cat(sel_labels, dim=0)

    # Матрица попарных косинусных расстояний
    mat = pairwise_cosine(codes).numpy()

    # Сортируем по меткам, чтобы одинаковые цифры располагались блоками
    idx = np.argsort(labels.numpy())
    mat = mat[idx][:, idx]
    labels = labels.numpy()[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(mat, cmap="viridis", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    # Отделяем блоки разных цифр линиями
    counts = [np.sum(labels == d) for d in DIGIT_COLORS]
    borders = np.cumsum(counts)[:-1]
    for b in borders:
        ax.axhline(b - 0.5, color="white", linewidth=0.5)
        ax.axvline(b - 0.5, color="white", linewidth=0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine distance")
    ax.set_title("CNN features cosine distance heatmap")
    fig.tight_layout()
    file_name = "-".join([OUT_LAYOUT_PATH, *map(str, DIGIT_COLORS)]) + ".png"
    fig.savefig(file_name, dpi=150)
    print(f"Сохранено в {file_name}")


if __name__ == "__main__":
    main()
