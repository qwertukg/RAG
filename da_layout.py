#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация косинусных расстояний между кодами, полученными методом main_da.

- Берёт LEVEL_CODE и мета-параметры из файлов памяти, созданных main_da:
    * mnist_memory.npz  (LEVEL_CODE)
    * mnist_memory.meta.json  (GRID, LEVELS, BITS_PER_CELL, K_BITS_PER_LEVEL, SEED)
- Кодирует изображения MNIST по тем же правилам (avgpool 28->7, квантизация уровней,
  конкатенация 49 блоков по 128 бит).
- Преобразует коды в плоские бит-векторы и строит UMAP-раскладку по метрике "cosine".
- Раскрашивает точки по цифрам.

Зависимости: numpy, torch, torchvision, umap-learn, matplotlib
"""

import json
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import umap

# ---- Файлы артефактов памяти (из main_da) ----
META_JSON = "mnist_memory.meta.json"
NPZ_FILE  = "mnist_memory.npz"

# ---- Параметры визуализации ----
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
LIMIT = None          # None = все доступные изображения каждой цифры; иначе берём первые LIMIT
OUT_LAYOUT_PATH = "da_layout.png"
POINT_SIZE = 1
ALPHA = 0.5

# ---- Загрузка мета-параметров и кодовой книги уровней из файлов памяти ----
with open(META_JSON, "r", encoding="utf-8") as f:
    meta = json.load(f)

GRID = int(meta["GRID"])
LEVELS = int(meta["LEVELS"])
BITS_PER_CELL = int(meta["BITS_PER_CELL"])
K_BITS_PER_LEVEL = int(meta["K_BITS_PER_LEVEL"])
SEED = int(meta["SEED"])

# LEVEL_CODE: форма (LEVELS+1, words_per_cell), dtype=uint64
#  - уровень 0 — нулевой код
LEVEL_CODE = np.load(NPZ_FILE)["LEVEL_CODE"]
WORDS_PER_CELL = LEVEL_CODE.shape[1]  # обычно 2 слова по 64 бита (итого 128 бит)

rng = np.random.default_rng(SEED)


# ------------------ Кодирование в точности как в main_da ------------------
def avgpool_28_to_7(x28: np.ndarray) -> np.ndarray:
    """
    28x28 -> 7x7: усреднение по блокам 4x4.
    """
    return x28.reshape(GRID, 28 // GRID, GRID, 28 // GRID).mean(axis=(1, 3))


def quantize_levels(x7: np.ndarray) -> np.ndarray:
    """
    Квантует средние значения в диапазон целых {0..LEVELS}.
    """
    q = np.floor(x7 * LEVELS).astype(np.int32)
    q[q < 0] = 0
    q[q > LEVELS] = LEVELS
    return q


def encode_image(img28: np.ndarray) -> np.ndarray:
    """
    Код одной картинки:
      - 7x7 усреднение
      - квантизация в уровни 0..LEVELS
      - конкатенация 49 блоков по 128 бит (WORDS_PER_CELL x uint64 на блок)
    Возвращает массив формы (GRID*GRID, WORDS_PER_CELL), dtype=uint64.
    """
    x7 = avgpool_28_to_7(img28)
    q = quantize_levels(x7)
    code = np.empty((GRID * GRID, WORDS_PER_CELL), dtype=np.uint64)
    t = 0
    for r in range(GRID):
        for c in range(GRID):
            lvl = int(q[r, c])
            code[t] = LEVEL_CODE[lvl]
            t += 1
    return code


def extract_codes(digits) -> tuple[np.ndarray, np.ndarray]:
    """
    Выбирает из train-части MNIST изображения указанных цифр, кодирует их
    и возвращает коды и метки.
    """
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST("./data", train=True, transform=tfm, download=True)
    codes = []
    labels = []
    for digit in digits:
        idx = torch.where(ds.targets == digit)[0]
        if LIMIT is not None:
            idx = idx[:LIMIT]
        if len(idx) == 0:
            raise ValueError(f"В датасете нет изображений цифры {digit}")
        for i in idx:
            img, _ = ds[i]
            code = encode_image(img.squeeze(0).numpy())
            codes.append(code)
            labels.append(digit)
    return np.stack(codes), np.array(labels, dtype=np.int16)


def codes_to_bits(codes: np.ndarray) -> np.ndarray:
    """
    Преобразует набор кодов (N, 49, WORDS_PER_CELL) uint64 -> (N, 49*BITS_PER_CELL) float32 {0,1}.
    """
    # Представляем как байты и распаковываем биты построчно для каждого объекта
    # Каждый блок содержит WORDS_PER_CELL*64 бит; всего 49 блоков
    bits = np.unpackbits(
        codes.view(np.uint8).reshape(len(codes), -1),
        axis=1
    ).astype(np.float32)
    return bits


def main() -> None:
    digits = list(DIGIT_COLORS.keys())

    # 1) Кодируем примеры выбранных классов
    codes, labels = extract_codes(digits)

    # 2) Разворачиваем коды в плоские бит-векторы (0/1)
    bits = codes_to_bits(codes)

    # 3) Строим UMAP по косинусной метрике поверх исходных бит-векторов
    umap_model = umap.UMAP(n_components=2, metric="cosine", random_state=SEED, n_jobs=1)
    coords = umap_model.fit_transform(bits)

    # 4) Визуализация
    fig, ax = plt.subplots(figsize=(6, 6))
    for digit, color in DIGIT_COLORS.items():
        mask = labels == digit
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, s=POINT_SIZE, label=str(digit),
            marker=',', antialiased=True, linewidths=0, alpha=ALPHA
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "datalim")
    ax.legend(title="digit", markerscale=8, frameon=True)
    ax.set_title("UMAP по косинусу над исходными бит-кодами (main_da)")
    fig.tight_layout()
    fig.savefig(OUT_LAYOUT_PATH, dpi=150)
    print(f"Сохранено в {OUT_LAYOUT_PATH}")


if __name__ == "__main__":
    main()