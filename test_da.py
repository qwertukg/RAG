# -*- coding: utf-8 -*-
# Тест/скрипт инференса: загружает память (mnist_memory.npz),
# кодирует sample.png и возвращает {"prediction": int, "confidence": float}
import json, sys
import numpy as np
from PIL import Image

MEM_NPZ  = "mnist_memory.npz"
META_JSON = "mnist_memory.meta.json"

def load_memory(mem_path=MEM_NPZ, meta_path=META_JSON):
    mem = np.load(mem_path)
    train_codes  = mem["train_codes"]        # shape: (N, B) uint64
    train_labels = mem["train_labels"]       # shape: (N,)
    train_pop    = mem["train_pop"]          # shape: (N,)
    LEVEL_CODE   = list(mem["LEVEL_CODE"].astype(np.uint64))  # list length = LEVELS+1
    # мета — для GRID/LEVELS и порядка блоков
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    GRID   = int(meta["GRID"])
    LEVELS = int(meta["LEVELS"])
    return train_codes, train_labels, train_pop, LEVEL_CODE, GRID, LEVELS

def encode_image(img28, GRID, LEVELS, LEVEL_CODE):
    """
    КОД ИЗОБРАЖЕНИЯ: конкатенация GRID*GRID блоков по 64 бита (uint64),
    каждый блок — код уровня яркости ячейки (0 даёт 0).
    Порядок: row-major (t = r*GRID + c).
    """
    # усреднение 28x28 -> GRIDxGRID
    x = img28.reshape(GRID, 28//GRID, GRID, 28//GRID).mean(axis=(1,3))
    q = np.floor(np.clip(x, 0, 1) * LEVELS).astype(np.int32)
    q[q > LEVELS] = LEVELS
    code = np.empty(GRID*GRID, dtype=np.uint64)
    t = 0
    for r in range(GRID):
        for c in range(GRID):
            lvl = int(q[r, c])
            code[t] = LEVEL_CODE[lvl]
            t += 1
    return code

def popcount_u64(arr_u64):
    # Быстрый попкаунт по блокам uint64
    v = arr_u64.view(np.uint8)
    return np.unpackbits(v, axis=-1).sum(axis=-1, dtype=np.int32)

def jaccard_blocks(query_blocks, base_blocks, base_popcnt=None):
    # Жаккар J=|A∧B|/|A∨B| для конкатенированных битовых блоков
    qa = popcount_u64(query_blocks).sum(dtype=np.int32)
    if base_popcnt is None:
        bb = popcount_u64(base_blocks).reshape(base_blocks.shape[0], -1).sum(axis=1, dtype=np.int32)
    else:
        bb = base_popcnt
    inter = popcount_u64(np.bitwise_and(base_blocks, query_blocks)).reshape(base_blocks.shape[0], -1).sum(axis=1, dtype=np.int32)
    union = np.maximum(qa + bb - inter, 1)
    return inter / union

def infer(image_path, K=7, mem_path=MEM_NPZ, meta_path=META_JSON):
    train_codes, train_labels, train_pop, LEVEL_CODE, GRID, LEVELS = load_memory(mem_path, meta_path)
    # Готовим картинку 28x28 (grayscale [0..1])
    img = Image.open(image_path).convert("L").resize((28, 28))
    img28 = (np.asarray(img, dtype=np.float32) / 255.0)

    # Кодируем и считаем похожесть
    code = encode_image(img28, GRID, LEVELS, LEVEL_CODE)
    sims = jaccard_blocks(code, train_codes, base_popcnt=train_pop)

    # top-K ближайших
    topk_idx = np.argpartition(sims, -K)[-K:]
    # сортировка по убыванию сходства
    topk_idx = topk_idx[np.argsort(sims[topk_idx])[::-1]]
    neigh_labels = train_labels[topk_idx]

    # мажоритарное голосование
    vals, counts = np.unique(neigh_labels, return_counts=True)
    pred = int(vals[np.argmax(counts)])
    confidence = float((neigh_labels == pred).sum() / K)  # доля победившего класса в топ-K

    return {"prediction": pred, "confidence": confidence}

# ---- CLI: печать JSON ----
if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.png"
    res = infer(img_path)
    print(json.dumps(res, ensure_ascii=False))