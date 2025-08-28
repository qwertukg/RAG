# mnist_discrete_exact_parallel.py
# Требует: numpy, torchvision, tqdm
# Строго по статье: 28x28 -> 7x7 avgpool; популяционное кодирование уровней;
# конкатенация 49 блоков по 128 бит; метрика Жаккара; k-NN (мажоритарное голосование).
# Параллелизация: этап тестирования выполняется в ThreadPoolExecutor.

import os
# (необязательно) ограничим внутренние потоки BLAS/Accelerate для лучшего масштабирования потоков
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ThreadPoolExecutor
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import numpy as np, json

# ------------------ Параметры (строго по статье) ------------------
GRID = 7                 # 28x28 -> 7x7 (усреднение 4x4)
LEVELS = 4               # уровни яркости: 0..LEVELS (0 = пустой код)
BITS_PER_CELL = 128      # длина кода на ячейку (2 * uint64)
K_BITS_PER_LEVEL = 16    # постоянный вес кода уровня
KNN_K = 7                # число соседей для k-NN (§2.2.5)
TRAIN_LIMIT = None       # можно ограничить, напр., 20000
SEED = 42

DATA_DIR = "./data"
OUT_NPZ  = "mnist_discrete_exact_memory.npz"
OUT_META = "mnist_discrete_exact.meta.json"

rng = np.random.default_rng(SEED)

# Глобальные контейнеры, которые наполним в main (удобно для потоков)
train_codes = None      # shape (N, 49, 2) uint64
train_labels = None     # shape (N,)
train_pop = None        # shape (N,)
test_imgs = None        # shape (T, 28, 28) float32
test_lbls = None        # shape (T,)

# ------------------ Кодовые книги (§2.0.1) ------------------
def const_weight_indices(bits: int, k: int) -> np.ndarray:
    idx = rng.choice(bits, size=k, replace=False).astype(np.int32)
    return np.sort(idx)

def idx_to_u64_pair(idx: np.ndarray) -> np.ndarray:
    w = np.zeros((2,), dtype=np.uint64)  # 2*64 = 128 бит
    for b in idx:
        if b < 64:
            w[0] |= (np.uint64(1) << np.uint64(b))
        else:
            w[1] |= (np.uint64(1) << np.uint64(b - 64))
    return w

# Уровень 0 — нулевой код; 1..LEVELS — const-weight коды на 128 бит
LEVEL_CODE_IDX = [np.empty((0,), dtype=np.int32)] + [
    const_weight_indices(BITS_PER_CELL, K_BITS_PER_LEVEL) for _ in range(LEVELS)
]
LEVEL_CODE = [np.zeros(2, dtype=np.uint64)] + [idx_to_u64_pair(i) for i in LEVEL_CODE_IDX[1:]]

# ------------------ Базовые функции ------------------
def avgpool_28_to_7(x28: np.ndarray) -> np.ndarray:
    # x28: (28,28) float32 [0..1] -> (7,7) средние по блокам 4x4
    return x28.reshape(GRID, 28//GRID, GRID, 28//GRID).mean(axis=(1,3))

def quantize_levels(x7: np.ndarray) -> np.ndarray:
    # [0..1] -> {0..LEVELS}
    q = np.floor(x7 * LEVELS).astype(np.int32)
    q[q > LEVELS] = LEVELS
    q[q < 0] = 0
    return q

def encode_image(img28: np.ndarray) -> np.ndarray:
    """
    §2.2.3 Конкатенация 49 блоков по 128 бит (два uint64) в порядке row-major.
    Внутри блока: §2.0.1 код уровня фиксированной длины и веса; §2.2.1 объединение (OR)
    тривиально (один признак — яркостной уровень).
    Возврат: np.ndarray shape (GRID*GRID, 2) dtype=uint64
    """
    x7 = avgpool_28_to_7(img28)        # (7,7)
    q  = quantize_levels(x7)           # (7,7) -> [0..LEVELS]
    code = np.empty((GRID*GRID, 2), dtype=np.uint64)
    t = 0
    for r in range(GRID):
        for c in range(GRID):
            lvl = int(q[r, c])
            code[t] = LEVEL_CODE[lvl]
            t += 1
    return code

def popcount_u64(arr_u64: np.ndarray) -> np.ndarray:
    # Быстрый popcount: смотрим как на байтовый массив и unpackbits
    v = arr_u64.view(np.uint8)
    return np.unpackbits(v, axis=-1).sum(axis=-1, dtype=np.int32)

def jaccard_blocks(query_blocks: np.ndarray, base_blocks: np.ndarray,
                   base_popcnt: np.ndarray | None = None) -> np.ndarray:
    """
    §2.2.4.2 Жаккар на конкатенированном коде: J = |A∧B| / |A∨B|
    query_blocks: (B,2) uint64
    base_blocks:  (N,B,2) uint64
    base_popcnt:  (N,) int32 (предрасчёт |B|)
    """
    qa = popcount_u64(query_blocks).sum(dtype=np.int32)
    if base_popcnt is None:
        bb = popcount_u64(base_blocks).reshape(base_blocks.shape[0], -1).sum(axis=1, dtype=np.int32)
    else:
        bb = base_popcnt
    inter_blocks = np.bitwise_and(base_blocks, query_blocks)    # (N,B,2)
    inter = popcount_u64(inter_blocks).reshape(base_blocks.shape[0], -1).sum(axis=1, dtype=np.int32)
    union = qa + bb - inter
    union = np.maximum(union, 1)
    return inter / union

def predict_label(code_blocks: np.ndarray) -> int:
    sims = jaccard_blocks(code_blocks, train_codes, base_popcnt=train_pop)  # (N,)
    # top-k
    idx = np.argpartition(sims, -KNN_K)[-KNN_K:]
    neigh = train_labels[idx]
    vals, counts = np.unique(neigh, return_counts=True)
    y = vals[np.argmax(counts)]
    # разрулим ничью: берём метку ближайшего по sim
    if np.sum(counts == counts.max()) > 1:
        y = train_labels[idx[np.argmax(sims[idx])]]
    return int(y)

# Рабочая функция для параллельного теста (потоки видят общую память)
def _predict_one(i: int) -> int:
    code = encode_image(test_imgs[i])   # (49,2) uint64
    return predict_label(code)

# ------------------ Main ------------------
if __name__ == "__main__":
    # --- данные
    os.makedirs(DATA_DIR, exist_ok=True)
    train_ds = datasets.MNIST(DATA_DIR, train=True,  transform=ToTensor(), download=True)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, transform=ToTensor(), download=True)

    if TRAIN_LIMIT is None:
        TRAIN_LIMIT = len(train_ds)

    # --- кодирование train
    N = TRAIN_LIMIT
    B = GRID * GRID
    train_codes = np.empty((N, B, 2), dtype=np.uint64)
    train_labels = np.empty((N,), dtype=np.int16)

    for i in tqdm(range(N), desc="Кодирую train"):
        img, y = train_ds[i]
        img28 = img.squeeze(0).numpy()
        train_codes[i]  = encode_image(img28)
        train_labels[i] = int(y)

    # --- предрасчёт |B| для ускорения Жаккара
    train_pop = popcount_u64(train_codes).reshape(N, -1).sum(axis=1, dtype=np.int32)

    # --- подготовим тест в NumPy (чтобы не гонять датасет внутри потоков)
    T = len(test_ds)
    test_imgs = np.empty((T, 28, 28), dtype=np.float32)
    test_lbls = np.empty((T,), dtype=np.int16)
    for i in tqdm(range(T), desc="Готовлю test"):
        img, y = test_ds[i]
        test_imgs[i] = img.squeeze(0).numpy()
        test_lbls[i] = int(y)

    # --- параллельный тест
    workers = min(os.cpu_count() or 8, 12)  # на M2 Pro обычно 8–12 потоков нормально
    with ThreadPoolExecutor(max_workers=workers) as ex:
        preds = list(tqdm(ex.map(_predict_one, range(T), chunksize=128),
                          total=T, desc="Параллельно тестирую"))
    preds = np.asarray(preds, dtype=np.int16)
    acc = (preds == test_lbls).mean()
    print(f"Accuracy (kNN + Jaccard, параллельно): {acc:.4f}")

    # --- сохранить артефакты
    np.savez_compressed(
        OUT_NPZ,
        train_codes=train_codes,
        train_labels=train_labels,
        train_pop=train_pop,
        LEVEL_CODE=np.array(LEVEL_CODE, dtype=np.uint64),
    )
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump({
            "GRID": GRID,
            "LEVELS": LEVELS,
            "BITS_PER_CELL": BITS_PER_CELL,
            "K_BITS_PER_LEVEL": K_BITS_PER_LEVEL,
            "KNN_K": KNN_K,
            "SEED": SEED,
            "block_order": "row-major (t = r*GRID + c)",
            "code_per_cell": "2xuint64 (128 bits)",
            "metric": "Jaccard on concatenated blocks",
            "search": "k-NN (majority vote) без весов",
            "parallel_eval": {
                "executor": "ThreadPoolExecutor",
                "workers": workers,
                "chunksize": 128
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {OUT_NPZ}, {OUT_META}")