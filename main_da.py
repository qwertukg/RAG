# mnist_full_discrete_pipeline_parallel.py
# Требует: Python 3.10+, numpy, torchvision, tqdm
# Пайплайн полностью по статье (MNIST) + параллелизация там, где это безопасно.

import os, json
# снизим конкуренцию внутренних тредов BLAS/Accelerate
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import numpy as np
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import ToTensor

# ------------------ Параметры ------------------
# Сенсорный фронт
GRID = 7                   # 28x28 -> 7x7 (усредняющий пуллинг 4x4)
LEVELS = 4                 # уровни квантизации яркости: 0..LEVELS
BITS_PER_CELL = 128        # длина кода на ячейку
K_BITS_PER_LEVEL = 16      # вес (число единиц) в коде уровня (const-weight)
# k-NN (fuzzy search)
KNN_K = 7
# Раскладка в 2D
GRID2 = 64                 # размер решётки
HASH_A = 2654435761        # Knuth multiplicative hashing
HASH_B = 97531
# Детекторы
DETECT_K = 256             # число детекторов (= длина финального кода)
DET_WIN = 5                # размер окна детектора
DET_MASK_DENSITY = 0.25    # доля единиц в маске детектора
DET_THR = 0.12             # порог активации детектора (доля от числа единиц маски)
# Класс-память
TARGET_DENSITY = 0.20      # плотность битов в класс-векторе после порогования
# Прочее
TRAIN_LIMIT = None         # для ускорения можно 20000
SEED = 42
DATA_DIR = "./data"
OUT_NPZ = "mnist_full_discrete_pipeline_parallel.npz"
OUT_META = "mnist_full_discrete_pipeline_parallel.meta.json"
# Параллельные настройки
WORKERS = min(os.cpu_count() or 8, 12)
CHUNK_ENCODE = 256
CHUNK_TEST = 128
CHUNK_CLASS = 128

rng = np.random.default_rng(SEED)

# ------------------ Утилиты ------------------
def avgpool_28_to_7(x28: np.ndarray) -> np.ndarray:
    return x28.reshape(GRID, 28//GRID, GRID, 28//GRID).mean(axis=(1,3))

def quantize_levels(x7: np.ndarray) -> np.ndarray:
    q = np.floor(x7 * LEVELS).astype(np.int32)
    q[q > LEVELS] = LEVELS
    q[q < 0] = 0
    return q

def popcount_u64(arr_u64: np.ndarray) -> np.ndarray:
    v = arr_u64.view(np.uint8)
    return np.unpackbits(v, axis=-1).sum(axis=-1, dtype=np.int32)

def jaccard_bool(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    uni = np.count_nonzero(a | b)
    return 0.0 if uni == 0 else inter / uni

# ------------------ Коды уровней (§2.0.1) и конкатенация (§2.2.3) ------------------
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

LEVEL_CODE_IDX = [np.empty((0,), dtype=np.int32)] + [
    const_weight_indices(BITS_PER_CELL, K_BITS_PER_LEVEL) for _ in range(LEVELS)
]
LEVEL_CODE = [np.zeros(2, dtype=np.uint64)] + [idx_to_u64_pair(i) for i in LEVEL_CODE_IDX[1:]]

def encode_image_blocks(img28: np.ndarray) -> np.ndarray:
    x7 = avgpool_28_to_7(img28)
    q  = quantize_levels(x7)
    code = np.empty((GRID*GRID, 2), dtype=np.uint64)
    t = 0
    for r in range(GRID):
        for c in range(GRID):
            lvl = int(q[r, c])
            code[t] = LEVEL_CODE[lvl]
            t += 1
    return code

# ------------------ k-NN с Жаккаром (§2.2.4.2, §2.2.5) ------------------
def jaccard_blocks(query_blocks: np.ndarray, base_blocks: np.ndarray,
                   base_popcnt: np.ndarray | None = None) -> np.ndarray:
    qa = popcount_u64(query_blocks).sum(dtype=np.int32)
    if base_popcnt is None:
        bb = popcount_u64(base_blocks).reshape(base_blocks.shape[0], -1).sum(axis=1, dtype=np.int32)
    else:
        bb = base_popcnt
    inter_blocks = np.bitwise_and(base_blocks, query_blocks)     # (N,B,2)
    inter = popcount_u64(inter_blocks).reshape(base_blocks.shape[0], -1).sum(axis=1, dtype=np.int32)
    union = qa + bb - inter
    union = np.maximum(union, 1)
    return inter / union

def predict_label_knn(code_blocks: np.ndarray, train_codes: np.ndarray,
                      train_labels: np.ndarray, train_pop: np.ndarray) -> int:
    sims = jaccard_blocks(code_blocks, train_codes, base_popcnt=train_pop)  # (N,)
    idx = np.argpartition(sims, -KNN_K)[-KNN_K:]
    neigh = train_labels[idx]
    vals, counts = np.unique(neigh, return_counts=True)
    y = vals[np.argmax(counts)]
    if np.sum(counts == counts.max()) > 1:
        y = train_labels[idx[np.argmax(sims[idx])]]
    return int(y)

# ------------------ Раскладка в 2D (дискретная проекция кодов) ------------------
def blocks_to_boolean(code_blocks: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(code_blocks.view(np.uint8), axis=-1)  # (49, 16*8) = (49,128)
    bits = bits.reshape(code_blocks.shape[0], -1)              # (49,128)
    return bits.reshape(-1).astype(bool)                       # (49*128,)

def boolean_to_grid(hv_bool: np.ndarray, grid2: int) -> np.ndarray:
    on_idx = np.flatnonzero(hv_bool)
    grid = np.zeros((grid2, grid2), dtype=np.float32)
    if on_idx.size == 0:
        return grid
    x = ((on_idx * HASH_A) ^ HASH_B) % grid2
    y = ((on_idx * HASH_B) ^ HASH_A) % grid2
    np.add.at(grid, (y, x), 1.0)
    m = grid.max()
    if m > 0:
        grid /= m
    return grid

# ------------------ Пространство детекторов ------------------
def build_detectors(k: int, win: int, density: float) -> np.ndarray:
    masks = np.zeros((k, win, win), dtype=np.uint8)
    total = win * win
    ones = max(1, int(round(density * total)))
    for i in range(k):
        idx = rng.choice(total, size=ones, replace=False)
        masks[i].flat[idx] = 1
    return masks

def patches_matrix(grid: np.ndarray, win: int) -> np.ndarray:
    G = grid.shape[0]
    H = G - win + 1
    if H <= 0:
        return np.zeros((0, win*win), dtype=np.float32)
    s0, s1 = grid.strides
    shape = (H, H, win, win)
    strides = (s0, s1, s0, s1)
    view = np.lib.stride_tricks.as_strided(grid, shape=shape, strides=strides, writeable=False)
    return view.reshape(H*H, win*win).copy()

def grid_to_detector_code(grid: np.ndarray, masks: np.ndarray, thr: float) -> np.ndarray:
    K, win, _ = masks.shape
    mats = patches_matrix(grid, win)          # (Npos, win*win)
    if mats.size == 0:
        return np.zeros((K,), dtype=bool)
    masks_flat = masks.reshape(K, win*win).astype(np.float32).T   # (win*win, K)
    scores = mats @ masks_flat                                    # (Npos, K)
    best = scores.max(axis=0)                                     # (K,)
    denom = masks.reshape(K, -1).sum(axis=1).astype(np.float32)
    norm = best / np.maximum(denom, 1.0)
    return norm >= thr

# ------------------ Ассоциативная память классов ------------------
def build_class_memory_on_detector_codes_parallel(train_imgs: np.ndarray, train_lbls: np.ndarray,
                                                  encode_blocks_fn, grid2: int, masks: np.ndarray,
                                                  target_density: float, workers: int, chunk: int):
    """
    Параллельно считаем детекторные коды и аккумулируем счётчики по классам,
    затем делаем top-k порогование до заданной плотности.
    """
    num_classes = 10
    N = train_imgs.shape[0]

    def worker_chunk(lo: int, hi: int):
        counts_local = np.zeros((num_classes, DETECT_K), dtype=np.int32)
        for i in range(lo, hi):
            blocks = encode_blocks_fn(train_imgs[i])
            hv = blocks_to_boolean(blocks)
            grid = boolean_to_grid(hv, grid2)
            code = grid_to_detector_code(grid, masks, DET_THR)
            counts_local[int(train_lbls[i])] += code.astype(np.int32)
        return counts_local

    # разбиение на чанки
    spans = [(i, min(i+chunk, N)) for i in range(0, N, chunk)]
    counts = np.zeros((num_classes, DETECT_K), dtype=np.int32)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker_chunk, lo, hi) for (lo,hi) in spans]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Class memory (reduce)"):
            counts += f.result()

    # top-k до целевой плотности
    k_on = max(1, int(round(target_density * DETECT_K)))
    class_hv = np.zeros((num_classes, DETECT_K), dtype=bool)
    for cls in range(num_classes):
        cls_counts = counts[cls]
        if k_on >= DETECT_K:
            idx = np.arange(DETECT_K)
        else:
            idx = np.argpartition(cls_counts, -k_on)[-k_on:]
        class_hv[cls, idx] = True
    return class_hv, counts

# ------------------ Параллельные воркеры для кодирования и теста ------------------
def encode_train_item(i: int, imgs: np.ndarray) -> tuple[int, np.ndarray]:
    return i, encode_image_blocks(imgs[i])

def predict_knn_item(i: int, test_imgs: np.ndarray, train_codes: np.ndarray,
                     train_labels: np.ndarray, train_pop: np.ndarray) -> tuple[int, int]:
    code = encode_image_blocks(test_imgs[i])
    pred = predict_label_knn(code, train_codes, train_labels, train_pop)
    return i, pred

def predict_class_item(i: int, test_imgs: np.ndarray, class_hv: np.ndarray, masks: np.ndarray) -> tuple[int, int]:
    blocks = encode_image_blocks(test_imgs[i])
    hv = blocks_to_boolean(blocks)
    grid = boolean_to_grid(hv, GRID2)
    code = grid_to_detector_code(grid, masks, DET_THR)
    # Жаккар к класс-векторам
    best, arg = -1.0, 0
    for c in range(class_hv.shape[0]):
        sim = jaccard_bool(code, class_hv[c])
        if sim > best:
            best, arg = sim, c
    return i, arg

# ------------------ Основной сценарий ------------------
def main():
    # --- данные
    os.makedirs(DATA_DIR, exist_ok=True)
    train_ds = datasets.MNIST(DATA_DIR, train=True,  transform=ToTensor(), download=True)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, transform=ToTensor(), download=True)

    # снимем в numpy для эффективной сериализации между потоками
    if TRAIN_LIMIT is None:
        trn_limit = len(train_ds)
    else:
        trn_limit = min(TRAIN_LIMIT, len(train_ds))

    N = trn_limit
    B = GRID * GRID

    train_imgs = np.empty((N, 28, 28), dtype=np.float32)
    train_lbls = np.empty((N,), dtype=np.int16)
    for i in tqdm(range(N), desc="Load train to NumPy"):
        img, y = train_ds[i]
        train_imgs[i] = img.squeeze(0).numpy()
        train_lbls[i] = int(y)

    T = len(test_ds)
    test_imgs = np.empty((T, 28, 28), dtype=np.float32)
    test_lbls = np.empty((T,), dtype=np.int16)
    for i in tqdm(range(T), desc="Load test to NumPy"):
        img, y = test_ds[i]
        test_imgs[i] = img.squeeze(0).numpy()
        test_lbls[i] = int(y)

    # ===== Часть A: базовый k-NN на конкатенированных блоках (глава 2) =====
    train_codes  = np.empty((N, B, 2), dtype=np.uint64)

    # Параллельное кодирование train
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = []
        for s in range(0, N, CHUNK_ENCODE):
            e = min(s + CHUNK_ENCODE, N)
            for i in range(s, e):
                futs.append(ex.submit(encode_train_item, i, train_imgs))
        for f in tqdm(as_completed(futs), total=len(futs), desc="Encode train (parallel)"):
            i, code = f.result()
            train_codes[i] = code

    # Предрасчёт |B| для ускорения Жаккара
    train_pop = popcount_u64(train_codes).reshape(N, -1).sum(axis=1, dtype=np.int32)

    # Параллельная оценка k-NN
    preds_knn = np.empty((T,), dtype=np.int16)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = []
        for s in range(0, T, CHUNK_TEST):
            e = min(s + CHUNK_TEST, T)
            for i in range(s, e):
                futs.append(ex.submit(predict_knn_item, i, test_imgs, train_codes, train_lbls, train_pop))
        for f in tqdm(as_completed(futs), total=len(futs), desc="Eval k-NN (parallel)"):
            i, p = f.result()
            preds_knn[i] = p
    acc_knn = (preds_knn == test_lbls).mean()
    print(f"[A] Accuracy k-NN on blocks (Jaccard, parallel): {acc_knn:.4f}")

    # ===== Часть B: раскладка -> детекторы -> класс-память =====
    masks = build_detectors(DETECT_K, DET_WIN, DET_MASK_DENSITY)

    # Параллельное построение класс-памяти (редукция локальных счётчиков)
    class_hv, counts = build_class_memory_on_detector_codes_parallel(
        train_imgs=train_imgs,
        train_lbls=train_lbls,
        encode_blocks_fn=encode_image_blocks,
        grid2=GRID2,
        masks=masks,
        target_density=TARGET_DENSITY,
        workers=WORKERS,
        chunk=CHUNK_CLASS
    )

    # Параллельная оценка по класс-векторам
    preds_cls = np.empty((T,), dtype=np.int16)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = []
        for s in range(0, T, CHUNK_TEST):
            e = min(s + CHUNK_TEST, T)
            for i in range(s, e):
                futs.append(ex.submit(predict_class_item, i, test_imgs, class_hv, masks))
        for f in tqdm(as_completed(futs), total=len(futs), desc="Eval class-memory (parallel)"):
            i, p = f.result()
            preds_cls[i] = p
    acc_cls = (preds_cls == test_lbls).mean()
    print(f"[B] Accuracy class-memory (detector codes, Jaccard, parallel): {acc_cls:.4f}")

    # ===== Сохранение артефактов =====
    np.savez_compressed(
        OUT_NPZ,
        train_codes=train_codes,
        train_labels=train_lbls,
        train_pop=train_pop,
        level_code=np.array(LEVEL_CODE, dtype=np.uint64),
        masks=masks.astype(np.uint8),
        class_hv=class_hv.astype(np.uint8),
        counts=counts.astype(np.int32),
    )
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump({
            "GRID": GRID,
            "LEVELS": LEVELS,
            "BITS_PER_CELL": BITS_PER_CELL,
            "K_BITS_PER_LEVEL": K_BITS_PER_LEVEL,
            "KNN_K": KNN_K,
            "GRID2": GRID2,
            "DETECT_K": DETECT_K,
            "DET_WIN": DET_WIN,
            "DET_MASK_DENSITY": DET_MASK_DENSITY,
            "DET_THR": DET_THR,
            "TARGET_DENSITY": TARGET_DENSITY,
            "SEED": SEED,
            "parallel": {
                "workers": WORKERS,
                "chunk_encode": CHUNK_ENCODE,
                "chunk_test": CHUNK_TEST,
                "chunk_class": CHUNK_CLASS,
                "blas_threads": {
                    "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")
                }
            },
            "pipeline": [
                "28x28 -> 7x7 (avgpool + quantize)",
                "population coding (49 x 128 const-weight) + concatenation",
                "Jaccard + k-NN",
                "boolean flatten -> 2D layout (hash-grid)",
                "detector bank (conv + max-pooling) -> final binary code",
                "class-memory (top-k to target density)",
                "inference by Jaccard to class vectors"
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {OUT_NPZ}, {OUT_META}")

if __name__ == "__main__":
    main()