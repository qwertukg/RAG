# mnist_sparse_bit_vectors.py
# Требует: numpy, torchvision, tqdm
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import numpy as np, json

# ------------------ Параметры (глава 2) ------------------
GRID = 7                # 28x28 -> 7x7 (пуллинг усреднением)
LEVELS = 4              # кванты яркости: 0..LEVELS (0 = "пусто")
BITS_PER_CELL = 128     # длина бит-вектора для одной ячейки (конкатенация по всем ячейкам)
K_BITS_PER_LEVEL = 16    # число единиц в коде уровня (constant-weight)
KNN_K = 7               # число соседей для fuzzy-поиска (§2.2.5)
TRAIN_LIMIT = None      # можно поставить 20000 для ускорения
SEED = 42

rng = np.random.default_rng(SEED)

# ------------------ Кодовые книги (population coding, §2.0.1) ------------------
# Для уровней 1..LEVELS задаём разрежённые коды длиной BITS_PER_CELL и весом K_BITS_PER_LEVEL.
# Уровень 0 — нулевой код.
def const_weight_u64(bits=64, k=6):
    assert bits <= 64, "для простоты используем 64 бита на ячейку"
    idx = rng.choice(bits, size=k, replace=False)
    val = 0
    for i in idx: val |= (1 << i)
    return np.uint64(val)

def const_weight_128(bits=128, k=6):
    assert bits % 64 == 0 and bits >= 64
    words = bits // 64
    idx = rng.choice(bits, size=k, replace=False)
    val = np.zeros(words, dtype=np.uint64)
    for i in idx:
        w = i // 64
        b = i % 64
        val[w] |= (np.uint64(1) << np.uint64(b))
    return val

LEVEL_CODE = [np.zeros(2, dtype=np.uint64)] + [const_weight_128(BITS_PER_CELL, K_BITS_PER_LEVEL) for _ in range(LEVELS)]

# ------------------ Вспомогательные функции ------------------
def avgpool_28_to_7(x28):
    # x28: (28,28) float32 [0..1] -> (7,7) средние блоков 4x4
    return x28.reshape(GRID, 28//GRID, GRID, 28//GRID).mean(axis=(1,3))

def quantize_levels(x7):
    # квантование в 0..LEVELS
    q = np.floor(x7 * LEVELS).astype(np.int32)
    q[q > LEVELS] = LEVELS
    return q

def encode_image(img28):
    """
    Код изображения: конкатенация 49 блоков по 128 бит (два uint64).
    Каждый блок = код уровня яркости ячейки (OR из §2.2.1, но у нас один уровень -> просто код уровня).
    Конкатенация блоков по §2.2.3.
    Возврат: np.ndarray shape (GRID*GRID, 2) dtype=uint64
    """
    x7 = avgpool_28_to_7(img28)              # (7,7)
    q = quantize_levels(x7)                  # (7,7) в [0..LEVELS]
    code = np.empty((GRID*GRID, 2), dtype=np.uint64)
    t = 0
    for r in range(GRID):
        for c in range(GRID):
            lvl = int(q[r, c])
            code[t] = LEVEL_CODE[lvl]        # уровень 0 -> 0, иначе sparse-блок
            t += 1
    return code

# Быстрый popcount по uint64 массиву
def popcount_u64(arr_u64):
    # arr_u64: (...,) uint64 -> (...,) int32, число установленных битов в каждом слове
    # Реализация через unpackbits по байтам
    v = arr_u64.view(np.uint8)
    return np.unpackbits(v, axis=-1).sum(axis=-1, dtype=np.int32)

def jaccard_blocks(query_blocks, base_blocks, base_popcnt=None):
    """
    Жаккар для конкатенированной битовой записи (§2.2.4.2):
        J = |A∧B| / |A∨B|.
    query_blocks: (B,2) uint64
    base_blocks:  (N,B,2) uint64
    base_popcnt:  (N,) int32 — precompute |B| если есть
    """
    # |A|, |B|
    qa = popcount_u64(query_blocks).sum(dtype=np.int32)
    if base_popcnt is None:
        bb = popcount_u64(base_blocks).reshape(base_blocks.shape[0], -1).sum(axis=1, dtype=np.int32)
    else:
        bb = base_popcnt

    # |A∧B|
    inter_blocks = np.bitwise_and(base_blocks, query_blocks)
    inter = popcount_u64(inter_blocks).reshape(base_blocks.shape[0], -1).sum(axis=1, dtype=np.int32)

    # |A∨B| = |A| + |B| - |A∧B|
    union = qa + bb - inter
    # Во избежание 0/0
    union = np.maximum(union, 1)
    return inter / union

# ------------------ Загрузка данных ------------------
train_ds = datasets.MNIST("./data", train=True,  transform=ToTensor(), download=True)
test_ds  = datasets.MNIST("./data", train=False, transform=ToTensor(), download=True)

if TRAIN_LIMIT is None: TRAIN_LIMIT = len(train_ds)

# ------------------ Кодирование памяти (memory, для fuzzy-поиска §2.2.5) ------------------
N = TRAIN_LIMIT
B = GRID * GRID
train_codes = np.empty((N, B, 2), dtype=np.uint64)
train_labels = np.empty((N,), dtype=np.int16)

for i in tqdm(range(N), desc="Кодирую train"):
    img, y = train_ds[i]
    img28 = img.squeeze(0).numpy()
    train_codes[i] = encode_image(img28)
    train_labels[i] = y

# Предрассчитываем |B| (плотность) для ускорения Жаккара
train_pop = popcount_u64(train_codes).reshape(N, -1).sum(axis=1, dtype=np.int32)

# ------------------ Оценка на тесте через k-NN (fuzzy-поиск §2.2.5) ------------------
def predict_label(code):
    sims = jaccard_blocks(code, train_codes, base_popcnt=train_pop)
    # top-k
    idx = np.argpartition(sims, -KNN_K)[-KNN_K:]
    neigh = train_labels[idx]
    # мажоритарное голосование
    # при равенстве — берём класс самого близкого
    vals, counts = np.unique(neigh, return_counts=True)
    y = vals[np.argmax(counts)]
    return int(y)

correct = total = 0
for i in tqdm(range(len(test_ds)), desc="Тестирую"):
    img, y = test_ds[i]
    code = encode_image(img.squeeze(0).numpy())
    pred = predict_label(code)
    correct += (pred == y)
    total += 1

print(f"Accuracy (kNN+Jaccard): {correct/total:.4f}")

# --- сохранить
np.savez_compressed(
    "mnist_memory.npz",
    train_codes=train_codes,
    train_labels=train_labels,
    train_pop=train_pop,
    LEVEL_CODE=np.array(LEVEL_CODE, dtype=np.uint64),
)
with open("mnist_memory.meta.json","w", encoding="utf-8") as f:
    json.dump({
        "GRID": GRID,
        "LEVELS": LEVELS,
        "BITS_PER_CELL": BITS_PER_CELL,
        "K_BITS_PER_LEVEL": K_BITS_PER_LEVEL,
        "KNN_K": KNN_K,
        "SEED": SEED,
        "block_order": "row-major (t=r*GRID+c)",
        "dtype": "2xuint64 per block, 49 blocks per image"
    }, f, ensure_ascii=False, indent=2)