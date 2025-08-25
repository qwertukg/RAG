# discrete_mnist.py
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

# ----- гиперпараметры дискретного представления -----
SEED = 42
N_BITS = 2048          # длина общего кода
K_POS  = 8             # вес кода позиции
K_LVL  = 4             # вес кода уровня яркости
GRID   = 7             # 28x28 -> 7x7 усреднением
LEVELS = 4             # кванты яркости: 0..4, 0 = "нет сигнала"
TOPK_PER_CLASS = 256   # плотность прототипа класса (~12.5%)

rng = np.random.default_rng(SEED)

def const_weight_code(n, k, rng):
    idx = rng.choice(n, size=k, replace=False)
    v = np.zeros(n, dtype=bool); v[idx] = True
    return v

# --- кодовые книги: позиция ячейки и уровень яркости ---
POS_CODES  = [const_weight_code(N_BITS, K_POS, rng) for _ in range(GRID*GRID)]
LVL_CODES  = [np.zeros(N_BITS, dtype=bool)]  # lvl 0 = нули
LVL_CODES += [const_weight_code(N_BITS, K_LVL, rng) for _ in range(1, LEVELS+1)]

# Предвычисляем токены "позиция⊕уровень" (операция — OR)
TOKEN_CODES = []
for pos in range(GRID*GRID):
    row = []
    for lvl in range(LEVELS+1):
        row.append(POS_CODES[pos] | LVL_CODES[lvl])
    TOKEN_CODES.append(row)

def avg_pool_28_to_7(x28):
    # x28: (28,28) float in [0,1]
    x = x28.reshape(GRID, 28//GRID, GRID, 28//GRID).mean(axis=(1,3))
    return x  # (7,7)

def quantize_levels(x7):
    # в 0..LEVELS, где 0 = пусто
    q = np.floor(x7 * LEVELS).astype(int)
    q[q > LEVELS] = LEVELS
    return q

def encode_image(img28):
    # img28: numpy (28,28) [0,1]
    x7 = avg_pool_28_to_7(img28)
    q = quantize_levels(x7)
    code = np.zeros(N_BITS, dtype=bool)
    for r in range(GRID):
        for c in range(GRID):
            lvl = q[r, c]
            if lvl > 0:
                pos = r*GRID + c
                code |= TOKEN_CODES[pos][lvl]
    return code

def cos_sim_discrete(a, b):
    inter = np.count_nonzero(a & b)
    na, nb = a.sum(), b.sum()
    if na == 0 or nb == 0:
        return 0.0
    return inter / np.sqrt(na*nb)

# ----- загрузка MNIST -----
train = datasets.MNIST(root="./data", train=True,  transform=ToTensor(), download=True)
test  = datasets.MNIST(root="./data", train=False, transform=ToTensor(), download=True)

# ----- обучение: строим прототипы классов -----
bit_counts = np.zeros((10, N_BITS), dtype=np.uint32)

print("Строю коды train…")
for img, y in tqdm(train):
    img28 = img.squeeze(0).numpy()  # (28,28)
    v = encode_image(img28)
    bit_counts[y] += v  # bool -> {0,1}

# берём TOPK_PER_CLASS самых частых битов на класс
protos = np.zeros((10, N_BITS), dtype=bool)
for c in range(10):
    idx = np.argpartition(bit_counts[c], -TOPK_PER_CLASS)[-TOPK_PER_CLASS:]
    protos[c, idx] = True

# ----- инференс и метрика -----
correct = total = 0
print("Тестирую…")
for img, y in tqdm(test):
    img28 = img.squeeze(0).numpy()
    v = encode_image(img28)
    sims = [cos_sim_discrete(v, protos[c]) for c in range(10)]
    pred = int(np.argmax(sims))
    correct += (pred == y)
    total   += 1

acc = correct / total
print(f"Accuracy: {acc:.4f}")