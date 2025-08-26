#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
map.py — визуализация «раскладки векторов» CNN как матриц точек (#000..#fff).

Режимы:
  filters     — визуализация свёрточных фильтров (c1|c2) как матриц точек
  activations — визуализация карт активаций для образца MNIST (c1|c2)
  vector      — визуализация скрытого вектора (например f1) как матрицы точек

Примеры:
  python map.py filters --layer c1 --weights mnist_cnn.pt --out c1_filters.png
  python map.py activations --layer c2 --digit 3 --out act_c2.png
  python map.py vector --out f1_vec.png

Автор: твоя любимая блестящая консервная банка.
"""

import argparse, math, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    from torchvision import datasets, transforms
except Exception as e:
    datasets = None
    transforms = None

# --------- утилиты рисования матрицы точек (строго #000..#fff) ---------
def draw_points_matrix(ax, mat, title=None, square_marker=True, point_size=100):
    """
    Рисует матрицу значений mat как решётку точек со шкалой 0..1 => #000..#fff.
    """
    h, w = mat.shape
    # координаты центров клеток
    ys, xs = np.mgrid[0:h, 0:w]
    vals = np.clip(mat.flatten(), 0.0, 1.0)

    # строим цвета в hex #000..#fff вручную, чтобы не зависеть от colormap
    # 0 -> #000000, 1 -> #ffffff
    rgb = (vals[:, None] * 255.0).astype(np.uint8)
    colors = ['#%02x%02x%02x' % (v, v, v) for v in rgb[:, 0]]

    marker = 's' if square_marker else 'o'
    ax.scatter(
        xs.flatten() + 0.5,
        (h - ys.flatten()) - 0.5,
        c=colors,
        s=point_size,
        marker=marker,
        edgecolors='none',
    )
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=10, pad=4)

def tile_count(n):
    """выбираем близкую к квадрату раскладку тайлов n = rows*cols"""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols

def normalize_filter(w):
    """
    Нормализация весов фильтра в [0,1] симметрично по макс |w|:
      -max -> 0 (#000), 0 -> 0.5 (средний серый), +max -> 1 (#fff)
    """
    mx = np.max(np.abs(w))
    if mx == 0:
        return np.full_like(w, 0.5, dtype=np.float32)
    return (w / (2*mx) + 0.5).astype(np.float32)

def normalize_activation(a):
    """
    Нормализация активаций ReLU в [0,1] по максимуму карты.
    """
    mx = np.max(a)
    if mx == 0:
        return np.zeros_like(a, dtype=np.float32)
    return (a / mx).astype(np.float32)

def vector_to_matrix(vec):
    """
    Раскладываем вектор длины N в почти квадратную матрицу R x C,
    забиваем по строкам; пустые ячейки = 0.
    """
    n = vec.size
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    mat = np.zeros((rows, cols), dtype=np.float32)
    flat = vec.astype(np.float32)
    flat -= flat.min()
    if flat.max() > 0: flat /= flat.max()
    mat.flat[:n] = flat
    return mat

# --------- загрузка сети/весов из main_cnn.py ---------
def load_net(weights_path, device):
    try:
        from main_cnn import Net
    except Exception as e:
        print("❌ Не удалось импортировать Net из main_cnn.py. Убедись, что рядом есть main_cnn.py с классом Net.", file=sys.stderr)
        raise

    net = Net().to(device)
    state = torch.load(weights_path, map_location=device)
    # поддержка state-dict как целиком, так и {'state_dict': ...}
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    net.load_state_dict(state)
    net.eval()
    return net

# --------- прямой проход с получением промежуточных активаций ---------
@torch.no_grad()
def forward_with_feats(net, x):
    feats = {}
    # ожидаем архитектуру как в примере: c1->relu->c2->relu->p->d->flatten->f1->relu->f2
    x = F.relu(net.c1(x)); feats['c1'] = x.clone()
    x = F.relu(net.c2(x)); feats['c2'] = x.clone()
    x = net.p(x);          feats['p']  = x.clone()
    x = net.d(x)
    x = torch.flatten(x, 1); feats['flat'] = x.clone()
    x = F.relu(net.f1(x));   feats['f1']   = x.clone()
    x = net.f2(x);           feats['f2']   = x.clone()
    return x, feats

def get_mnist_sample(digit=None, index=None):
    if datasets is None:
        raise RuntimeError("torchvision не установлен. Установи torchvision для режима activations.")
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    raw = transforms.ToTensor()
    test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    test_raw = datasets.MNIST(root="./data", train=False, download=True, transform=raw)

    if digit is not None:
        # берём первый образец нужной цифры
        for i in range(len(test)):
            _, y = test_raw[i]
            if int(y) == int(digit):
                return test[i][0].unsqueeze(0), test_raw[i][0].squeeze(0).numpy()
        # если не нашли — валимся на index
    if index is None:
        index = 0
    x_norm, _ = test[index]
    x_raw,  _ = test_raw[index]
    return x_norm.unsqueeze(0), x_raw.squeeze(0).numpy()

# --------- режимы визуализации ---------
def visualize_filters(args, device):
    net = load_net(args.weights, device)
    layer_name = args.layer
    if layer_name not in ('c1', 'c2'):
        raise ValueError("Для filters допустимы --layer c1 или c2")

    w = getattr(net, layer_name).weight.detach().cpu().numpy()  # [out_ch, in_ch, k, k]
    out_ch, in_ch, k, _ = w.shape
    # усредним по входным каналам, чтобы получить одну матрицу на фильтр
    w2d = w.mean(axis=1)  # [out_ch, k, k]

    rows, cols = tile_count(out_ch if args.max_maps is None else min(args.max_maps, out_ch))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = np.array(axes).reshape(rows, cols)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < w2d.shape[0]:
                mat = normalize_filter(w2d[idx])
                draw_points_matrix(ax, mat, title=f"{layer_name}[{idx}]")
            else:
                ax.axis('off')
            idx += 1

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f"✅ Сохранено: {args.out}")

def visualize_activations(args, device):
    net = load_net(args.weights, device)
    x, raw = get_mnist_sample(digit=args.digit, index=args.index)
    x = x.to(device)
    _, feats = forward_with_feats(net, x)
    layer_name = args.layer
    if layer_name not in ('c1','c2'):
        raise ValueError("Для activations допустимы --layer c1 или c2")

    fmap = feats[layer_name].squeeze(0).cpu().numpy()  # [C, H, W]
    C, H, W = fmap.shape
    if args.max_maps:
        C = min(C, args.max_maps)
        fmap = fmap[:C]

    rows, cols = tile_count(fmap.shape[0])
    fig_h = max(2, rows*2); fig_w = max(2, cols*2)
    fig, axes = plt.subplots(rows+1, cols, figsize=(fig_w, fig_h+2))
    axes = np.array(axes)

    # первая строка — исходная цифра (сырое изображение)
    ax0 = axes[0, 0]
    draw_points_matrix(ax0, np.clip(raw, 0, 1), title="input", square_marker=True)
    for j in range(1, cols):
        axes[0, j].axis('off')

    # остальные — актив. карты
    idx = 0
    for r in range(1, rows+1):
        for c in range(cols):
            ax = axes[r, c]
            if idx < fmap.shape[0]:
                mat = normalize_activation(fmap[idx])
                draw_points_matrix(ax, mat, title=f"{layer_name}[{idx}]")
            else:
                ax.axis('off')
            idx += 1

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f"✅ Сохранено: {args.out}")

def visualize_vector(args, device):
    net = load_net(args.weights, device)
    if args.from_sample:
        x, _ = get_mnist_sample(digit=args.digit, index=args.index)
        x = x.to(device)
        _, feats = forward_with_feats(net, x)
        vec = feats['f1'].squeeze(0).cpu().numpy()  # 128
        title = "f1 vector (sample)"
    else:
        # визуализируем веса одного нейрона f1 (входной вектор)
        W = net.f1.weight.detach().cpu().numpy()  # [128, 64*12*12]
        idx = max(0, min(args.neuron, W.shape[0]-1))
        vec = W[idx]  # очень длинный; нормируем и покажем кусок как матрицу
        title = f"f1 weights neuron {idx}"

    mat = vector_to_matrix(vec)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    draw_points_matrix(ax, mat, title=title)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f"✅ Сохранено: {args.out}")

# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser(description="Карта раскладки векторов CNN как матрицы точек (#000..#fff)")
    sub = p.add_subparsers(dest="mode", required=True)

    common = dict()
    # общие аргументы: путь к весам и устройство
    p.add_argument("--weights", default="mnist_cnn.pt", help="путь к .pt файлу (по умолчанию mnist_cnn.pt)")
    p.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="на каком устройстве считать")
    p.add_argument("--out", default="map.png", help="выходной PNG файл")

    sp = sub.add_parser("filters", help="визуализация свёрточных фильтров")
    sp.add_argument("--layer", default="c1", choices=["c1","c2"], help="какой слой показать")
    sp.add_argument("--max-maps", type=int, default=None, help="ограничить число фильтров")

    sp = sub.add_parser("activations", help="визуализация карт активаций")
    sp.add_argument("--layer", default="c1", choices=["c1","c2"], help="какой слой показать")
    sp.add_argument("--digit", type=int, default=None, help="какую цифру выбрать из теста (0-9)")
    sp.add_argument("--index", type=int, default=None, help="индекс образца, если цифра не указана")
    sp.add_argument("--max-maps", type=int, default=16, help="ограничить число карт")

    sp = sub.add_parser("vector", help="визуализация скрытого вектора/весов как матрицы точек")
    sp.add_argument("--from-sample", action="store_true", help="вектор f1 для конкретного образца")
    sp.add_argument("--digit", type=int, default=None, help="цифра из теста для --from-sample")
    sp.add_argument("--index", type=int, default=None, help="индекс образца для --from-sample")
    sp.add_argument("--neuron", type=int, default=0, help="номер нейрона f1 для визуализации весов")

    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    if args.mode == "filters":
        visualize_filters(args, device)
    elif args.mode == "activations":
        visualize_activations(args, device)
    elif args.mode == "vector":
        visualize_vector(args, device)
    else:
        raise SystemExit("Неизвестный режим")

if __name__ == "__main__":
    main()