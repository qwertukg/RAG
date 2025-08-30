#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация кластеризации DAMP по «общим признакам»:
1 пиксель = 1 прототип, цвет = победивший класс-вектор (0..9),
т.е. какой класс даёт максимальное Жаккар-сходство с детекторным кодом этого прототипа.

Источники:
  - mnist_damp_detectors_torch.npz  (обязателен)
  - mnist_damp_detectors_torch.meta.json (опционально; если рядом — берём параметры)
  - main_damp_mnist_torch.py (желательно; используем его encode и классы)

Режимы:
  --mode predict  (по умолчанию) — цвет по предсказанному классу через детекторы + class_hv
  --mode truth              — цвет по истинной метке прототипа (если нужен бэкап)

Примеры:
  python viz_damp_clusters.py --npz mnist_damp_detectors_torch.npz --out damp_clusters.png --scale 10
  python viz_damp_clusters.py --npz mnist_damp_detectors_torch.npz --mode truth --out damp_truth.png --scale 10
"""

from __future__ import annotations
import argparse, os, json, sys
import numpy as np
import matplotlib.pyplot as plt

# --- Пытаемся импортировать вашу реализацию ---
try:
    import main_damp_mnist_torch as DML
    _HAS_DML = True
except Exception as e:
    DML = None
    _HAS_DML = False

# --- опционально: для чтения истинных меток (режим truth) ---
try:
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, help="Путь к mnist_damp_detectors_torch.npz", default="data.npz")
    ap.add_argument("--meta", type=str, default=None, help="Путь к .meta.json (если None — искать рядом с NPZ)")
    ap.add_argument("--data-dir", type=str, default="./data", help="Каталог MNIST (для режима truth и/или кодирования)")
    ap.add_argument("--download", action="store_true", help="Разрешить докачку MNIST (при необходимости)")
    ap.add_argument("--mode", type=str, choices=["predict", "truth"], default="predict",
                    help="predict: по детекторам+class_hv; truth: по истинным меткам прототипов")
    ap.add_argument("--lam-d", type=float, default=None, help="Переопределить LAM_D (если не брать из meta/DML)")
    ap.add_argument("--mu-e-detect", type=float, default=None, help="Переопределить MU_E_DETECT")
    ap.add_argument("--mu-d", type=float, default=None, help="Переопределить MU_D")
    ap.add_argument("--sigma", type=int, default=None, help="Переопределить SIGMA (макс. активных детекторов)")
    ap.add_argument("--scale", type=int, default=10, help="Апскейл PNG (1 клетка решётки → scale пикселей)")
    ap.add_argument("--out", type=str, default="damp_clusters.png", help="Путь сохранения PNG")
    ap.add_argument("--show", action="store_true", help="Показать окно после сохранения")
    return ap.parse_args()


def load_npz(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    need = ["damp_grid", "proto_idx", "detectors"]
    for k in need:
        if k not in z:
            raise KeyError(f"В NPZ нет ключа '{k}'.")
    grid = z["damp_grid"].astype(int)    # [H,W]  значения 0..P-1
    proto_idx = z["proto_idx"].astype(int)  # [P] индексы в train MNIST
    det = z["detectors"]                 # [M,7] (c_y, c_x, r, lam, n_points, energy, bit_index)
    class_hv = z.get("class_hv", None)
    if class_hv is not None:
        class_hv = class_hv.astype(bool)
    return grid, proto_idx, det, class_hv


def load_meta(meta_path: str | None, npz_path: str):
    if meta_path is None:
        base = os.path.splitext(npz_path)[0]
        cand = base + ".meta.json"
        if os.path.isfile(cand):
            meta_path = cand
    if meta_path and os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_palette_10():
    # 10 различимых цветов (как в предыдущем скрипте)
    return [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]


def save_label_map(label_grid: np.ndarray, out_path: str, scale: int, title: str = ""):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    H, W = label_grid.shape
    cmap = ListedColormap(build_palette_10())
    norm = BoundaryNorm(boundaries=np.arange(-0.5, 10.5, 1.0), ncolors=10)

    # аккуратный «пиксель-арт»: 1 клетка = 1 пиксель * scale
    fig = plt.figure(figsize=(W*scale/100.0, H*scale/100.0), dpi=100)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.imshow(label_grid, cmap=cmap, norm=norm, interpolation="nearest")

    if title:
        ax.set_title(title, fontsize=8, color="#444", pad=2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    args = parse_args()

    grid, proto_idx, det, class_hv = load_npz(args.npz)
    meta = load_meta(args.meta, args.npz)

    # Параметры: сначала из meta, затем DML, затем CLI
    def get_param(key, dflt=None):
        if key in meta.get("DETECTORS", {}):
            return meta["DETECTORS"][key]
        if _HAS_DML and hasattr(DML, key):
            return getattr(DML, key)
        return dflt

    lam_d       = args.lam_d       if args.lam_d       is not None else get_param("LAM_D", 0.70)
    mu_e_detect = args.mu_e_detect if args.mu_e_detect is not None else get_param("MU_E_DETECT", 0.02)
    mu_d        = args.mu_d        if args.mu_d        is not None else get_param("MU_D", 0.08)
    sigma       = args.sigma       if args.sigma       is not None else get_param("SIGMA", None)

    H, W = grid.shape
    P = proto_idx.shape[0]
    print(f"[INFO] Grid: {H}x{W} (P={P}) | Mode: {args.mode}")

    if args.mode == "truth":
        if not _HAS_TORCH:
            print("[ERR] torchvision не найден. Установите torchvision или используйте --mode predict.")
            sys.exit(1)
        # Читаем train MNIST, берём метки прототипов
        tr = datasets.MNIST(args.data_dir, train=True, transform=ToTensor(), download=args.download)
        labels_for_protos = np.array([int(tr[i][1]) for i in proto_idx], dtype=np.int16)
        label_grid = labels_for_protos[grid]
        save_label_map(label_grid, args.out, scale=max(1, int(args.scale)),
                       title=f"DAMP truth labels ({H}x{W})")
        print(f"[OK] Saved: {args.out}")
        if args.show:
            try:
                import PIL.Image as Image
                Image.open(args.out).show()
            except Exception:
                plt.imshow(label_grid, interpolation="nearest")
                plt.axis("off")
                plt.show()
        return

    # --- mode == predict ---
    if not _HAS_DML:
        print("[ERR] Не найден main_damp_mnist_torch.py — нужен для кодирования и детектирования.")
        print("     Либо поместите модуль рядом, либо используйте --mode truth.")
        sys.exit(1)

    # Восстанавливаем прототипы (коды) и пространство детекторов, как в test_loader
    from torch import tensor
    import torch

    DEVICE = DML.DEVICE
    # Загружаем train MNIST (для восстановления изображений прототипов)
    tr = datasets.MNIST(args.data_dir, train=True, transform=ToTensor(), download=args.download)
    imgs = torch.stack([tr[i][0] for i in proto_idx], dim=0).to(DEVICE)     # [P,1,28,28]
    proto_codes = DML.encode_batch_bool(imgs)                                 # [P,B]

    # Строим DAMP-объект и подставляем решётку
    damp = DML.DAMPLayoutTorch(codes_bool=proto_codes, H=H, W=W,
                               lam_far=DML.LAM_FAR, lam_near=DML.LAM_NEAR,
                               eta=DML.ETA, r_energy=DML.R_ENERGY, pair_radius=DML.PAIR_RADIUS)
    damp.grid_idx = grid

    # Пространство детекторов
    max_bit = int(det[:, 6].max()) if det.size else -1
    out_bits = max(get_param("DETECT_K", 512), max_bit + 1)
    space = DML.DetectorSpace(damp, out_bits=out_bits)
    space.detectors = [
        DML.Detector(c=(float(r[0]), float(r[1])), r=float(r[2]), lam=float(r[3]),
                     n_points=int(r[4]), energy=float(r[5]), bit_index=int(r[6]))
        for r in det
    ]
    space.finalize_detection_matrix(mu_e=float(mu_e_detect))

    # Если class_hv нет — попробуем честно сказать и выйти
    if class_hv is None:
        print("[ERR] В NPZ нет 'class_hv'. Для режима predict он обязателен (это память классов).")
        print("     Либо перезапустите обучение с сохранением class_hv, либо используйте --mode truth.")
        sys.exit(1)

    # Детекторный код для всех прототипов (как если бы они были входами)
    with torch.no_grad():
        codes = space.detect_batch_from_codes(proto_codes, lam_a=float(lam_d),
                                              mu_d=float(mu_d), sigma=(None if sigma in [None, 0] else int(sigma)))
        codes_np = codes.detach().cpu().numpy().astype(bool)  # [P, K]

    # Предсказанный класс для каждого прототипа (Жаккар к памяти класса)
    preds = np.zeros((P,), dtype=np.int32)
    for i in range(P):
        z = codes_np[i]
        best, arg = -1.0, 0
        for c in range(10):
            inter = np.count_nonzero(z & class_hv[c])
            uni   = np.count_nonzero(z | class_hv[c])
            sim = 0.0 if uni == 0 else (inter / uni)
            if sim > best:
                best, arg = sim, c
        preds[i] = arg

    # Формируем карту HxW, где в каждой клетке — предсказанный класс прототипа из этой клетки
    label_grid = preds[grid]  # [H,W], значения 0..9

    title = f"DAMP predicted clusters ({H}x{W})  |  λd={lam_d}, μe={mu_e_detect}, μd={mu_d}, σ={sigma}"
    save_label_map(label_grid, args.out, scale=max(1, int(args.scale)), title=title)
    print(f"[OK] Saved: {args.out}")
    if args.show:
        try:
            import PIL.Image as Image
            Image.open(args.out).show()
        except Exception:
            plt.imshow(label_grid, interpolation="nearest")
            plt.axis("off")
            plt.title(title)
            plt.show()


if __name__ == "__main__":
    main()
