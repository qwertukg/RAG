#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуальный смоук-тест для DAMP (§5) + Детекторы (§6) на PyTorch/CUDA.

Функционально похож на test_damp.py, но:
  • строит наглядные графики:
      1) Тепловая карта Ê (E_norm) с окружностями детекторов (валидных) поверх;
      2) Гистограмма радиусов детекторов;
      3) Для выбранных индексов: исходное изображение, карта A^λ(x) объекта,
         и подсветка кругов детекторов, которые сработали (E(d,A) ≥ μ_d).
      4) Гистограмма bits_on по сэмплу тестовых изображений.
  • печатает отчёт по выбранным индексам (JSON-объект в строке):
      {"idx": ix, "prediction": int(pred), "confidence": float(conf), "truth": truth, "bits_on": int(bits_on)}

Скрипт ожидает рядом модуль: main_damp_mnist_torch.py  (обновлённый с multi-stimuli + σ).

Примеры:
  # Обучение + визуализация (с более «острыми» картами, см. обсуждение):
  python test_damp_viz.py --idx 0,1,2 --train-n 60000 --proto 48x48 --attempts 2304 \
    --lam-near 0.90 --r-energy 4 \
    --lam-d 0.82 --mu-e-build 0.12 --eps 2.0 --min-samples 4 \
    --mu-e-detect 0.02 --mu-d 0.05 \
    --detect-k 1024 --save damp_assets.npz --save-dir viz_out --show

  # Быстрая визуализация уже обученных артефактов:
  python test_damp_viz.py --idx 0,1,2 --load damp_assets.npz --save-dir viz_out --show
"""

from __future__ import annotations
import argparse
import os
import json
from typing import List, Tuple, Optional

import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- Импорт основной реализации ---
import main_damp_mnist_torch as DML


# ================= Утилиты =================

def parse_hw(s: str) -> Tuple[int, int]:
    s = s.strip()
    if "x" in s or "X" in s:
        a, b = s.lower().split("x", 1)
        return int(a), int(b)
    v = int(s)
    return v, v


def parse_indices(spec: str) -> List[int]:
    spec = str(spec)
    if "," not in spec:
        return [int(spec)]
    return [int(x) for x in spec.split(",") if x.strip()]


@torch.no_grad()
def encode_one_bool(img28_t: torch.Tensor) -> torch.BoolTensor:
    """[1,1,28,28] → [1,B] bool на DEVICE."""
    return DML.encode_batch_bool(img28_t.to(DML.DEVICE))


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ================= Визуализация =================

def plot_enorm_with_detectors(space: "DML.DetectorSpace",
                              mu_e_for_valid: float,
                              save_path: Optional[str] = None,
                              title: str = "Ê (E_norm) + detectors (valid)") -> None:
    """Тепловая карта Ê с окружностями «валидных» детекторов (те, что вошли в W_detect)."""
    E = space.E_norm  # [H,W]
    H, W = E.shape

    # восстановим порядок валидных детекторов (как в finalize_detection_matrix)
    det_valid = _rebuild_valid_detectors(space, mu_e_for_valid)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(E, cmap="magma", origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Ê")
    # окружности
    for d in det_valid:
        circ = Circle((d.c[1], d.c[0]), d.r, fill=False, lw=1.0, ec="cyan", alpha=0.7)
        ax.add_patch(circ)
        ax.plot([d.c[1]], [d.c[0]], marker="o", ms=2, color="white", alpha=0.9)
    ax.set_aspect("equal", "box")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_detector_radius_hist(space: "DML.DetectorSpace",
                              save_path: Optional[str] = None,
                              title: str = "Histogram of detector radii r") -> None:
    rs = [float(d.r) for d in space.detectors] if space.detectors else []
    fig, ax = plt.subplots(figsize=(7, 4))
    if rs:
        ax.hist(rs, bins=min(30, max(5, int(np.sqrt(len(rs))*2))), edgecolor="black")
        ax.set_title(f"{title} (n={len(rs)}), min/med/max={np.min(rs):.2f}/{np.median(rs):.2f}/{np.max(rs):.2f}")
    else:
        ax.text(0.5, 0.5, "No detectors", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
    ax.set_xlabel("r")
    ax.set_ylabel("count")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_sample_activations(space: "DML.DetectorSpace",
                            te: datasets.MNIST,
                            idxs: List[int],
                            lam_a: float,
                            mu_e_detect: float,
                            mu_d: float,
                            sigma: Optional[int],
                            save_dir: str,
                            title_prefix: str = "Sample") -> None:
    """Для каждого idx: показывает само изображение, карту A^λ и подсвечивает детекторы, которые сработали."""
    H, W = space.layout.H, space.layout.W

    # список «валидных» детекторов в том порядке, как в столбцах W_detect
    det_valid = _rebuild_valid_detectors(space, mu_e_detect)

    # заранее проверим размерность W_detect
    if space.W_detect is None or space.W_detect.numel() == 0:
        print("[Viz] Нет валидных детекторов — пропускаю детальные визуализации по образцам.")
        return
    Kvalid = space.W_detect.shape[1]
    if len(det_valid) != Kvalid:
        print(f"[Warn] det_valid ({len(det_valid)}) != Kvalid ({Kvalid}) — возможен дрифт; визуализация может быть неточной.")

    for j, ix in enumerate(idxs):
        img_t = te[ix][0].unsqueeze(0).to(DML.DEVICE)  # [1,1,28,28]
        Q = encode_one_bool(img_t)                     # [1,B] bool

        # A^λ (агрегированная по стимулам, тут стимул один)
        A_flat = None
        for A in space.activation_batch_from_codes(Q, lam_a=lam_a, batch=1):
            A_flat = A  # [1,HW]
            break
        if A_flat is None:
            print(f"[Viz] Не удалось получить A^λ для idx={ix}")
            continue
        A_map = A_flat[0].reshape(H, W).detach().cpu().numpy()

        # Энергии детекторов и сработавшие
        S = (A_flat @ space.W_detect) / space.den_detect.clamp_min(1e-12)  # [1,Kvalid]
        S = S[0]
        fired_mask = (S >= mu_d).detach().cpu().numpy().astype(bool)       # [Kvalid]

        # Рисуем: 1) исходное 28×28; 2) карта A^λ с кругами (красим сработавшие)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(te[ix][0].squeeze(0), cmap="gray", origin="upper")
        axes[0].set_title(f"{title_prefix} #{ix}: digit={int(te[ix][1])}")
        axes[0].axis("off")

        im = axes[1].imshow(A_map, cmap="viridis", origin="lower")
        axes[1].set_title(f"A^λ heatmap (λ={lam_a:.2f}), mu_d={mu_d:.2f}")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="A^λ")

        # Окружности детекторов
        for k, d in enumerate(det_valid):
            color = "lime" if (k < len(fired_mask) and fired_mask[k]) else "red"
            alpha = 0.9 if color == "lime" else 0.35
            lw = 1.6 if color == "lime" else 0.8
            circ = Circle((d.c[1], d.c[0]), d.r, fill=False, lw=lw, ec=color, alpha=alpha)
            axes[1].add_patch(circ)
            axes[1].plot([d.c[1]], [d.c[0]], marker="o", ms=2, color=color, alpha=alpha)

        axes[1].set_aspect("equal", "box")
        for a in axes:  # убрать тики
            a.set_xticks([]); a.set_yticks([])

        ensure_dir(save_dir)
        out_path = os.path.join(save_dir, f"sample_{ix:05d}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_bits_on_hist(space: "DML.DetectorSpace",
                      te: datasets.MNIST,
                      lam_a: float,
                      mu_e_detect: float,
                      mu_d: float,
                      sigma: Optional[int],
                      sample_n: int = 256,
                      save_path: Optional[str] = None,
                      title: str = "bits_on over sample") -> None:
    """Гистограмма числа активных битов на сэмпле тестовых изображений."""
    space.finalize_detection_matrix(mu_e=mu_e_detect)
    if space.W_detect is None or space.W_detect.shape[1] == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No valid detectors", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    T = len(te)
    n = min(sample_n, T)
    imgs_t = torch.stack([te[i][0] for i in range(n)], dim=0).to(DML.DEVICE)
    Q = DML.encode_batch_bool(imgs_t)  # [n,B]
    with torch.no_grad():
        codes = space.detect_batch_from_codes(Q, lam_a=lam_a, mu_d=mu_d, sigma=sigma).detach().cpu().numpy()
    bits_on = codes.sum(axis=1).astype(int)

    fig, ax = plt.subplots(figsize=(7, 4))
    if bits_on.size:
        ax.hist(bits_on, bins=min(40, max(5, int(np.sqrt(n)*2))), edgecolor="black")
        ax.set_title(f"{title} (n={n})  min/med/max: {bits_on.min()} / {np.median(bits_on):.0f} / {bits_on.max()}")
    else:
        ax.text(0.5, 0.5, "Empty codes", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
    ax.set_xlabel("# of active bits")
    ax.set_ylabel("count")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _rebuild_valid_detectors(space: "DML.DetectorSpace", mu_e: float) -> List["DML.Detector"]:
    """
    Восстанавливаем порядок валидных детекторов (как в finalize_detection_matrix):
    берём только те, у которых в круге (Ê>=μ_e) есть хотя бы одна точка.
    Это нужно для сопоставления столбцов W_detect с геометрией окружностей.
    """
    H, W = space.layout.H, space.layout.W
    E = space.E_norm.reshape(H, W)
    valid: List[DML.Detector] = []
    for d in space.detectors:
        y0 = int(max(0, np.floor(d.c[0]-d.r))); y1 = int(min(H, np.ceil(d.c[0]+d.r)+1))
        x0 = int(max(0, np.floor(d.c[1]-d.r))); x1 = int(min(W, np.ceil(d.c[1]+d.r)+1))
        YY, XX = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing='ij')
        circle = ((YY - d.c[0])**2 + (XX - d.c[1])**2) <= (d.r*d.r)
        subE = E[y0:y1, x0:x1]
        mask = circle & (subE >= mu_e)
        if np.any(mask):
            valid.append(d)
    return valid


# ================= Обёртки TRAIN/LOAD =================

@torch.no_grad()
def build_space(train_imgs_t: torch.Tensor, H: int, W: int, train_n: int,
                lam_far: float, lam_near: float, steps_far: int, steps_near: int,
                p_per_step: int, min_near_steps: int,
                detect_k: int, lam_d: float, mu_e_build: float, eps: float,
                min_samples: int, attempts: int,
                rng: np.random.Generator):
    """Собрать DAMP+детекторы с нуля. Возвращает (space, damp, proto_idx_np)."""
    P = H * W
    idx = rng.choice(train_n, size=P, replace=False)
    proto_codes = DML.encode_batch_bool(train_imgs_t[idx].to(DML.DEVICE))  # [P, NBITS] bool

    damp = DML.DAMPLayoutTorch(codes_bool=proto_codes, H=H, W=W,
                               lam_far=lam_far, lam_near=lam_near,
                               eta=DML.ETA, r_energy=DML.R_ENERGY, pair_radius=DML.PAIR_RADIUS)
    damp.run(steps_far=steps_far, steps_near=steps_near, p_per_step=p_per_step, min_near_steps=min_near_steps)

    space = DML.DetectorSpace(layout=damp, out_bits=detect_k)
    space.build_level(lam_d=lam_d, eps=eps, min_samples=min_samples,
                      mu_e=mu_e_build, attempts=attempts, max_detectors=detect_k)
    return space, damp, idx.astype(np.int32)


def save_assets(path: str, damp: "DML.DAMPLayoutTorch", space: "DML.DetectorSpace",
                proto_idx: np.ndarray, class_hv: np.ndarray) -> None:
    det = np.array([(d.c[0], d.c[1], d.r, d.lam, d.n_points, d.energy, d.bit_index)
                    for d in space.detectors], dtype=np.float32)
    np.savez_compressed(path,
        proto_idx=np.asarray(proto_idx, dtype=np.int32),
        damp_grid=damp.grid_idx.astype(np.int32),
        detectors=det,
        class_hv=class_hv.astype(np.uint8)
    )


def load_assets(path: str, detect_k_arg: int,
                train_ds: datasets.VisionDataset) -> tuple[DML.DetectorSpace, np.ndarray]:
    """Восстановить пространство детекторов и класс-память. Протокоды восстанавливаем по proto_idx."""
    z = np.load(path, allow_pickle=True)
    grid = z["damp_grid"].astype(int)
    proto_idx = z["proto_idx"].astype(int)
    det = z["detectors"]  # [M,7]
    class_hv = z.get("class_hv", None)

    # восстановим коды прототипов
    imgs = torch.stack([train_ds[i][0] for i in proto_idx], dim=0).to(DML.DEVICE)  # [P,1,28,28]
    proto_codes = DML.encode_batch_bool(imgs)

    # DAMP
    H, W = grid.shape
    damp = DML.DAMPLayoutTorch(codes_bool=proto_codes, H=H, W=W,
                               lam_far=DML.LAM_FAR, lam_near=DML.LAM_NEAR,
                               eta=DML.ETA, r_energy=DML.R_ENERGY, pair_radius=DML.PAIR_RADIUS)
    damp.grid_idx = grid

    # Space
    max_bit = int(det[:, 6].max()) if det.size else -1
    out_bits = max(detect_k_arg, max_bit + 1)
    space = DML.DetectorSpace(damp, out_bits=out_bits)
    space.detectors = [
        DML.Detector(c=(float(r[0]), float(r[1])), r=float(r[2]), lam=float(r[3]),
                     n_points=int(r[4]), energy=float(r[5]), bit_index=int(r[6]))
        for r in det
    ]

    if class_hv is not None:
        class_hv = class_hv.astype(bool)
    return space, class_hv


# ================= CLI =================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=str, default="0", help="Список индексов из тестового MNIST для визуализации/отчёта (например: 0,1,2).")

    # TRAIN режим
    ap.add_argument("--train-n", type=int, default=10000)
    ap.add_argument("--proto", type=str, default="32x32")
    ap.add_argument("--detect-k", type=int, default=getattr(DML, "DETECT_K", 512))

    # DAMP
    ap.add_argument("--lam-far", type=float, default=getattr(DML, "LAM_FAR", 0.65))
    ap.add_argument("--lam-near", type=float, default=getattr(DML, "LAM_NEAR", 0.80))
    ap.add_argument("--r-energy", type=float, default=getattr(DML, "R_ENERGY", 6.0),
                    help="Радиус для энергетики Ê (меньше — локальнее и острее карта).")
    ap.add_argument("--steps-far", type=int, default=8)
    ap.add_argument("--steps-near", type=int, default=8)
    ap.add_argument("--p-per-step", type=int, default=16384)
    ap.add_argument("--min-near-steps", type=int, default=2)

    # Детекторы (построение)
    ap.add_argument("--lam-d", type=float, default=getattr(DML, "LAM_D", 0.70))
    ap.add_argument("--mu-e-build", type=float, default=getattr(DML, "MU_E_BUILD", 0.02))
    ap.add_argument("--eps", type=float, default=getattr(DML, "DBSCAN_EPS", 5.0))
    ap.add_argument("--min-samples", type=int, default=getattr(DML, "DBSCAN_MIN_SAMPLES", 2))
    ap.add_argument("--attempts", type=int, default=getattr(DML, "DETECT_ATTEMPTS", 1024))

    # Детекторы (детектирование)
    ap.add_argument("--mu-e-detect", type=float, default=getattr(DML, "MU_E_DETECT", 0.02))
    ap.add_argument("--mu-d", type=float, default=getattr(DML, "MU_D", 0.08))
    ap.add_argument("--sigma", type=int, default=None, help="Насыщение σ для детектирования (макс. активных битов на объект).")

    # Память классов
    ap.add_argument("--target-density", type=float, default=getattr(DML, "TARGET_DENSITY", 0.35))

    # LOAD/SAVE
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--save", type=str, default=None)

    # Визуализация
    ap.add_argument("--save-dir", type=str, default="viz_out",
                    help="Куда сохранять картинки (папка будет создана).")
    ap.add_argument("--sample-n", type=int, default=256,
                    help="Сколько тестовых изображений взять для гистограммы bits_on.")
    ap.add_argument("--show", action="store_true", help="Показывать графики на экране (помимо сохранения).")

    # Прочее
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    data_dir = getattr(DML, "DATA_DIR", "./data")

    # проброс параметров, влияющих на форму Ê
    DML.R_ENERGY = float(args.r_energy)
    DML.LAM_NEAR = float(args.lam_near)

    print(f"[Info] Device: {DML.DEVICE}")

    # ==== ДАННЫЕ ====
    tr = datasets.MNIST(data_dir, train=True,  transform=ToTensor(), download=True)
    te = datasets.MNIST(data_dir, train=False, transform=ToTensor(), download=True)

    # ==== LOAD / TRAIN ====
    if args.load:
        space, class_hv = load_assets(args.load, detect_k_arg=args.detect_k, train_ds=tr)
        print(f"[LOAD] Detectors loaded: {len(space.detectors)} | out_bits={space.out_bits}")
        if class_hv is None:
            train_n = min(args.train_n, len(tr))
            train_imgs_t = torch.stack([tr[i][0] for i in range(train_n)], dim=0).to(DML.DEVICE)
            train_codes = DML.encode_batch_bool(train_imgs_t)
            train_lbls = np.array([int(tr[i][1]) for i in range(train_n)], dtype=np.int16)
            class_hv = DML.build_class_memory(space, codes_bool=train_codes, labels=train_lbls,
                                              lam_a=args.lam_d, mu_e_detect=args.mu_e_detect, mu_d=args.mu_d,
                                              detect_k=space.out_bits, target_density=args.target_density,
                                              sigma=args.sigma)
    else:
        train_n = min(args.train_n, len(tr))
        train_imgs_t = torch.stack([tr[i][0] for i in range(train_n)], dim=0).to(DML.DEVICE)
        train_lbls = np.array([int(tr[i][1]) for i in range(train_n)], dtype=np.int16)

        H, W = parse_hw(args.proto)
        space, damp, proto_idx = build_space(
            train_imgs_t=train_imgs_t, H=H, W=W, train_n=train_n,
            lam_far=args.lam_far, lam_near=args.lam_near,
            steps_far=args.steps_far, steps_near=args.steps_near,
            p_per_step=args.p_per_step, min_near_steps=args.min_near_steps,
            detect_k=args.detect_k, lam_d=args.lam_d, mu_e_build=args.mu_e_build, eps=args.eps,
            min_samples=args.min_samples, attempts=args.attempts,
            rng=rng)
        print(f"[TRAIN] Detectors built: {len(space.detectors)}")

        train_codes = DML.encode_batch_bool(train_imgs_t)
        class_hv = DML.build_class_memory(space, codes_bool=train_codes, labels=train_lbls,
                                          lam_a=args.lam_d, mu_e_detect=args.mu_e_detect, mu_d=args.mu_d,
                                          detect_k=args.detect_k, target_density=args.target_density,
                                          sigma=args.sigma)
        if args.save:
            save_assets(args.save, damp=damp, space=space, proto_idx=proto_idx, class_hv=class_hv)
            print(f"[SAVE] Артефакты сохранены в {args.save}")

    # ==== ВИЗУАЛИЗАЦИИ ГЛОБАЛЬНЫЕ ====
    ensure_dir(args.save_dir)
    space.finalize_detection_matrix(mu_e=args.mu_e_detect)

    # 1) Ê карта + окружности валидных детекторов
    plot_enorm_with_detectors(space, mu_e_for_valid=args.mu_e_detect,
                              save_path=os.path.join(args.save_dir, "enorm_with_detectors.png"),
                              title=f"Ê (E_norm, λ_near={args.lam_near:.2f}, R={args.r_energy:.1f}) + detectors (valid)")

    # 2) Гистограмма радиусов
    plot_detector_radius_hist(space,
                              save_path=os.path.join(args.save_dir, "detector_radius_hist.png"))

    # 3) По образцам: исходник + A^λ и подсветка сработавших детекторов
    idxs = parse_indices(args.idx)
    plot_sample_activations(space, te, idxs,
                            lam_a=args.lam_d, mu_e_detect=args.mu_e_detect,
                            mu_d=args.mu_d, sigma=args.sigma,
                            save_dir=args.save_dir,
                            title_prefix="Sample")

    # 4) Гистограмма bits_on по сэмплу тест-наборa
    plot_bits_on_hist(space, te,
                      lam_a=args.lam_d, mu_e_detect=args.mu_e_detect, mu_d=args.mu_d, sigma=args.sigma,
                      sample_n=args.sample_n,
                      save_path=os.path.join(args.save_dir, "bits_on_hist.png"))

    # ==== ОТЧЁТ ПО ВЫБРАННЫМ ИНДЕКСАМ (JSON в stdout) ====
    te_imgs_sel = torch.stack([te[i][0] for i in idxs], dim=0).to(DML.DEVICE)
    te_codes_sel = DML.encode_batch_bool(te_imgs_sel)
    preds_batch = DML.predict_batch(space, class_hv, te_codes_sel,
                                    lam_a=args.lam_d, mu_e_detect=args.mu_e_detect, mu_d=args.mu_d,
                                    sigma=args.sigma)
    for j, ix in enumerate(idxs):
        pred, conf, bits_on = preds_batch[j]
        truth = int(te[ix][1])
        rec = {
            "idx": int(ix),
            "prediction": int(pred),
            "confidence": float(conf),
            "truth": truth,
            "bits_on": int(bits_on),
        }
        print(json.dumps(rec, ensure_ascii=False))

    # ==== Показ на экран (опционально) ====
    if args.show:
        # Быстрый предпросмотр главных артефактов
        imgs = [
            os.path.join(args.save_dir, "enorm_with_detectors.png"),
            os.path.join(args.save_dir, "detector_radius_hist.png"),
            os.path.join(args.save_dir, "bits_on_hist.png"),
        ]
        imgs += [os.path.join(args.save_dir, f) for f in sorted(os.listdir(args.save_dir)) if f.startswith("sample_")]
        for path in imgs:
            if not os.path.isfile(path):
                continue
            img = plt.imread(path)
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(w/150, h/150))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(os.path.basename(path))
            plt.show()


if __name__ == "__main__":
    main()
