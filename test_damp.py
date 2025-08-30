#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Смоук‑тест для новой реализации на PyTorch/CUDA:
  TRAIN: строит прототипы → DAMP (GPU) → детекторы (гл. 6) → class_hv, затем предсказывает
  LOAD: загружает сохранённые артефакты (.npz) и сразу предсказывает

Совместим с модулем: main_damp_mnist_torch.py

Примеры:
  # Полное обучение и предсказание для трёх индексов
  python test_damp.py --idx 0,1,2 --train-n 10000 --proto 32x32 --detect-k 1024 \
    --lam-d 0.03 --mu-e-build 0.01 --mu-e-detect 0.01 --mu-d 0.06 \
    --eps 6.0 --min-samples 2 --attempts 2048 \
    --steps-far 8 --steps-near 8 --p-per-step 16384 --min-near-steps 2 --target-density 0.35 \
    --save damp_assets.npz

  # Быстрая проверка (без переобучения)
  python test_damp.py --idx 0,1,2 --load damp_assets.npz
"""

from __future__ import annotations
import argparse
import json
from typing import List, Tuple

import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

# --- Импорт новой реализации ---
import main_damp_mnist_v2 as m


# ================= Утилиты парсинга =================

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


# ================= Обёртки над реализацией =================

@torch.no_grad()
def encode_one_bool(img28_np: np.ndarray) -> torch.BoolTensor:
    """28×28 numpy → [NBITS] bool на DEVICE."""
    t = torch.from_numpy(img28_np).unsqueeze(0).unsqueeze(0).to(m.DEVICE)  # [1,1,28,28]
    return m.encode_batch_bool(t)[0]  # [NBITS]


def build_space(train_imgs_t: torch.Tensor, H: int, W: int, train_n: int,
                lam_far: float, lam_near: float, steps_far: int, steps_near: int,
                p_per_step: int, min_near_steps: int,
                detect_k: int, lam_d: float, mu_e_build: float, eps: float,
                min_samples: int, attempts: int,
                rng: np.random.Generator):
    """Собрать DAMP+детекторы с нуля. Возвращает (space, damp, proto_idx)."""
    P = H * W
    idx = rng.choice(train_n, size=P, replace=False)
    proto_codes = m.encode_batch_bool(train_imgs_t[idx].to(m.DEVICE))  # [P, NBITS] bool
    proto_pops = proto_codes.float().sum(dim=1)                        # [P]

    damp = m.DAMPLayoutTorch(proto_codes, proto_pops, H=H, W=W,
                             lam_far=lam_far, lam_near=lam_near,
                             eta=m.ETA, r_energy=m.R_ENERGY, pair_radius=m.PAIR_RADIUS)
    damp.run(steps_far=steps_far, steps_near=steps_near, p_per_step=p_per_step, min_near_steps=min_near_steps)

    space = m.DetectorSpace(layout=damp, out_bits=detect_k)
    space.build_level(lam_d=lam_d, eps=eps, min_samples=min_samples,
                      mu_e=mu_e_build, attempts=attempts, max_detectors=detect_k)
    return space, damp, idx


def save_assets(path: str, damp: "m.DAMPLayoutTorch", space: "m.DetectorSpace",
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
                train_ds: datasets.VisionDataset) -> tuple[m.DetectorSpace, np.ndarray]:
    """Восстановить пространство детекторов и класс‑память. Протокоды восстанавливаем по proto_idx."""
    z = np.load(path, allow_pickle=True)
    grid = z["damp_grid"].astype(int)
    proto_idx = z["proto_idx"].astype(int)
    det = z["detectors"]  # [M,7]
    class_hv = z.get("class_hv", None)

    # восстановим коды прототипов
    imgs = torch.stack([train_ds[i][0] for i in proto_idx], dim=0).to(m.DEVICE)  # [P,1,28,28]
    proto_codes = m.encode_batch_bool(imgs)
    proto_pops  = proto_codes.float().sum(dim=1)

    # DAMP
    H, W = grid.shape
    damp = m.DAMPLayoutTorch(proto_codes, proto_pops, H=H, W=W,
                             lam_far=m.LAM_FAR, lam_near=m.LAM_NEAR,
                             eta=m.ETA, r_energy=m.R_ENERGY, pair_radius=m.PAIR_RADIUS)
    damp.grid_idx = grid

    # Space
    max_bit = int(det[:, 6].max()) if det.size else -1
    out_bits = max(detect_k_arg, max_bit + 1)
    space = m.DetectorSpace(damp, out_bits=out_bits)
    space.detectors = [
        m.Detector(c=(float(r[0]), float(r[1])), r=float(r[2]), lam=float(r[3]),
                   n_points=int(r[4]), energy=float(r[5]), bit_index=int(r[6]))
        for r in det
    ]

    if class_hv is not None:
        class_hv = class_hv.astype(bool)
    return space, class_hv


# ================= CLI =================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=str, default="0", help="индекс(ы) test MNIST: '0' или '0,1,2'")

    # TRAIN режим
    ap.add_argument("--train-n", type=int, default=8000, help="сколько train примеров для памяти")
    ap.add_argument("--proto", type=str, default="32x32", help="HxW прототипов для DAMP, напр. 32x32")
    ap.add_argument("--detect-k", type=int, default=getattr(m, "DETECT_K", 512), help="макс. детекторов / длина кода")
    # DAMP
    ap.add_argument("--lam-far", type=float, default=getattr(m, "LAM_FAR", 0.03))
    ap.add_argument("--lam-near", type=float, default=getattr(m, "LAM_NEAR", 0.03))
    ap.add_argument("--steps-far", type=int, default=4)
    ap.add_argument("--steps-near", type=int, default=4)
    ap.add_argument("--p-per-step", type=int, default=4096)
    ap.add_argument("--min-near-steps", type=int, default=2, help="гарантировать ≥N near-шагов")
    # Детекторы (построение)
    ap.add_argument("--lam-d", type=float, default=getattr(m, "LAM_D", 0.03))
    ap.add_argument("--mu-e-build", type=float, default=getattr(m, "MU_E_BUILD", 0.02))
    ap.add_argument("--eps", type=float, default=getattr(m, "DBSCAN_EPS", 5.0))
    ap.add_argument("--min-samples", type=int, default=getattr(m, "DBSCAN_MIN_SAMPLES", 2))
    ap.add_argument("--attempts", type=int, default=getattr(m, "DETECT_ATTEMPTS", 1024))
    # Детекторы (детектирование)
    ap.add_argument("--mu-e-detect", type=float, default=getattr(m, "MU_E_DETECT", 0.02))
    ap.add_argument("--mu-d", type=float, default=getattr(m, "MU_D", 0.08))
    # Память классов
    ap.add_argument("--target-density", type=float, default=getattr(m, "TARGET_DENSITY", 0.35))

    # LOAD/SAVE
    ap.add_argument("--load", type=str, default=None, help="путь к .npz с готовыми детекторами/раскладкой/памятью")
    ap.add_argument("--save", type=str, default=None, help="сохранить артефакты в .npz после обучения")

    # Прочее
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data_dir = getattr(m, "DATA_DIR", "./data")

    # ==== ДАННЫЕ ====
    tr = datasets.MNIST(data_dir, train=True,  transform=ToTensor(), download=True)
    te = datasets.MNIST(data_dir, train=False, transform=ToTensor(), download=True)

    # ==== LOAD режим (без обучения) ====
    if args.load:
        space, class_hv = load_assets(args.load, detect_k_arg=args.detect_k, train_ds=tr)
        print(f"[LOAD] Detectors loaded: {len(space.detectors)} | out_bits={space.out_bits}")
        if class_hv is None:
            train_n = min(args.train_n, len(tr))
            train_imgs_t = torch.stack([tr[i][0] for i in range(train_n)], dim=0).to(m.DEVICE)
            train_codes = m.encode_batch_bool(train_imgs_t)
            train_lbls = np.array([int(tr[i][1]) for i in range(train_n)], dtype=np.int16)
            class_hv = m.build_class_memory(space, imgs=train_codes, labels=train_lbls,
                                            lam_a=args.lam_d, mu_e_detect=args.mu_e_detect, mu_d=args.mu_d,
                                            detect_k=space.out_bits, target_density=args.target_density)
    else:
        # ==== TRAIN режим ====
        train_n = min(args.train_n, len(tr))
        train_imgs_t = torch.stack([tr[i][0] for i in range(train_n)], dim=0).to(m.DEVICE)  # [N,1,28,28]
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

        train_codes = m.encode_batch_bool(train_imgs_t)
        class_hv = m.build_class_memory(space, imgs=train_codes, labels=train_lbls,
                                        lam_a=args.lam_d, mu_e_detect=args.mu_e_detect, mu_d=args.mu_d,
                                        detect_k=args.detect_k, target_density=args.target_density)

        if args.save:
            save_assets(args.save, damp=damp, space=space, proto_idx=proto_idx, class_hv=class_hv)
            print(f"[SAVE] Артефакты сохранены в {args.save}")

    # ==== ПРЕДСКАЗАНИЯ ====
    idxs = parse_indices(args.idx)
    results = []
    for ix in idxs:
        img, truth = te[ix]
        img = img.squeeze(0).numpy()
        q_bool = encode_one_bool(img)
        pred, conf, bits_on = m.predict_digit(space, class_hv, q_bool,
                                              lam_a=args.lam_d, mu_e_detect=args.mu_e_detect, mu_d=args.mu_d)
        results.append({
            "idx": ix,
            "prediction": pred,
            "confidence": conf,
            "truth": int(truth),
            "bits_on": bits_on,
        })

    # ==== Диагностика по bits_on на батче из теста ====
    sample_n = min(128, len(te))
    te_imgs_t = torch.stack([te[i][0] for i in range(sample_n)], dim=0).to(m.DEVICE)
    te_codes = m.encode_batch_bool(te_imgs_t)
    bits_on_list = []
    for i in range(sample_n):
        code, _ = space.detect_from_code(te_codes[i], lam_a=args.lam_d, mu_e=args.mu_e_detect, mu_d=args.mu_d)
        bits_on_list.append(int(code.sum()))
    bits_on_arr = np.array(bits_on_list)
    print(f"[Check] bits_on over {sample_n} tests — min/med/max: {bits_on_arr.min()} / {np.median(bits_on_arr)} / {bits_on_arr.max()} | zero_frac={(bits_on_arr==0).mean():.2%}")

    # ==== Отчёт ====
    if len(results) == 1:
        print(results[0])
    else:
        ok = sum(int(r["prediction"] == r["truth"]) for r in results)
        print(f"Batch {len(results)} — acc={ok/len(results):.3f}")
        for r in results:
            print(r)


if __name__ == "__main__":
    main()
