#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный смоук‑тест для пайплайна DAMP → детекторы → класс‑память
с двумя режимами работы:
  1) TRAIN: построить всё с нуля (прототипы → DAMP → детекторы → class_hv)
  2) LOAD: загрузить готовые артефакты из .npz и сразу делать предсказания

Особенности:
  • Объединённая активация детекторов по top‑K прототипам (seeds_k) и мягкое NMS по IoU (iou_thr)
  • Класс‑память строится только по реально активным битам (top‑k на активном подмножестве)
  • Мини‑диагностика: распечатка распределения bits_on по батчу тестовых картинок

Совместим с `main_da_damp_full.py` (предпочтительно) и с более ранним `main_da_damp.py`.

Примеры запуска:
  # Полное обучение и предсказание для трёх индексов
  python smoke_predict.py --idx 0,1,2 --train-n 10000 --proto 32x32 --detect-k 1024 \
    --lam-d 0.03 --mu-e-build 0.01 --mu-e-detect 0.01 --mu-d 0.06 \
    --eps 6.0 --min-samples 2 --attempts 2048 --seeds-k 24 --iou-thr 0.7 \
    --steps-far 8 --steps-near 8 --p-per-step 16384 --min-near-steps 2 --target-density 0.35 \
    --save damp_assets.npz

  # Быстрый прогон без переобучения (загружаем готовые артефакты)
  python smoke_predict.py --idx 0,1,2 --load damp_assets.npz
"""

from __future__ import annotations
import argparse
import json
import math
from typing import List, Tuple

import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor

# --- Импорт основной реализации ---
import main_damp_mnist as m


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


# ================= Кодирование и общие куски =================

def one_code(img28: np.ndarray) -> np.ndarray:
    """28×28 float32 [0..1] → packed uint64 вектор (6272 бит = 98×uint64)."""
    return m.blocks_to_u64bits(m.encode_image_blocks(img28))


def build_class_memory(space: "m.DetectorSpace", imgs: np.ndarray, labels: np.ndarray,
                       lam_a: float, mu_e_detect: float, mu_d: float,
                       detect_k: int, target_density: float) -> np.ndarray:
    """Построить класс‑память: top‑k только по реально активным битам."""
    counts = np.zeros((10, detect_k), dtype=np.int32)
    for i in range(len(imgs)):
        q64 = one_code(imgs[i])
        code, _ = space.detect_from_code(q64, lam_a=lam_a, mu_e=mu_e_detect, mu_d=mu_d)
        counts[int(labels[i])] += code.astype(np.int32)

    active_mask = counts.sum(axis=0) > 0
    active_idx = np.where(active_mask)[0]
    num_active = int(active_idx.size)

    k_on = max(1, int(round(target_density * max(1, num_active))))
    class_hv = np.zeros((10, detect_k), dtype=bool)
    if num_active == 0:
        return class_hv

    for c in range(10):
        cls_counts_active = counts[c, active_idx]
        if k_on >= num_active:
            sel_rel = np.arange(num_active)
        else:
            sel_rel = np.argpartition(cls_counts_active, -k_on)[-k_on:]
        sel = active_idx[sel_rel]
        class_hv[c, sel] = True
    return class_hv


def predict_digit(space: "m.DetectorSpace", class_hv: np.ndarray, img28: np.ndarray,
                  lam_a: float, mu_e_detect: float, mu_d: float) -> Tuple[int, float, int]:
    q64 = one_code(img28)
    code, _ = space.detect_from_code(q64, lam_a=lam_a, mu_e=mu_e_detect, mu_d=mu_d)

    best, arg = -1.0, 0
    for c in range(10):
        inter = np.count_nonzero(code & class_hv[c])
        uni = np.count_nonzero(code | class_hv[c])
        sim = 0.0 if uni == 0 else inter / uni
        if sim > best:
            best, arg = sim, c
    return int(arg), float(best), int(code.sum())


# ================= TRAIN: прототипы → DAMP → детекторы =================

def build_space(train_imgs: np.ndarray, H: int, W: int, train_n: int,
                lam_far: float, lam_near: float, steps_far: int, steps_near: int,
                p_per_step: int, min_near_steps: int,
                detect_k: int, lam_d: float, mu_e_build: float, eps: float,
                min_samples: int, attempts: int, seeds_k: int, iou_thr: float,
                rng: np.random.Generator) -> Tuple["m.DetectorSpace", "m.DAMPLayoutPacked", np.ndarray, np.ndarray]:
    # Прототипы
    P = H * W
    proto_idx = rng.choice(train_n, size=P, replace=False)
    proto_codes64 = np.stack([one_code(train_imgs[i]) for i in proto_idx])
    proto_pops = m.popcount_u64_vec(proto_codes64)

    # DAMP
    damp = m.DAMPLayoutPacked(proto_codes64, proto_pops, H=H, W=W,
                              lam_far=lam_far, lam_near=lam_near,
                              eta=m.ETA, r_energy=m.R_ENERGY, pair_radius=m.PAIR_RADIUS)
    # Гарантируем несколько near‑шагов, если метод поддерживает параметр
    try:
        damp.run(steps_far=steps_far, steps_near=steps_near, p_per_step=p_per_step, min_near_steps=min_near_steps)  # type: ignore
    except TypeError:
        damp.run(steps_far=steps_far, steps_near=steps_near, p_per_step=p_per_step)

    # Детекторы
    space = m.DetectorSpace(damp, out_bits=detect_k)
    try:
        space.build_level(lam_d=lam_d, eps=eps, min_samples=min_samples,
                          mu_e=mu_e_build, attempts=attempts, max_detectors=detect_k,
                          seeds_k=seeds_k, iou_thr=iou_thr)  # type: ignore
    except TypeError:
        space.build_level(lam_d=lam_d, eps=eps, min_samples=min_samples,
                          mu_e=mu_e_build, attempts=attempts, max_detectors=detect_k)

    return space, damp, proto_codes64, proto_pops


# ================= Сохранение / загрузка артефактов =================

def save_assets(path: str, damp: "m.DAMPLayoutPacked", space: "m.DetectorSpace",
                proto_codes64: np.ndarray, proto_pops: np.ndarray, class_hv: np.ndarray) -> None:
    det = np.array([(d.c[0], d.c[1], d.r, d.lam, d.n_points, d.energy, d.bit_index)
                    for d in space.detectors], dtype=np.float32)
    np.savez_compressed(path,
        damp_grid=damp.grid_idx.astype(np.int32),
        proto_codes64=proto_codes64,
        proto_pops=proto_pops,
        detectors=det,
        class_hv=class_hv.astype(np.uint8)
    )


def load_assets(path: str, detect_k_arg: int) -> Tuple["m.DetectorSpace", np.ndarray]:
    z = np.load(path, allow_pickle=True)
    grid = z["damp_grid"].astype(int)
    proto_codes64 = z["proto_codes64"]
    proto_pops = z["proto_pops"]
    det = z["detectors"]  # shape [M,7]
    class_hv = z.get("class_hv", None)

    # Восстановить DAMP и пространство
    damp = m.DAMPLayoutPacked(proto_codes64, proto_pops, H=grid.shape[0], W=grid.shape[1],
                              lam_far=getattr(m, "LAM_FAR", 0.03), lam_near=getattr(m, "LAM_NEAR", 0.03),
                              eta=m.ETA, r_energy=m.R_ENERGY, pair_radius=m.PAIR_RADIUS)
    damp.grid_idx = grid

    # out_bits ≥ max(bit_index)+1
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
    ap.add_argument("--proto", type=str, default="24x24", help="HxW прототипов для DAMP, напр. 24x24")
    ap.add_argument("--detect-k", type=int, default=getattr(m, "DETECT_K", 256), help="макс. детекторов / длина кода")
    # DAMP
    ap.add_argument("--lam-far", type=float, default=getattr(m, "LAM_FAR", 0.03))
    ap.add_argument("--lam-near", type=float, default=getattr(m, "LAM_NEAR", 0.03))
    ap.add_argument("--steps-far", type=int, default=4)
    ap.add_argument("--steps-near", type=int, default=4)
    ap.add_argument("--p-per-step", type=int, default=4096)
    ap.add_argument("--min-near-steps", type=int, default=2, help="гарантировать ≥N near-шагов")
    # Детекторы (построение)
    ap.add_argument("--lam-d", type=float, default=getattr(m, "LAM_D", 0.03))
    ap.add_argument("--mu-e-build", type=float, default=getattr(m, "MU_E_BUILD", 0.04))
    ap.add_argument("--eps", type=float, default=getattr(m, "DBSCAN_EPS", 3.0))
    ap.add_argument("--min-samples", type=int, default=getattr(m, "DBSCAN_MIN_SAMPLES", 2))
    ap.add_argument("--attempts", type=int, default=getattr(m, "DETECT_ATTEMPTS", 240))
    ap.add_argument("--seeds-k", type=int, default=getattr(m, "SEEDS_TOPK", 16))
    ap.add_argument("--iou-thr", type=float, default=getattr(m, "IOU_THR", 0.6))
    # Детекторы (детектирование)
    ap.add_argument("--mu-e-detect", type=float, default=getattr(m, "MU_E_DETECT", 0.03))
    ap.add_argument("--mu-d", type=float, default=getattr(m, "MU_D", 0.10))
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
        space, class_hv = load_assets(args.load, detect_k_arg=args.detect_k)
        print(f"[LOAD] Detectors loaded: {len(space.detectors)} | out_bits={space.out_bits}")
        if class_hv is None:
            # Нужно построить память классов на лету
            train_n = min(args.train_n, len(tr))
            train_imgs = np.stack([tr[i][0].squeeze(0).numpy() for i in range(train_n)], axis=0)
            train_lbls = np.array([int(tr[i][1]) for i in range(train_n)], dtype=np.int16)
            class_hv = build_class_memory(space, train_imgs, train_lbls,
                                          lam_a=args.lam_d, mu_e_detect=args.mu_e_detect, mu_d=args.mu_d,
                                          detect_k=space.out_bits, target_density=args.target_density)
    else:
        # ==== TRAIN режим ====
        train_n = min(args.train_n, len(tr))
        train_imgs = np.stack([tr[i][0].squeeze(0).numpy() for i in range(train_n)], axis=0)
        train_lbls = np.array([int(tr[i][1]) for i in range(train_n)], dtype=np.int16)

        H, W = parse_hw(args.proto)
        space, damp, proto_codes64, proto_pops = build_space(
            train_imgs=train_imgs, H=H, W=W, train_n=train_n,
            lam_far=args.lam_far, lam_near=args.lam_near,
            steps_far=args.steps_far, steps_near=args.steps_near,
            p_per_step=args.p_per_step, min_near_steps=args.min_near_steps,
            detect_k=args.detect_k, lam_d=args.lam_d, mu_e_build=args.mu_e_build, eps=args.eps,
            min_samples=args.min_samples, attempts=args.attempts, seeds_k=args.seeds_k, iou_thr=args.iou_thr,
            rng=rng)
        print(f"[TRAIN] Detectors built: {len(space.detectors)}")

        class_hv = build_class_memory(space, train_imgs, train_lbls,
                                      lam_a=args.lam_d, mu_e_detect=args.mu_e_detect, mu_d=args.mu_d,
                                      detect_k=args.detect_k, target_density=args.target_density)

        if args.save:
            save_assets(args.save, damp=damp, space=space,
                        proto_codes64=proto_codes64, proto_pops=proto_pops, class_hv=class_hv)
            print(f"[SAVE] Артефакты сохранены в {args.save}")

    # ==== ПРЕДСКАЗАНИЯ ====
    idxs = parse_indices(args.idx)
    results = []
    for ix in idxs:
        img, truth = te[ix]
        img = img.squeeze(0).numpy()
        pred, conf, bits_on = predict_digit(space, class_hv, img,
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
    bits_on_list = []
    for i in range(sample_n):
        img = te[i][0].squeeze(0).numpy()
        q64 = one_code(img)
        code, _ = space.detect_from_code(q64, lam_a=args.lam_d, mu_e=args.mu_e_detect, mu_d=args.mu_d)
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
