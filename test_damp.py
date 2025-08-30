#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Смоук-тест для новой реализации на PyTorch/CUDA:
  TRAIN: строит прототипы → DAMP (GPU) → детекторы (гл. 6) → class_hv, затем предсказывает
  LOAD: загружает сохранённые артефакты (.npz) и сразу предсказывает

Совместим с модулем: main_damp_mnist_torch.py

Примеры:
  # Полное обучение и предсказание для трёх индексов
  python test_damp.py --idx 0,1,2 --train-n 10000 --proto 32x32 --detect-k 1024 \
    --lam-d 0.70 --mu-e-build 0.02 --mu-e-detect 0.02 --mu-d 0.08 \
    --eps 5.0 --min-samples 2 --attempts 1024 \
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

# --- Импорт НОВОЙ реализации ---
import main_damp_mnist_torch as DML


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
    t = torch.from_numpy(img28_np).unsqueeze(0).unsqueeze(0).to(DML.DEVICE)  # [1,1,28,28]
    return DML.encode_batch_bool(t)[0]  # [NBITS]


def build_space(train_imgs_t: torch.Tensor, H: int, W: int, train_n: int,
                lam_far: float, lam_near: float, steps_far: int, steps_near: int,
                p_per_step: int, min_near_steps: int,
                detect_k: int, lam_d: float, mu_e_build: float, eps: float,
                min_samples: int, attempts: int,
                rng: np.random.Generator):
    """Собрать DAMP+детекторы с нуля. Возвращает (space, damp, proto_idx_np)."""
    P = H * W
    idx = rng.choice(train_n, size=P, replace=False)
    proto_codes = DML.encode_batch_bool(train_imgs_t[idx].to(DML.DEVICE))     # [P, NBITS] bool

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
    with open(DML.OUT_META, "w", encoding="utf-8") as f:
        json.dump({
            "SENSOR_FRONTEND": {
                "type": "binary_pixels",
                "NBITS": DML.NBITS,
                "BIN_THRESHOLD": DML.BIN_THRESHOLD,
                "note": "Прямое кодирование 28×28 → 784 bool (без 7×7 и популяционного кодирования)."
            },
            "DAMP": {
                "H": DML.DAMP_H, "W": DML.DAMP_W,
                "LAM_FAR": DML.LAM_FAR, "LAM_NEAR": DML.LAM_NEAR, "ETA": DML.ETA,
                "R_ENERGY": DML.R_ENERGY, "PAIR_RADIUS": DML.PAIR_RADIUS
            },
            "DETECTORS": {
                "DETECT_K": DML.DETECT_K, "LAM_D": DML.LAM_D,
                "MU_E_BUILD": DML.MU_E_BUILD, "MU_E_DETECT": DML.MU_E_DETECT, "MU_D": DML.MU_D,
                "DBSCAN_EPS": DML.DBSCAN_EPS, "DBSCAN_MIN_SAMPLES": DML.DBSCAN_MIN_SAMPLES,
                "ATTEMPTS": DML.DETECT_ATTEMPTS,
                "SIGMA": DML.SIGMA,
                "strict_build": True,
                "nms_rule": "no-center-overlap, keep higher n/r, parallel candidates",
                "multi_stimuli_aggregation": "A = max_s τ(sim(s,·))"
            },
            "TARGET_DENSITY": DML.TARGET_DENSITY,
            "SEED": DML.SEED,
            "DEVICE": str(DML.DEVICE),
            "pipeline": [
                "binary pixel coding (28×28 ≥ threshold) → 784-bit bool code",
                "DAMP over P=H×W prototypes (Jaccard GPU, S on CPU, grid_idx-aware)",
                "detector space (DBSCAN level, e_d on Ê≥μ_e, no center overlap, keep higher n/r, parallel candidates)",
                "class-memory (top-k only active bits; batch detect via A@W; optional σ saturation; multi-stimuli max)",
                "inference by Jaccard to class vectors"
            ]
        }, f, ensure_ascii=False, indent=2)


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
    ap.add_argument("--idx", type=str, default="0",
                    help="Список индексов из тестового MNIST для предсказания, без пробелов (напр. '0' или '0,1,2').")

    # TRAIN режим
    ap.add_argument("--train-n", type=int, default=10000,
                    help="Сколько train-примеров использовать для построения класс-памяти и выборки прототипов для DAMP (§5, §6).")
    ap.add_argument("--proto", type=str, default="32x32",
                    help="Размер решётки прототипов DAMP в формате HxW; общее число прототипов P=H·W влияет на топологию и качество (§5).")
    ap.add_argument("--detect-k", type=int, default=getattr(DML, "DETECT_K", 512),
                    help="Максимальное число детекторов/длина выходного кода; каждому детектору соответствует bit_index; коллизии допустимы (§6).")

    # DAMP
    ap.add_argument("--lam-far", type=float, default=getattr(DML, "LAM_FAR", 0.65),
                    help="Порог λ для далёкой фазы DAMP в τ(x)=x·σ(η(x−λ)); определяет «репульсию» на дальних шагах (§5.3–§5.4).")
    ap.add_argument("--lam-near", type=float, default=getattr(DML, "LAM_NEAR", 0.80),
                    help="Порог λ для ближней фазы DAMP; регулирует локальное упорядочивание и «притяжение» (§5.3–§5.4).")
    ap.add_argument("--steps-far", type=int, default=8,
                    help="Число итераций фазы 'far' (перестановки пар по критерию φ_s<φ_c); 0 отключает фазу (§5.4).")
    ap.add_argument("--steps-near", type=int, default=8,
                    help="Число итераций фазы 'near' (локальная оптимизация по критерию φ_s>φ_c) (§5.4).")
    ap.add_argument("--p-per-step", type=int, default=16384,
                    help="Сколько случайных пар прототипов рассматривать на одном шаге DAMP; больше — быстрее сходимость, но дольше шаг (§5.4).")
    ap.add_argument("--min-near-steps", type=int, default=2,
                    help="Минимально гарантированное число near-итераций до ранней остановки даже при нулевых перестановках; стабилизирует локальную топологию (§5.4).")

    # Детекторы (построение)
    ap.add_argument("--lam-d", type=float, default=getattr(DML, "LAM_D", 0.70),
                    help="Порог λ в τ для построения/детектирования A^λ; управляет тем, какие прототипы вносят вклад в карту активаций (§6.1).")
    ap.add_argument("--mu-e-build", type=float, default=getattr(DML, "MU_E_BUILD", 0.02),
                    help="Порог по нормированной энергии Ê при построении: учитываются только точки с Ê≥μ_e при поиске кластеров на A^λ (§6.2).")
    ap.add_argument("--eps", type=float, default=getattr(DML, "DBSCAN_EPS", 5.0),
                    help="Параметр ε (радиус окрестности) алгоритма DBSCAN для кластеризации точек {Ê≥μ_e} на карте A^λ (§6.2).")
    ap.add_argument("--min-samples", type=int, default=getattr(DML, "DBSCAN_MIN_SAMPLES", 2),
                    help="Параметр min_samples DBSCAN — минимальный размер кластера, чтобы считать его валидным (§6.2).")
    ap.add_argument("--attempts", type=int, default=getattr(DML, "DETECT_ATTEMPTS", 1024),
                    help="Сколько сидов/центров перебрать для генерации кандидатов детекторов; больше — выше покрытие, дольше расчёт (§6.2).")

    # Детекторы (детектирование)
    ap.add_argument("--mu-e-detect", type=float, default=getattr(DML, "MU_E_DETECT", 0.02),
                    help="Порог Ê при инференсе: в круге детектора учитываются только точки с Ê≥μ_e при вычислении E(d,A) (§6.3).")
    ap.add_argument("--mu-d", type=float, default=getattr(DML, "MU_D", 0.08),
                    help="Порог срабатывания детектора μ_d: детектор активен если E(d,A)/e_d ≥ μ_d; нормировка на энергию детектора (§6.3).")

    # Память классов
    ap.add_argument("--target-density", type=float, default=getattr(DML, "TARGET_DENSITY", 0.35),
                    help="Целевая доля единиц в класс-векторе после порогования по счётчикам только среди реально активных битов; задаёт разреженность памяти (§6.4).")

    # LOAD/SAVE
    ap.add_argument("--load", type=str, default=None,
                    help="Путь к .npz с артефактами (раскладка DAMP, детекторы, class_hv); в этом режиме обучение пропускается, версии алгоритма должны совпадать (§5–§6).")
    ap.add_argument("--save", type=str, default=None,
                    help="Сохранить артефакты (раскладка DAMP, детекторы, class_hv) в .npz для последующих запусков без переобучения (§5–§6).")

    # Прочее
    ap.add_argument("--seed", type=int, default=0,
                    help="ГПСЧ для воспроизводимости (выбор прототипов, порядок пар DAMP, сиды детекторов, битовые индексы и пр.).")
    ap.add_argument("--r-energy", type=float, default=4.0,
                    help="Радиус для энергетики точки при расчёте Ê (меньше = локальнее, острее).")

    args = ap.parse_args()

    DML.R_ENERGY = args.r_energy

    rng = np.random.default_rng(args.seed)
    data_dir = getattr(DML, "DATA_DIR", "./data")

    print(f"[Info] Device: {DML.DEVICE}")

    # ==== ДАННЫЕ ====
    tr = datasets.MNIST(data_dir, train=True,  transform=ToTensor(), download=True)
    te = datasets.MNIST(data_dir, train=False, transform=ToTensor(), download=True)

    # ==== LOAD режим (без обучения) ====
    if args.load:
        space, class_hv = load_assets(args.load, detect_k_arg=args.detect_k, train_ds=tr)
        print(f"[LOAD] Detectors loaded: {len(space.detectors)} | out_bits={space.out_bits}")
        if class_hv is None:
            # Нужно построить память классов на лету (на GPU)
            train_n = min(args.train_n, len(tr))
            train_imgs_t = torch.stack([tr[i][0] for i in range(train_n)], dim=0).to(DML.DEVICE)
            train_codes = DML.encode_batch_bool(train_imgs_t)
            train_lbls = np.array([int(tr[i][1]) for i in range(train_n)], dtype=np.int16)
            class_hv = DML.build_class_memory(space, codes_bool=train_codes, labels=train_lbls,
                                              lam_a=args.lam_d, mu_e_detect=args.mu_e_detect, mu_d=args.mu_d,
                                              detect_k=space.out_bits, target_density=args.target_density)
    else:
        # ==== TRAIN режим ====
        train_n = min(args.train_n, len(tr))
        train_imgs_t = torch.stack([tr[i][0] for i in range(train_n)], dim=0).to(DML.DEVICE)  # [N,1,28,28]
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

        rs = [d.r for d in space.detectors]
        print("r min/med/max:", (min(rs) if rs else None), (np.median(rs) if rs else None), (max(rs) if rs else None))

        # оценим долю клеток с Ê≥μ_e на этапе ПОСТРОЕНИЯ (если она велика — кластеры будут пухлыми)
        mu_e_b = args.mu_e_build
        E = space.E_norm  # [H,W]
        print("frac(E>=mu_e_build):", float((E >= mu_e_b).mean()))

        train_codes = DML.encode_batch_bool(train_imgs_t)
        class_hv = DML.build_class_memory(space, codes_bool=train_codes, labels=train_lbls,
                                          lam_a=args.lam_d, mu_e_detect=args.mu_e_detect, mu_d=args.mu_d,
                                          detect_k=args.detect_k, target_density=args.target_density)

        if args.save:
            save_assets(args.save, damp=damp, space=space, proto_idx=proto_idx, class_hv=class_hv)
            print(f"[SAVE] Артефакты сохранены в {args.save}")

    # ==== ПРЕДСКАЗАНИЯ ====
    idxs = parse_indices(args.idx)
    results = []
    # подготовим векторизованный инференс
    te_imgs_sel = torch.stack([te[i][0] for i in idxs], dim=0).to(DML.DEVICE)
    te_codes_sel = DML.encode_batch_bool(te_imgs_sel)
    preds_batch = DML.predict_batch(space, class_hv, te_codes_sel,
                                    lam_a=args.lam_d, mu_e_detect=args.mu_e_detect, mu_d=args.mu_d)
    for j, ix in enumerate(idxs):
        pred, conf, bits_on = preds_batch[j]
        truth = int(te[ix][1])
        results.append({
            "idx": ix,
            "prediction": int(pred),
            "truth": truth,
            "confidence": float(conf),
            "bits_on": int(bits_on),
        })

    # ==== Диагностика по bits_on на батче из теста ====
    sample_n = min(128, len(te))
    te_imgs_t = torch.stack([te[i][0] for i in range(sample_n)], dim=0).to(DML.DEVICE)
    te_codes = DML.encode_batch_bool(te_imgs_t)
    space.finalize_detection_matrix(mu_e=args.mu_e_detect)
    print("[Detectors] built:", len(space.detectors), "| valid:", space.W_detect.shape[1])

    with torch.no_grad():
        codes = space.detect_batch_from_codes(te_codes, lam_a=args.lam_d, mu_d=args.mu_d).cpu().numpy()
    bits_on_arr = codes.sum(axis=1).astype(int)
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
