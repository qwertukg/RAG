#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST → DAMP (§5) → Детекторы (§6) — реализация на PyTorch (с поддержкой CUDA).

Что изменено по сравнению с NumPy-версией:
  • Все тяжёлые куски переведены на PyTorch, с расчётом попарных Жаккаров на GPU.
  • Используются булевы коды (0/1) вместо битовой упаковки — это позволяет считать
    пересечения через матричное умножение на GPU: inter = A @ B^T.
  • DAMP хранит матрицу сходств на CPU (numpy), но строится из GPU-вычислений.
  • Построение детекторов — строго «по документу»: A^λ(c) от ОДНОГО центра,
    e_d — только по точкам с Ê≥μ_e в круге, запрет перекрытия центров на уровне,
    при конфликте оставляем детектор с большим заполнением n/r. Случайный bit_index.
  • Убрано всё неиспользуемое (kNN-бейзлайн, u64-утилиты и пр.).
  • ДОБАВЛЕНЫ оптимизации и параллелизм (без изменения алгоритма):
      – батчевое вычисление A для множества запросов на GPU (матрица Жаккара разом);
      – предвычисленная матрица детектирования W (HW×K) на GPU для мгновенного вычисления E(d,A) в батче;
      – двухфазная сборка детекторов: параллельная генерация кандидатов по сидингу + глобальное разрешение конфликтов «no-center-overlap, keep higher n/r»;
      – кэширование геометрии кругов и масок по Ê на этапе детектирования;
      – настройки батч‑размеров и потоков загрузки.

Зависимости: torch, torchvision, numpy, tqdm
"""

from __future__ import annotations
import os
import math
import json
from dataclasses import dataclass
from typing import Tuple, List, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== ПАРАМЕТРЫ =====================
# Сенсорный фронт и коды
GRID = 7                   # 28x28 -> 7x7 (avgpool 4x4)
LEVELS = 4                 # уровни квантизации 0..LEVELS
BITS_PER_CELL = 128
K_BITS_PER_LEVEL = 16      # вес кода уровня (const-weight)
NBITS = GRID * GRID * BITS_PER_CELL

# DAMP раскладка прототипов
DAMP_H = 32               # 32x32 = 1024 прототипов
DAMP_W = 32
LAM_FAR = 0.65  # §7.1.7: стартовый порог раскладки
LAM_NEAR = 0.80  # §7.1.7: финальный порог раскладки
ETA = 50.0  # ≈ жёсткая отсечка (η→∞), §7.1.7
R_ENERGY = 6.0            # радиус для энергетики точки (осталось без изменения)            # радиус для энергетики точки
PAIR_RADIUS = 16.0         # r≈d/2 для 32×32, §7.1.7         # локальный радиус для пары при шаге DAMP

# Детекторы
DETECT_K = 512            # максимум детекторов (= длина выходного кода)
MU_E_BUILD = 0.02         # порог по точкам при построении уровня
MU_E_DETECT = 0.02        # порог по точкам при детектировании
MU_D = 0.08               # порог уровня детектора (нормированная энергия)
LAM_D = 0.70               # уровни детекторов в табл. 5: 0.5…0.85; по умолчанию берём среднее              # λ для τ в детектировании/построении
DBSCAN_EPS = 5.0
DBSCAN_MIN_SAMPLES = 2
DETECT_ATTEMPTS = 1024

# Класс‑память
TARGET_DENSITY = 0.35

# Батчи / загрузка
BATCH_ENCODE = 4096        # размер батча для кодирования и активаций
NUM_WORKERS = min(os.cpu_count() or 8, 8)
PIN_MEMORY = True

# Прочее
TRAIN_LIMIT = None        # можно ограничить train для отладки
SEED = 42
DATA_DIR = "./data"
OUT_NPZ = "mnist_damp_detectors_torch.npz"
OUT_META = "mnist_damp_detectors_torch.meta.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN = torch.Generator(device="cpu").manual_seed(SEED)

# ===================== КОДБУК УРОВНЕЙ (Torch) =====================
# level_code_bool: [LEVELS+1, BITS_PER_CELL] (level 0 — все нули)

def make_level_code_bool() -> torch.BoolTensor:
    code = torch.zeros((LEVELS + 1, BITS_PER_CELL), dtype=torch.bool)
    for lvl in range(1, LEVELS + 1):
        idx = torch.randperm(BITS_PER_CELL, generator=GEN)[:K_BITS_PER_LEVEL]
        code[lvl, idx] = True
    return code

LEVEL_CODE_BOOL = make_level_code_bool().to(DEVICE)

# ===================== КОДИРОВАНИЕ (Torch) =====================

def encode_batch_bool(imgs: torch.Tensor) -> torch.BoolTensor:
    """imgs: [N,1,28,28] float in [0,1] → [N, NBITS] bool."""
    x = F.avg_pool2d(imgs, kernel_size=4, stride=4)  # [N,1,7,7]
    x = x.squeeze(1)                                  # [N,7,7]
    q = torch.clamp(torch.floor(x * LEVELS), 0, LEVELS).to(torch.long)  # [N,7,7]
    N = q.shape[0]
    codes = LEVEL_CODE_BOOL[q.view(N, -1)]            # [N,49,128]
    codes = codes.view(N, -1, BITS_PER_CELL)         # [N,49,128]
    return codes.reshape(N, -1)                       # [N, NBITS]

# ===================== Жаккар на GPU =====================

@torch.no_grad()
def jaccard_matrix_bool(A: torch.BoolTensor) -> torch.Tensor:
    Af = A.float()
    pop = Af.sum(dim=1)                           # [N]
    inter = Af @ Af.t()                           # [N,N]
    union = pop.unsqueeze(1) + pop.unsqueeze(0) - inter
    return (inter / union.clamp_min(1e-6)).to(torch.float32)

@torch.no_grad()
def jaccard_batch(P_codes: torch.BoolTensor, Q_codes: torch.BoolTensor) -> torch.Tensor:
    """P_codes: [Np,B], Q_codes: [Nq,B] → S: [Nq,Np] (Жаккар, на GPU)."""
    Pf, Qf = P_codes.float(), Q_codes.float()
    inter = Qf @ Pf.t()                            # [Nq,Np]
    pop_p = Pf.sum(dim=1)                          # [Np]
    pop_q = Qf.sum(dim=1)                          # [Nq]
    union = pop_q.unsqueeze(1) + pop_p.unsqueeze(0) - inter
    return inter / union.clamp_min(1e-6)

# ===================== DAMP (хранит S на CPU, считает S на GPU) =====================

@dataclass
class DAMPLayoutTorch:
    codes_bool: torch.BoolTensor   # [N,B] (на DEVICE)
    H: int
    W: int
    lam_far: float = LAM_FAR
    lam_near: float = LAM_NEAR
    eta: float = ETA
    r_energy: float = R_ENERGY
    pair_radius: float = PAIR_RADIUS

    def __post_init__(self):
        N = self.codes_bool.shape[0]
        assert self.H * self.W == N
        self.N = N
        self.grid_idx = np.random.default_rng(SEED).permutation(N).reshape(self.H, self.W)
        self._sim: np.ndarray | None = None  # [N,N] float32 на CPU

    def _ensure_sim(self):
        if self._sim is not None:
            return
        S = jaccard_matrix_bool(self.codes_bool)  # GPU
        self._sim = S.detach().cpu().numpy()
        self._sim = np.maximum(self._sim, self._sim.T).astype(np.float32)

    def coords_of(self, idx: int) -> Tuple[int, int]:
        y, x = np.argwhere(self.grid_idx == idx)[0]
        return int(y), int(x)

    def _local_window(self, cy: int, cx: int, r: float):
        H, W = self.H, self.W
        ys = np.arange(max(0, int(cy - r)), min(H, int(cy + r) + 1))
        xs = np.arange(max(0, int(cx - r)), min(W, int(cx + r) + 1))
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        dy = (Y - cy).astype(np.float32)
        dx = (X - cx).astype(np.float32)
        D = np.sqrt(dy * dy + dx * dx)
        mask = D <= r
        return Y[mask], X[mask], D[mask]

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _sim_lambda(self, base: np.ndarray, lam: float) -> np.ndarray:
        return base * self._sigmoid(self.eta * (base - lam))

    # ===== Векторизованная нормированная энергия точек (E_norm) =====
    def compute_E_norm(self, lam: float) -> np.ndarray:
        self._ensure_sim()
        N = self.N
        ys, xs = np.divmod(np.arange(N, dtype=np.int32), self.W)
        Y = ys[:, None]
        X = xs[:, None]
        Y2 = ys[None, :]
        X2 = xs[None, :]
        D = np.sqrt((Y - Y2) ** 2 + (X - X2) ** 2).astype(np.float32)  # [N,N]
        Wmask = (D <= self.r_energy) & (D > 0)
        W = np.zeros_like(D, dtype=np.float32)
        W[Wmask] = 1.0 / np.maximum(D[Wmask], 1e-6)
        S_l = self._sim_lambda(self._sim, lam)
        E = (W * S_l).sum(axis=1, dtype=np.float32)  # [N]
        E = E.reshape(self.H, self.W)
        m = float(E.max()) if E.size else 1.0
        return (E / max(m, 1e-9)).astype(np.float32)

    def point_energy(self, idx: int, r: float | None = None, lam: float | None = None) -> float:
        if r is None: r = self.r_energy
        if lam is None: lam = self.lam_far
        cy, cx = self.coords_of(idx)
        Y, X, D = self._local_window(cy, cx, r)
        neigh = self.grid_idx[Y, X].ravel()
        base = np.clip(self._sim[idx, neigh], 0.0, 1.0)
        s = self._sim_lambda(base, lam)
        D = np.maximum(D, 1e-6)
        return float((s / D).sum())

    def _pair_energy(self, i1: int, i2: int, mode: str = "far") -> Tuple[float, float]:
        self._ensure_sim()
        y1, x1 = self.coords_of(i1)
        y2, x2 = self.coords_of(i2)
        r = self.pair_radius if self.pair_radius > 0 else max(self.H, self.W)
        Y1, X1, D1 = self._local_window(y1, x1, r)
        Y2, X2, D2 = self._local_window(y2, x2, r)
        idx1 = self.grid_idx[Y1, X1].ravel()
        idx2 = self.grid_idx[Y2, X2].ravel()
        lam = self.lam_far if mode == "far" else self.lam_near
        s1_self = self._sim_lambda(self._sim[i1, idx1], lam)
        s2_self = self._sim_lambda(self._sim[i2, idx2], lam)
        s1_cross = self._sim_lambda(self._sim[i1, idx2], lam)
        s2_cross = self._sim_lambda(self._sim[i2, idx1], lam)
        D1 = np.maximum(D1.ravel(), 1e-6)
        D2 = np.maximum(D2.ravel(), 1e-6)
        if mode == "far":
            phi_c = float((s1_self * D1).sum() + (s2_self * D2).sum())
            phi_s = float((s1_cross * D2).sum() + (s2_cross * D1).sum())
        else:
            phi_c = float((s1_self / D1).sum() + (s2_self / D2).sum())
            phi_s = float((s1_cross / D2).sum() + (s2_cross / D1).sum())
        return phi_c, phi_s

    def _random_pairs(self, p: int) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int,int]] = []
        for _ in range(p):
            a = np.random.randint(0, self.N)
            b = np.random.randint(0, self.N)
            while b == a:
                b = np.random.randint(0, self.N)
            pairs.append((int(a), int(b)))
        return pairs

    def step(self, p: int = 4096, mode: str = "far") -> int:
        swapped = 0
        for (i1, i2) in self._random_pairs(p):
            phi_c, phi_s = self._pair_energy(i1, i2, mode=mode)
            better = (phi_s < phi_c) if mode == "far" else (phi_s > phi_c)
            if better:
                y1, x1 = self.coords_of(i1)
                y2, x2 = self.coords_of(i2)
                self.grid_idx[y1, x1], self.grid_idx[y2, x2] = self.grid_idx[y2, x2], self.grid_idx[y1, x1]
                swapped += 1
        return swapped

    def run(self, steps_far: int = 4, steps_near: int = 4, p_per_step: int = 4096, min_near_steps: int = 2) -> None:
        for _ in tqdm(range(steps_far), desc="DAMP far"):
            if self.step(p=p_per_step, mode="far") == 0:
                break
        for i in tqdm(range(steps_near), desc="DAMP near"):
            swaps = self.step(p=p_per_step, mode="near")
            if (i + 1) < min_near_steps:
                continue
            if swaps == 0:
                break

# ===================== Детекторы (строго по документу) =====================

@dataclass
class Detector:
    c: Tuple[float, float]
    r: float
    lam: float
    n_points: int
    energy: float
    bit_index: int

class SimpleDBSCAN:
    def __init__(self, eps: float = 2.0, min_samples: int = 4):
        self.eps = eps; self.min_samples = min_samples
    def fit_predict(self, P: np.ndarray) -> np.ndarray:
        if len(P) == 0: return np.empty((0,), dtype=np.int32)
        M = P.shape[0]
        labels = -np.ones(M, dtype=np.int32)
        visited = np.zeros(M, dtype=bool)
        D = np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(axis=-1))
        cid = 0
        for i in range(M):
            if visited[i]: continue
            visited[i] = True
            neigh = np.where(D[i] <= self.eps)[0]
            if neigh.size < self.min_samples:
                labels[i] = -1; continue
            labels[i] = cid
            seeds = list(neigh); j = 0
            while j < len(seeds):
                q = seeds[j]
                if not visited[q]:
                    visited[q] = True
                    nq = np.where(D[q] <= self.eps)[0]
                    if nq.size >= self.min_samples:
                        for u in nq:
                            if u not in seeds: seeds.append(int(u))
                if labels[q] == -1:
                    labels[q] = cid
                j += 1
            cid += 1
        return labels

@dataclass
class DetectorSpace:
    layout: DAMPLayoutTorch
    out_bits: int = DETECT_K
    E_norm: np.ndarray | None = None
    detectors: List[Detector] | None = None

    # Кэш для быстрых детектирований
    grid_order_t: torch.LongTensor | None = None      # [HW]
    E_norm_t: torch.Tensor | None = None              # [HW]
    W_detect: torch.Tensor | None = None              # [HW, Kdet] веса Ê для детектирования
    den_detect: torch.Tensor | None = None            # [Kdet] знаменатель e_d
    det_bits: torch.LongTensor | None = None          # [Kdet] соответствие детектор→bit_index

    def __post_init__(self):
        if self.detectors is None: self.detectors = []
        # Векторизованно считаем E_norm
        self.E_norm = self.layout.compute_E_norm(lam=self.layout.lam_near)
        # Тензоры для GPU‑детектирования
        self.grid_order_t = torch.from_numpy(self.layout.grid_idx.reshape(-1)).to(DEVICE, dtype=torch.long)
        self.E_norm_t = torch.from_numpy(self.E_norm.reshape(-1)).to(DEVICE, dtype=torch.float32)

    @staticmethod
    def _sigmoid_np(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # ==== Батч‑активации на GPU ====
    @torch.no_grad()
    def activation_batch_from_codes(self, Q_bool: torch.BoolTensor, lam_a: float = LAM_D, batch: int = BATCH_ENCODE) -> Iterable[torch.Tensor]:
        """Итератор по батчам A_flat: [nq, HW] (на DEVICE).
           Считаем разом Жаккар Q против прототипов, применяем τ, затем переставляем
           столбцы по grid_idx, чтобы получить A в порядке клетки решётки.
        """
        P = self.layout.codes_bool  # [Np, B]
        order = self.grid_order_t   # [HW]
        for s in range(0, Q_bool.shape[0], batch):
            e = min(s + batch, Q_bool.shape[0])
            Q = Q_bool[s:e]
            S = jaccard_batch(P, Q)               # [nq,Np]
            tau = S * torch.sigmoid(self.layout.eta * (S - lam_a))
            tau_perm = tau.index_select(dim=1, index=order)  # [nq, HW]
            yield tau_perm.to(torch.float32)

    # ==== Подготовка матрицы детектирования (GPU) ====
    def finalize_detection_matrix(self, mu_e: float = MU_E_DETECT):
        if not self.detectors:
            self.W_detect = self.den_detect = self.det_bits = None
            return
        H, W = self.layout.H, self.layout.W
        HW = H * W
        Kd = len(self.detectors)
        Wmat = torch.zeros((HW, Kd), device=DEVICE, dtype=torch.float32)
        den = torch.zeros((Kd,), device=DEVICE, dtype=torch.float32)
        bits = torch.empty((Kd,), device=DEVICE, dtype=torch.long)

        E = self.E_norm.reshape(H, W)
        for k, d in enumerate(self.detectors):
            y0 = int(max(0, math.floor(d.c[0]-d.r)))
            y1 = int(min(H, math.ceil(d.c[0]+d.r)+1))
            x0 = int(max(0, math.floor(d.c[1]-d.r)))
            x1 = int(min(W, math.ceil(d.c[1]+d.r)+1))
            YY, XX = np.meshgrid(np.arange(y0,y1), np.arange(x0,x1), indexing='ij')
            circle = ((YY - d.c[0])**2 + (XX - d.c[1])**2) <= (d.r*d.r)
            subE = E[y0:y1, x0:x1]
            mask = circle & (subE >= mu_e)
            if not np.any(mask):
                continue
            lin = (YY * W + XX).reshape(-1)[mask.reshape(-1)]
            lin_t = torch.from_numpy(lin.astype(np.int64)).to(DEVICE)
            Wmat[lin_t, k] = torch.from_numpy(subE.reshape(-1)[mask.reshape(-1)].astype(np.float32)).to(DEVICE)
            den[k] = max(float(d.energy), 1e-12)
            bits[k] = int(d.bit_index)

        self.W_detect = Wmat
        self.den_detect = den
        self.det_bits = bits

    # ==== Детектирование батчами (GPU) ====
    @torch.no_grad()
    def detect_batch_from_codes(self, Q_bool: torch.BoolTensor, lam_a: float = LAM_D, mu_d: float = MU_D) -> torch.BoolTensor:
        assert self.W_detect is not None and self.den_detect is not None and self.det_bits is not None, "Call finalize_detection_matrix() first"
        out = torch.zeros((Q_bool.shape[0], self.out_bits), device=DEVICE, dtype=torch.bool)
        pos = 0
        for A_flat in self.activation_batch_from_codes(Q_bool, lam_a=lam_a, batch=BATCH_ENCODE):
            nq = A_flat.shape[0]
            S = (A_flat @ self.W_detect) / self.den_detect.clamp_min(1e-12)  # [nq,K]
            on = S >= mu_d
            idx = self.det_bits.unsqueeze(0).expand_as(on)
            out_part = torch.zeros((nq, self.out_bits), device=DEVICE, dtype=torch.bool)
            out_part.scatter_(dim=1, index=idx, src=on)
            out[pos:pos+nq] = out_part
            pos += nq
        return out

    # ==== Построение уровня детекторов (параллельные кандидаты + глобальное разрешение) ====
    def _candidates_from_seed(self, lam_d: float, eps: float, min_samples: int, mu_e: float, seed_flat: int) -> List[Detector]:
        H, W = self.layout.H, self.layout.W
        iy = int(seed_flat // W); ix = int(seed_flat % W)
        idx = int(self.layout.grid_idx[iy, ix])
        A = self.activation_from_center(idx, lam_a=lam_d)
        clusters = self._cluster_points(A, mu_e=mu_e, eps=eps, min_samples=min_samples)
        cands: List[Detector] = []
        for P in clusters:
            c_d = self._centroid(P, A)
            r_d = self._optimal_radius(P, c_d)
            y0 = int(max(0, math.floor(c_d[0]-r_d))); y1 = int(min(H, math.ceil(c_d[0]+r_d)+1))
            x0 = int(max(0, math.floor(c_d[1]-r_d))); x1 = int(min(W, math.ceil(c_d[1]+r_d)+1))
            YY, XX = np.meshgrid(np.arange(y0,y1), np.arange(x0,x1), indexing='ij')
            E = self.E_norm[y0:y1, x0:x1]
            circle = ((YY - c_d[0])**2 + (XX - c_d[1])**2) <= (r_d*r_d)
            mask = circle & (E >= mu_e)
            n_pts = int(mask.sum())
            if n_pts == 0:
                continue
            e_d = float((A[y0:y1, x0:x1][mask] * E[mask]).sum())
            cands.append(Detector(c=c_d, r=float(r_d), lam=float(lam_d), n_points=n_pts, energy=e_d, bit_index=-1))
        return cands

    def build_level(self, lam_d: float = LAM_D, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES,
                    mu_e: float = MU_E_BUILD, attempts: int = DETECT_ATTEMPTS, max_detectors: int | None = DETECT_K) -> None:
        H, W = self.layout.H, self.layout.W
        seeds = np.random.default_rng(SEED+1).permutation(H * W)[:attempts]
        self.layout._ensure_sim()

        candidates: List[Detector] = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futs = [ex.submit(self._candidates_from_seed, lam_d, eps, min_samples, mu_e, int(s)) for s in seeds]
            for f in tqdm(as_completed(futs), total=len(futs), desc=f"detectors λ={lam_d:.2f} (candidates)"):
                candidates.extend(f.result())

        def centers_overlap(c1, r1, c2, r2) -> bool:
            dy = c1[0]-c2[0]; dx = c1[1]-c2[1]
            d  = math.hypot(dy, dx)
            return (d <= r1) or (d <= r2)

        candidates.sort(key=lambda d: (d.n_points / max(d.r, 1e-6)), reverse=True)
        kept: List[Detector] = []
        for d in candidates:
            ok = True
            for k in kept:
                if centers_overlap(d.c, d.r, k.c, k.r):
                    ok = False
                    break
            if ok:
                kept.append(d)
            if max_detectors is not None and len(kept) >= max_detectors:
                break

        rng = np.random.default_rng(SEED+2)
        for d in kept:
            d.bit_index = int(rng.integers(0, self.out_bits))
        self.detectors = kept

    def activation_from_center(self, proto_idx: int, lam_a: float = LAM_D) -> np.ndarray:
        self.layout._ensure_sim()
        sims = self.layout._sim[proto_idx].astype(np.float32)
        tau = sims * self._sigmoid_np(self.layout.eta * (sims - lam_a))
        return tau.reshape(self.layout.H, self.layout.W)

    def _cluster_points(self, A: np.ndarray, mu_e: float, eps: float, min_samples: int) -> List[np.ndarray]:
        mask = (self.E_norm >= mu_e) & (A > 0)
        Ys, Xs = np.where(mask)
        if Ys.size == 0: return []
        P = np.stack([Ys.astype(np.float32), Xs.astype(np.float32)], axis=1)
        labels = SimpleDBSCAN(eps=eps, min_samples=min_samples).fit_predict(P)
        clusters: List[np.ndarray] = []
        for cid in sorted(set(labels.tolist())):
            if cid < 0: continue
            clusters.append(P[labels == cid])
        return clusters

    def _centroid(self, P: np.ndarray, A: np.ndarray) -> Tuple[float, float]:
        ys = np.clip(P[:,0].astype(int), 0, self.layout.H-1)
        xs = np.clip(P[:,1].astype(int), 0, self.layout.W-1)
        W = A[ys, xs] * self.E_norm[ys, xs]
        s = float(W.sum())
        if s <= 1e-12: return float(P[:,0].mean()), float(P[:,1].mean())
        return float((P[:,0]*W).sum()/s), float((P[:,1]*W).sum()/s)

    def _optimal_radius(self, P: np.ndarray, c: Tuple[float,float]) -> float:
        cy, cx = c
        r_all = np.sqrt((P[:,0]-cy)**2 + (P[:,1]-cx)**2)
        order = np.argsort(r_all)
        best_r, best_val, cnt = 1.0, -1.0, 0
        for idx in order:
            r = max(float(r_all[idx]), 1e-6)
            cnt += 1
            val = cnt/(math.pi*r*r)
            if val > best_val: best_val, best_r = val, r
        return float(best_r)

# ===================== КЛАСС‑ПАМЯТЬ И ПРЕДСКАЗАНИЯ =====================

def build_class_memory(space: DetectorSpace, codes_bool: torch.BoolTensor, labels: np.ndarray,
                       lam_a: float, mu_e_detect: float, mu_d: float,
                       detect_k: int, target_density: float) -> np.ndarray:
    space.finalize_detection_matrix(mu_e=mu_e_detect)
    class_hv = np.zeros((10, detect_k), dtype=bool)
    counts = np.zeros((10, detect_k), dtype=np.int32)

    with torch.no_grad():
        for s in tqdm(range(0, codes_bool.shape[0], BATCH_ENCODE), desc="Class memory (batch)"):
            e = min(s + BATCH_ENCODE, codes_bool.shape[0])
            Q = codes_bool[s:e]
            codes = space.detect_batch_from_codes(Q, lam_a=lam_a, mu_d=mu_d).detach().cpu().numpy()
            for i in range(Q.shape[0]):
                counts[int(labels[s+i])] += codes[i].astype(np.int32)

    active_mask = counts.sum(axis=0) > 0
    active_idx = np.where(active_mask)[0]
    num_active = int(active_idx.size)
    k_on = max(1, int(round(target_density * max(1, num_active))))
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

@torch.no_grad()
def predict_batch(space: DetectorSpace, class_hv: np.ndarray, Q_bool: torch.BoolTensor,
                  lam_a: float, mu_e_detect: float, mu_d: float) -> np.ndarray:
    space.finalize_detection_matrix(mu_e=mu_e_detect)
    codes = space.detect_batch_from_codes(Q_bool, lam_a=lam_a, mu_d=mu_d).detach().cpu().numpy()
    preds = []
    for i in range(codes.shape[0]):
        code = codes[i]
        best, arg = -1.0, 0
        for c in range(10):
            inter = np.count_nonzero(code & class_hv[c])
            uni = np.count_nonzero(code | class_hv[c])
            sim = 0.0 if uni == 0 else inter / uni
            if sim > best:
                best, arg = sim, c
        preds.append((arg, best, int(code.sum())))
    return np.array(preds, dtype=object)

# ===================== MAIN =====================

def main():
    print(f"[Info] Device: {DEVICE}")
    os.makedirs(DATA_DIR, exist_ok=True)

    train_ds = datasets.MNIST(DATA_DIR, train=True,  transform=ToTensor(), download=True)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, transform=ToTensor(), download=True)

    trn_limit = len(train_ds) if TRAIN_LIMIT is None else min(TRAIN_LIMIT, len(train_ds))
    train_imgs = torch.stack([train_ds[i][0] for i in range(trn_limit)], dim=0).to(DEVICE)
    train_lbls = np.array([int(train_ds[i][1]) for i in range(trn_limit)], dtype=np.int16)

    P = DAMP_H * DAMP_W
    proto_idx = torch.randperm(trn_limit, generator=GEN)[:P]
    proto_codes = encode_batch_bool(train_imgs[proto_idx])

    damp = DAMPLayoutTorch(codes_bool=proto_codes, H=DAMP_H, W=DAMP_W,
                           lam_far=LAM_FAR, lam_near=LAM_NEAR, eta=ETA, r_energy=R_ENERGY, pair_radius=PAIR_RADIUS)
    damp.run(steps_far=8, steps_near=8, p_per_step=16384, min_near_steps=2)

    space = DetectorSpace(layout=damp, out_bits=DETECT_K)
    space.build_level(lam_d=LAM_D, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                      mu_e=MU_E_BUILD, attempts=DETECT_ATTEMPTS, max_detectors=DETECT_K)
    print("[Detectors] built:", len(space.detectors))

    train_codes_bool = encode_batch_bool(train_imgs)
    class_hv = build_class_memory(space,
                                  codes_bool=train_codes_bool,
                                  labels=train_lbls,
                                  lam_a=LAM_D, mu_e_detect=MU_E_DETECT, mu_d=MU_D,
                                  detect_k=DETECT_K, target_density=TARGET_DENSITY)

    T = len(test_ds)
    test_imgs = torch.stack([test_ds[i][0] for i in range(T)], dim=0).to(DEVICE)
    test_codes_bool = encode_batch_bool(test_imgs)
    preds = predict_batch(space, class_hv, test_codes_bool,
                          lam_a=LAM_D, mu_e_detect=MU_E_DETECT, mu_d=MU_D)
    ok = sum(int(p[0] == int(test_ds[i][1])) for i, p in enumerate(preds))
    bits = np.array([p[2] for p in preds[:min(256, len(preds))]])
    acc = ok / T
    print(f"[Eval] acc@{T}={acc:.3f} | bits_on min/med/max: {bits.min()} / {np.median(bits)} / {bits.max()}")

    det_np = np.array([(d.c[0], d.c[1], d.r, d.lam, d.n_points, d.energy, d.bit_index) for d in space.detectors], dtype=np.float32)
    np.savez_compressed(
        OUT_NPZ,
        proto_idx=proto_idx.cpu().numpy(),
        damp_grid=damp.grid_idx.astype(np.int32),
        detectors=det_np,
        class_hv=class_hv.astype(np.uint8),
    )
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump({
            "GRID": GRID,
            "LEVELS": LEVELS,
            "BITS_PER_CELL": BITS_PER_CELL,
            "K_BITS_PER_LEVEL": K_BITS_PER_LEVEL,
            "DAMP": {
                "H": DAMP_H, "W": DAMP_W,
                "LAM_FAR": LAM_FAR, "LAM_NEAR": LAM_NEAR, "ETA": ETA,
                "R_ENERGY": R_ENERGY, "PAIR_RADIUS": PAIR_RADIUS
            },
            "DETECTORS": {
                "DETECT_K": DETECT_K, "LAM_D": LAM_D,
                "MU_E_BUILD": MU_E_BUILD, "MU_E_DETECT": MU_E_DETECT, "MU_D": MU_D,
                "DBSCAN_EPS": DBSCAN_EPS, "DBSCAN_MIN_SAMPLES": DBSCAN_MIN_SAMPLES,
                "ATTEMPTS": DETECT_ATTEMPTS,
                "strict_build": True, "nms_rule": "no-center-overlap, keep higher n/r, parallel candidates"
            },
            "TARGET_DENSITY": TARGET_DENSITY,
            "SEED": SEED,
            "DEVICE": str(DEVICE),
            "pipeline": [
                "28x28 -> 7x7 (avgpool + quantize)",
                "population coding (49 x 128 const-weight) + concatenation (bool)",
                "DAMP over P=H*W prototypes (Jaccard GPU, S on CPU)",
                "detector space (DBSCAN level, e_d on Ê>=μ_e, no center overlap, keep higher n/r, parallel candidates)",
                "class-memory (top-k only active bits; batch detect via A@W)",
                "inference by Jaccard to class vectors"
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {OUT_NPZ}, {OUT_META}")


if __name__ == "__main__":
    main()
