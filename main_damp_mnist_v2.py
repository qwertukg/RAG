#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST → DAMP (§5) → Детекторы (§6) — реализация на PyTorch (CUDA),
максимально распараллелена и векторизована без изменения алгоритма и
расхождений с документом.

Ключевые ускорения:
  • **GPU**: попарные Жаккары через матричное умножение (bool→float), TF32/AMP не используем по умолчанию
    (численная стабильность), но включаем высокую точность матмулов.
  • **Batch-пайплайн**: кодирование, активации A^λ и детектирование выполняются для батчей.
  • **Sparse ускорение детекторов**: для каждого детектора заранее строится разреженная матрица W ∈ R^{D×(H·W)}
    (в строке j — веса Ê для пикселей круга), и уровень s_batch = W · A_flat^T считается одним sparse@dense на GPU.
  • **OR по битам**: активации детекторов агрегируются в биты через scatter_reduce (amax) по batch'у на GPU.
  • **I/O**: DataLoader с pin_memory для CPU→GPU, без лишних копий.

Строго по документу:
  • Построение детекторов от ОДНОГО центра c (A^λ(c)), DBSCAN, центр/радиус, e_d по Ê≥μ_e в круге.
  • «Центры не перекрывать» на одном уровне; при конфликте — оставить детектор с бОльшим заполнением n/r.
  • Случайный bit_index; коллизии допустимы.

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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

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
LAM_FAR = 0.03
LAM_NEAR = 0.03
ETA = 12.0
R_ENERGY = 6.0            # радиус для энергетики точки
PAIR_RADIUS = 8.0         # локальный радиус для пары при шаге DAMP

# Детекторы
DETECT_K = 512            # максимум детекторов (= длина выходного кода)
MU_E_BUILD = 0.02         # порог по точкам при построении уровня
MU_E_DETECT = 0.02        # порог по точкам при детектировании
MU_D = 0.08               # порог уровня детектора (нормированная энергия)
LAM_D = 0.03              # λ для τ в детектировании/построении
DBSCAN_EPS = 5.0
DBSCAN_MIN_SAMPLES = 2
DETECT_ATTEMPTS = 1024

# Класс‑память
TARGET_DENSITY = 0.35

# Прочее
TRAIN_LIMIT = None        # можно ограничить train для отладки
SEED = 42
DATA_DIR = "./data"
OUT_NPZ = "mnist_damp_detectors_torch.npz"
OUT_META = "mnist_damp_detectors_torch.meta.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN = torch.Generator(device="cpu").manual_seed(SEED)

torch.backends.cudnn.benchmark = True
# TF32 (ускоряет матмулы на RTX Ampere+, минимально влияет на численность)
torch.set_float32_matmul_precision("high")

# ===================== КОДБУК УРОВНЕЙ (Torch) =====================
# level_code_bool: [LEVELS+1, BITS_PER_CELL] (level 0 — все нули)

def make_level_code_bool() -> torch.BoolTensor:
    code = torch.zeros((LEVELS + 1, BITS_PER_CELL), dtype=torch.bool)
    for lvl in range(1, LEVELS + 1):
        idx = torch.randperm(BITS_PER_CELL, generator=GEN)[:K_BITS_PER_LEVEL]
        code[lvl, idx] = True
    return code

LEVEL_CODE_BOOL = make_level_code_bool().to(DEVICE)

# ===================== КОДИРОВАНИЕ (batch) =====================

def encode_batch_bool(imgs: torch.Tensor) -> torch.BoolTensor:
    """imgs: [N,1,28,28] float in [0,1] → [N, NBITS] bool."""
    x = F.avg_pool2d(imgs, kernel_size=4, stride=4)  # [N,1,7,7]
    x = x.squeeze(1)                                  # [N,7,7]
    q = torch.clamp(torch.floor(x * LEVELS), 0, LEVELS).to(torch.long)  # [N,7,7]
    N = q.shape[0]
    codes = LEVEL_CODE_BOOL[q.view(N, -1)]            # [N,49,128]
    codes = codes.view(N, -1, BITS_PER_CELL)
    return codes.reshape(N, -1)

# ===================== Жаккар (GPU, batch) =====================

def jaccard_matrix_bool(A: torch.BoolTensor) -> torch.Tensor:
    """A: [N, B] bool → S: [N,N] float32 (Жаккар)."""
    Af = A.float()
    pop = Af.sum(dim=1)                           # [N]
    inter = Af @ Af.t()                           # [N,N]
    union = pop.unsqueeze(1) + pop.unsqueeze(0) - inter
    S = inter / union.clamp_min(1e-6)
    return S.to(torch.float32)

@torch.no_grad()
def jaccard_batch_to_all(Q: torch.BoolTensor, P_codes: torch.BoolTensor, P_pop: torch.Tensor) -> torch.Tensor:
    """Q: [M,B] bool коды запросов; P_codes: [N,B] bool; P_pop: [N] float
       → sims: [M,N] float на DEVICE.
    """
    Qf = Q.float()                                # [M,B]
    inter = Qf @ P_codes.float().t()              # [M,N]
    pop_q = Qf.sum(dim=1, keepdim=True)           # [M,1]
    union = pop_q + P_pop.unsqueeze(0) - inter    # [M,N]
    return inter / union.clamp_min(1e-6)

# ===================== DAMP (S на CPU, считает S на GPU) =====================

@dataclass
class DAMPLayoutTorch:
    codes_bool: torch.BoolTensor   # [N,B] (на DEVICE)
    pops: torch.Tensor             # [N] float (на DEVICE)
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
        with torch.no_grad():
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

    def point_energy(self, idx: int, r: float | None = None, lam: float | None = None) -> float:
        self._ensure_sim()
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
        rng = np.random.default_rng(SEED)
        for _ in range(p):
            a = int(rng.integers(0, self.N))
            b = int(rng.integers(0, self.N))
            while b == a:
                b = int(rng.integers(0, self.N))
            pairs.append((a, b))
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

    # Runtime структуры для батчевого детектирования на GPU
    W_coo: torch.Tensor | None = None      # sparse COO: [D, H*W]
    denom: torch.Tensor | None = None      # [D]
    bit_index: torch.Tensor | None = None  # [D] long
    prepared_mu_e: float | None = None

    def __post_init__(self):
        if self.detectors is None: self.detectors = []
        if self.E_norm is None:
            E = np.zeros((self.layout.H, self.layout.W), dtype=np.float32)
            for idx in range(self.layout.N):
                E.flat[idx] = self.layout.point_energy(idx, r=self.layout.r_energy, lam=self.layout.lam_near)
            m = float(E.max()) if E.size else 1.0
            self.E_norm = (E / max(m, 1e-9)).astype(np.float32)

    @staticmethod
    def _sigmoid_np(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def activation_tau_batch(self, Q_bool: torch.BoolTensor, lam_a: float) -> torch.Tensor:
        """Вернёт τ(Q) как [M, H*W] на DEVICE без CPU-копий."""
        sims = jaccard_batch_to_all(Q_bool, self.layout.codes_bool, self.layout.pops)  # [M,N]
        tau = sims * torch.sigmoid(self.layout.eta * (sims - lam_a))                   # [M,N]
        return tau  # [M, H*W] (N=H*W)

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

    @staticmethod
    def _centers_overlap(c1, r1, c2, r2) -> bool:
        dy = c1[0]-c2[0]; dx = c1[1]-c2[1]
        d  = math.hypot(dy, dx)
        return (d <= r1) or (d <= r2)

    def build_level(self, lam_d: float = LAM_D, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES,
                    mu_e: float = MU_E_BUILD, attempts: int = DETECT_ATTEMPTS, max_detectors: int | None = DETECT_K) -> None:
        H, W = self.layout.H, self.layout.W
        seeds = np.random.default_rng(SEED+1).permutation(H * W)[:attempts]
        self.layout._ensure_sim()

        for s in tqdm(seeds, desc=f"detectors λ={lam_d:.2f}"):
            iy = int(s // W); ix = int(s % W)
            idx = int(self.layout.grid_idx[iy, ix])
            sims = self.layout._sim[idx].astype(np.float32)   # [N]
            tau = sims * self._sigmoid_np(self.layout.eta * (sims - lam_d))
            A = tau.reshape(H, W)

            # кластеризация
            mask = (self.E_norm >= mu_e) & (A > 0)
            Ys, Xs = np.where(mask)
            if Ys.size == 0:
                continue
            P = np.stack([Ys.astype(np.float32), Xs.astype(np.float32)], axis=1)
            labels = SimpleDBSCAN(eps=eps, min_samples=min_samples).fit_predict(P)

            for cid in sorted(set(labels.tolist())):
                if cid < 0: continue
                Pc = P[labels == cid]
                c_d = self._centroid(Pc, A)
                r_d = self._optimal_radius(Pc, c_d)

                y0 = int(max(0, math.floor(c_d[0]-r_d))); y1 = int(min(H, math.ceil(c_d[0]+r_d)+1))
                x0 = int(max(0, math.floor(c_d[1]-r_d))); x1 = int(min(W, math.ceil(c_d[1]+r_d)+1))
                subA = A[y0:y1, x0:x1]; subE = self.E_norm[y0:y1, x0:x1]
                YY, XX = np.meshgrid(np.arange(y0,y1), np.arange(x0,x1), indexing='ij')

                circle = ((YY - c_d[0])**2 + (XX - c_d[1])**2) <= (r_d*r_d)
                mask_e = circle & (subE >= mu_e)
                n_pts = int(mask_e.sum())
                if n_pts == 0:
                    continue
                e_d = float((subA[mask_e] * subE[mask_e]).sum())

                # правило «центры не перекрывать» с приоритетом по n/r
                allow = True
                to_remove: List[int] = []
                new_fill = n_pts / max(r_d, 1e-6)
                for i, d in enumerate(self.detectors):
                    if abs(d.lam - lam_d) > 1e-9:
                        continue
                    if not self._centers_overlap(c_d, r_d, d.c, d.r):
                        continue
                    old_fill = d.n_points / max(d.r, 1e-6)
                    if new_fill > old_fill:
                        to_remove.append(i)
                    else:
                        allow = False
                        break
                if not allow:
                    continue
                for j in sorted(to_remove, reverse=True):
                    self.detectors.pop(j)

                bit = int(np.random.default_rng(SEED+2).integers(0, self.out_bits))
                self.detectors.append(Detector(c=c_d, r=float(r_d), lam=float(lam_d),
                                               n_points=n_pts, energy=e_d, bit_index=bit))
                if max_detectors is not None and len(self.detectors) >= max_detectors:
                    return

    # ======= Подготовка разреженной матрицы W для батчевого детектирования =======
    def prepare_runtime(self, mu_e_detect: float) -> None:
        if self.prepared_mu_e is not None and abs(self.prepared_mu_e - mu_e_detect) < 1e-12 and self.W_coo is not None:
            return
        H, W = self.layout.H, self.layout.W
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        denom = []
        bits = []
        En = self.E_norm  # np.ndarray [H,W]
        for j, d in enumerate(self.detectors):
            y0 = int(max(0, math.floor(d.c[0]-d.r))); y1 = int(min(H, math.ceil(d.c[0]+d.r)+1))
            x0 = int(max(0, math.floor(d.c[1]-d.r))); x1 = int(min(W, math.ceil(d.c[1]+d.r)+1))
            YY, XX = np.meshgrid(np.arange(y0,y1), np.arange(x0,x1), indexing='ij')
            circle = ((YY - d.c[0])**2 + (XX - d.c[1])**2) <= (d.r*d.r)
            mask = circle & (En[y0:y1, x0:x1] >= mu_e_detect)
            if not np.any(mask):
                # пустой — всё равно добавим пустую строку, чтобы индексы совпадали
                denom.append(max(float(0.0), 1e-12))
                bits.append(int(d.bit_index))
                continue
            ys, xs = np.where(mask)
            flat = (ys + y0) * W + (xs + x0)
            w = En[y0:y1, x0:x1][ys, xs].astype(np.float32)
            rows.extend([j]*len(flat))
            cols.extend(flat.tolist())
            vals.extend(w.tolist())
            dden = float(d.energy) if d.energy > 0.0 else float(w.sum())
            denom.append(max(dden, 1e-12))
            bits.append(int(d.bit_index))

        D = max(1, len(self.detectors))
        if len(cols) == 0:
            # ни одного валидного пикселя — делаем нулевую COO
            indices = torch.zeros((2,1), dtype=torch.long, device=DEVICE)
            values  = torch.zeros((1,), dtype=torch.float32, device=DEVICE)
        else:
            indices = torch.tensor([rows, cols], dtype=torch.long, device=DEVICE)
            values  = torch.tensor(vals, dtype=torch.float32, device=DEVICE)
        self.W_coo = torch.sparse_coo_tensor(indices, values, size=(D, H*W), device=DEVICE).coalesce()
        self.denom = torch.tensor(denom, dtype=torch.float32, device=DEVICE)
        self.bit_index = torch.tensor(bits, dtype=torch.long, device=DEVICE)
        self.prepared_mu_e = mu_e_detect

    # ======= Батчевое детектирование на GPU =======
    @torch.no_grad()
    def detect_from_batch(self, Q_bool: torch.BoolTensor, lam_a: float, mu_e: float, mu_d: float) -> torch.BoolTensor:
        assert self.W_coo is not None, "call prepare_runtime(mu_e) first"
        tau = self.activation_tau_batch(Q_bool, lam_a=lam_a)              # [M, H*W]
        # s = (W * A)^T: [D,HW] @ [HW,M] -> [D,M] -> [M,D]
        s = torch.sparse.mm(self.W_coo, tau.t()).t()                      # [M,D]
        s_norm = s / self.denom.clamp_min(1e-12)                          # [M,D]
        active_det = (s_norm >= mu_d)                                     # [M,D] bool
        # Свести к битам OR по детекторам с одинаковым bit_index
        M = active_det.shape[0]
        bits = torch.zeros((M, self.out_bits), dtype=torch.float32, device=DEVICE)
        index = self.bit_index.unsqueeze(0).expand(M, -1)                 # [M,D]
        bits.scatter_reduce_(1, index, active_det.float(), reduce="amax", include_self=False)
        return bits > 0.5                                                 # [M, out_bits] bool

# ===================== КЛАСС‑ПАМЯТЬ (batch) =====================

def build_class_memory(space: DetectorSpace, loader: DataLoader,
                       lam_a: float, mu_e_detect: float, mu_d: float,
                       detect_k: int, target_density: float) -> np.ndarray:
    counts = np.zeros((10, detect_k), dtype=np.int32)
    space.prepare_runtime(mu_e_detect)
    for imgs, labels in tqdm(loader, desc="Class memory (reduce)"):
        imgs = imgs.to(DEVICE, non_blocking=True)
        codes_bool = encode_batch_bool(imgs)
        bits = space.detect_from_batch(codes_bool, lam_a=lam_a, mu_e=mu_e_detect, mu_d=mu_d)  # [B,K]
        bits_cpu = bits.cpu().numpy().astype(np.int32)
        for c in range(10):
            mask = (labels.numpy() == c)
            if not np.any(mask):
                continue
            counts[c] += bits_cpu[mask].sum(axis=0)

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

# ===================== УТИЛИТЫ =====================

def make_loader(ds, batch_size=1024, shuffle=False) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=min(4, os.cpu_count() or 0), pin_memory=True, drop_last=False)

# ===================== MAIN =====================

def main():
    print(f"[Info] Device: {DEVICE}")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Данные
    train_ds = datasets.MNIST(DATA_DIR, train=True,  transform=ToTensor(), download=True)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, transform=ToTensor(), download=True)

    if TRAIN_LIMIT is not None:
        train_ds = Subset(train_ds, list(range(min(TRAIN_LIMIT, len(train_ds)))))

    # Прототипы DAMP: P = H*W случайных индексов
    P = DAMP_H * DAMP_W
    all_idx = torch.randperm(len(train_ds), generator=GEN)
    proto_idx = all_idx[:P].cpu().tolist()

    # Кодируем прототипы батчами
    proto_loader = make_loader(Subset(train_ds, proto_idx), batch_size=1024, shuffle=False)
    codes_list = []
    with torch.no_grad():
        for imgs, _ in tqdm(proto_loader, desc="Encode prototypes"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            codes_list.append(encode_batch_bool(imgs))
    proto_codes = torch.cat(codes_list, dim=0)             # [P, NBITS] bool
    proto_pops  = proto_codes.float().sum(dim=1)           # [P] float

    # DAMP раскладка (S на GPU → CPU)
    damp = DAMPLayoutTorch(codes_bool=proto_codes, pops=proto_pops, H=DAMP_H, W=DAMP_W,
                           lam_far=LAM_FAR, lam_near=LAM_NEAR, eta=ETA, r_energy=R_ENERGY, pair_radius=PAIR_RADIUS)
    damp.run(steps_far=8, steps_near=8, p_per_step=16384, min_near_steps=2)

    # Пространство детекторов
    space = DetectorSpace(layout=damp, out_bits=DETECT_K)
    space.build_level(lam_d=LAM_D, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                      mu_e=MU_E_BUILD, attempts=DETECT_ATTEMPTS, max_detectors=DETECT_K)
    print("[Detectors] built:", len(space.detectors))

    # Класс‑память (батчи)
    train_loader = make_loader(train_ds, batch_size=1024, shuffle=False)
    class_hv = build_class_memory(space, train_loader,
                                  lam_a=LAM_D, mu_e_detect=MU_E_DETECT, mu_d=MU_D,
                                  detect_k=DETECT_K, target_density=TARGET_DENSITY)

    # Инференс на тесте батчами
    test_loader = make_loader(test_ds, batch_size=1024, shuffle=False)
    ok = 0
    total = 0
    bits_on_stats: List[int] = []

    space.prepare_runtime(MU_E_DETECT)
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Eval class-memory"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            q_bool = encode_batch_bool(imgs)                        # [B, NBITS]
            bits = space.detect_from_batch(q_bool, lam_a=LAM_D, mu_e=MU_E_DETECT, mu_d=MU_D)  # [B,K]
            # Jaccard к класс-векторам (на GPU)
            hv = torch.from_numpy(class_hv).to(DEVICE, dtype=torch.bool)  # [10,K]
            inter = (bits.unsqueeze(1) & hv.unsqueeze(0)).sum(dim=2).float()   # [B,10]
            union = (bits.unsqueeze(1) | hv.unsqueeze(0)).sum(dim=2).clamp_min(1).float()
            sim = inter / union
            pred = sim.argmax(dim=1)
            ok += (pred.cpu() == labels).sum().item()
            total += labels.size(0)
            bits_on_stats.extend(bits.sum(dim=1).int().cpu().tolist())

    acc = ok / max(1, total)
    bits_arr = np.array(bits_on_stats)
    print(f"[Eval] acc@{total}={acc:.3f} | bits_on min/med/max: {bits_arr.min()} / {np.median(bits_arr)} / {bits_arr.max()}")

    # Сохранение артефактов
    det_np = np.array([(d.c[0], d.c[1], d.r, d.lam, d.n_points, d.energy, d.bit_index) for d in space.detectors], dtype=np.float32)
    np.savez_compressed(
        OUT_NPZ,
        proto_idx=np.array(proto_idx, dtype=np.int32),
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
                "strict_build": True, "nms_rule": "no-center-overlap, keep higher n/r"
            },
            "TARGET_DENSITY": TARGET_DENSITY,
            "SEED": SEED,
            "DEVICE": str(DEVICE),
            "pipeline": [
                "28x28 -> 7x7 (avgpool + quantize)",
                "population coding (49 x 128 const-weight) + concatenation (bool)",
                "DAMP over P=H*W prototypes (Jaccard GPU, S on CPU)",
                "detector space (DBSCAN level, e_d on Ê>=μ_e, no center overlap, keep higher n/r)",
                "class-memory (top-k only active bits, batched)",
                "inference by Jaccard to class vectors (batched)"
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {OUT_NPZ}, {OUT_META}")


if __name__ == "__main__":
    main()
