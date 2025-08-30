#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный пайплайн MNIST с DAMP (§5) и детекторами «по документу» (§6),
включая:
  • кодирование 28×28 → 7×7 → популяционные коды 49×128 (const-weight) → конкатенация;
  • упакованные битсеты (6272 бита → 98×uint64) и Жаккар над ними;
  • DAMP-раскладку прототипов (pairwise Jaccard, far/near, гарантируем min шагов near);
  • построение пространства детекторов: A^{λ}(S) от top-K прототипов, DBSCAN, центр/радиус,
    энергия детектора e_d = Σ A·Ê внутри круга, NMS по IoU кругов с критерием «плотности энергии»;
  • детектирование: E(d,A) = (Σ_{Ê≥μ_e, внутри круга} A·Ê) / e_d ≥ μ_d;
  • класс‑память: счётчики → top‑k только среди реально встречавшихся битов, Жаккар к класс‑векторам.

Зависимости: numpy, tqdm, torchvision, torch
"""

from __future__ import annotations
import os, json, math
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence, Tuple, List

import numpy as np
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import ToTensor

# ===================== ПАРАМЕТРЫ =====================
# Сенсорный фронт и коды (как в main_da.py)
GRID = 7                   # 28x28 -> 7x7 (avgpool 4x4)
LEVELS = 4                 # уровни квантизации 0..LEVELS
BITS_PER_CELL = 128
K_BITS_PER_LEVEL = 16      # вес кода уровня (const-weight)

# k-NN (базовый путь для сравнения)
KNN_K = 7

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
SEEDS_TOPK = 16           # сколько прототипов объединять в A^{λ}(S)
IOU_THR = 0.6             # допуск перекрытия детекторов

# Класс‑память
TARGET_DENSITY = 0.35

# Прочее
TRAIN_LIMIT = None        # можно ограничить train для отладки
SEED = 42
DATA_DIR = "./data"
OUT_NPZ = "mnist_damp_detectors.npz"
OUT_META = "mnist_damp_detectors.meta.json"
WORKERS = min(os.cpu_count() or 8, 12)
CHUNK_ENCODE = 256
CHUNK_TEST = 128
CHUNK_CLASS = 128

# ===================== RNG и LUT =====================
rng = np.random.default_rng(SEED)
LUT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# ===================== УТИЛИТЫ КОДИРОВАНИЯ =====================

def avgpool_28_to_7(x28: np.ndarray) -> np.ndarray:
    return x28.reshape(GRID, 28//GRID, GRID, 28//GRID).mean(axis=(1,3))

def quantize_levels(x7: np.ndarray) -> np.ndarray:
    q = np.floor(x7 * LEVELS).astype(np.int32)
    q[q > LEVELS] = LEVELS
    q[q < 0] = 0
    return q

# код уровня: const-weight индексы -> упакованный 128-битный (как 2×uint64)

def const_weight_indices(bits: int, k: int, rng_: np.random.Generator) -> np.ndarray:
    idx = rng_.choice(bits, size=k, replace=False).astype(np.int32)
    return np.sort(idx)

def idx_to_u64_pair(idx: np.ndarray) -> np.ndarray:
    w = np.zeros((2,), dtype=np.uint64)
    for b in idx:
        if b < 64: w[0] |= (np.uint64(1) << np.uint64(b))
        else:      w[1] |= (np.uint64(1) << np.uint64(b - 64))
    return w

LEVEL_CODE_IDX = [np.empty((0,), dtype=np.int32)] + [
    const_weight_indices(BITS_PER_CELL, K_BITS_PER_LEVEL, rng) for _ in range(LEVELS)
]
LEVEL_CODE = [np.zeros(2, dtype=np.uint64)] + [idx_to_u64_pair(i) for i in LEVEL_CODE_IDX[1:]]

# ---- кодирование 28x28 -> (49, 2*uint64) ----

def encode_image_blocks(img28: np.ndarray) -> np.ndarray:
    x7 = avgpool_28_to_7(img28)
    q  = quantize_levels(x7)
    code = np.empty((GRID*GRID, 2), dtype=np.uint64)
    t = 0
    for r in range(GRID):
        for c in range(GRID):
            code[t] = LEVEL_CODE[int(q[r, c])]
            t += 1
    return code

# ---- развёртка в битовый вектор 49*128=6272 и упаковка в uint64 ----

NBITS = GRID*GRID*BITS_PER_CELL            # 6272
NBLK64 = (NBITS + 63)//64                  # 98

def blocks_to_u64bits(code_blocks: np.ndarray) -> np.ndarray:
    # code_blocks: (49, 2) uint64 — 128 бит на блок
    bits = np.unpackbits(code_blocks.view(np.uint8), axis=-1)
    bits = bits.reshape(code_blocks.shape[0], -1).reshape(-1).astype(np.uint8)  # (6272,)
    pad = (NBLK64*64 - NBITS)
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    v = np.zeros((NBLK64,), dtype=np.uint64)
    by = bits.reshape(NBLK64, 64)
    for i in range(64):
        v |= (by[:, i].astype(np.uint64) << np.uint64(i))
    return v

# ---- Жаккар для упакованных битсетов ----

def popcount_u64_vec(arr_u64: np.ndarray) -> np.ndarray:
    v = arr_u64.view(np.uint8)
    return LUT8[v].sum(axis=-1, dtype=np.int32)

def jaccard_u64(a: np.ndarray, b: np.ndarray, pc_a: int | None = None, pc_b: np.ndarray | None = None) -> np.ndarray:
    inter = popcount_u64_vec(np.bitwise_and(b, a))
    if pc_a is None: pc_a = int(popcount_u64_vec(a).sum())
    if pc_b is None: pc_b = popcount_u64_vec(b)
    uni = np.maximum(pc_a + pc_b - inter, 1)
    return inter / uni

# ===================== DAMP =====================

@dataclass
class DAMPLayoutPacked:
    codes64: np.ndarray          # [N, NBLK64] uint64
    pops: np.ndarray             # [N] int32 popcount
    H: int
    W: int
    lam_far: float = LAM_FAR
    lam_near: float = LAM_NEAR
    eta: float = ETA
    r_energy: float = R_ENERGY
    pair_radius: float = PAIR_RADIUS
    rng: np.random.Generator = np.random.default_rng(SEED)

    def __post_init__(self):
        self.N = self.codes64.shape[0]
        assert self.H * self.W == self.N
        self.grid_idx = self.rng.permutation(self.N).reshape(self.H, self.W)
        self._sim: np.ndarray | None = None  # [N,N] float32

    def _ensure_sim(self):
        if self._sim is not None:
            return
        N = self.N
        self._sim = np.zeros((N, N), dtype=np.float32)
        for i in tqdm(range(N), desc="pairwise sim (DAMP)"):
            a = self.codes64[i]
            sim = jaccard_u64(a, self.codes64, pc_a=int(self.pops[i]), pc_b=self.pops)
            self._sim[i] = sim.astype(np.float32)
        # симметризуем
        self._sim = np.maximum(self._sim, self._sim.T)

    def coords_of(self, idx: int) -> Tuple[int, int]:
        y, x = np.argwhere(self.grid_idx == idx)[0]
        return int(y), int(x)

    def _local_window(self, cy: int, cx: int, r: float):
        ys = np.arange(max(0, int(cy - r)), min(self.H, int(cy + r) + 1))
        xs = np.arange(max(0, int(cx - r)), min(self.W, int(cx + r) + 1))
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
        pairs = []
        for _ in range(p):
            a, b = rng.integers(0, self.N, size=2)
            while b == a:
                b = rng.integers(0, self.N)
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

    # Вектор сходств «запрос против всех прототипов»
    def sim_to_all(self, q64: np.ndarray) -> np.ndarray:
        pc_q = int(popcount_u64_vec(q64).sum())
        return jaccard_u64(q64, self.codes64, pc_a=pc_q, pc_b=self.pops)

# ===================== ДЕТЕКТОРЫ =====================

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
    layout: DAMPLayoutPacked
    out_bits: int = DETECT_K
    E_norm: np.ndarray | None = None
    detectors: List[Detector] | None = None
    rng: np.random.Generator = np.random.default_rng(SEED+1)

    def __post_init__(self):
        if self.detectors is None: self.detectors = []
        if self.E_norm is None:
            # Нормированная энергия точек (через point_energy, затем max-нормировка)
            E = np.zeros((self.layout.H, self.layout.W), dtype=np.float32)
            for idx in range(self.layout.N):
                E.flat[idx] = self.layout.point_energy(idx, r=self.layout.r_energy, lam=self.layout.lam_near)
            m = float(E.max()) if E.size else 1.0
            self.E_norm = (E / max(m, 1e-9)).astype(np.float32)

    def activation_from_code(self, q64: np.ndarray, lam_a: float = LAM_D) -> np.ndarray:
        base = self.layout.sim_to_all(q64).astype(np.float32)
        # τ(x) = x·σ(η(x−λ))
        tau = base * (1.0 / (1.0 + np.exp(-self.layout.eta * (base - lam_a))))
        return tau.reshape(self.layout.H, self.layout.W)

    def activation_from_indices(self, proto_indices: np.ndarray, lam_a: float = LAM_D) -> np.ndarray:
        """Объединённая активация A^{λ}(S) по множеству прототипов S (максимум по τ)."""
        self.layout._ensure_sim()
        sims = self.layout._sim[proto_indices]  # shape: [K, N]
        if sims.ndim == 1:
            sims = sims[None, :]
        tau = sims * (1.0 / (1.0 + np.exp(-self.layout.eta * (sims - lam_a))))
        A = tau.max(axis=0).reshape(self.layout.H, self.layout.W)
        return A

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

    @staticmethod
    def _circle_iou(c1, r1, c2, r2) -> float:
        dy = c1[0]-c2[0]; dx = c1[1]-c2[1]
        d  = math.hypot(dy, dx)
        if d >= r1 + r2: return 0.0
        if d <= abs(r1 - r2):
            inter = math.pi * min(r1, r2)**2
        else:
            a1 = math.acos(max(-1.0, min(1.0, (d*d + r1*r1 - r2*r2) / (2*d*r1))))
            a2 = math.acos(max(-1.0, min(1.0, (d*d + r2*r2 - r1*r1) / (2*d*r2))))
            inter = r1*r1*a1 + r2*r2*a2 - 0.5*math.sqrt(max(0.0, (-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2)))
        union = math.pi*(r1*r1 + r2*r2) - inter
        return 0.0 if union <= 0 else inter/union

    def _next_free_bit(self) -> int:
        used = {d.bit_index for d in self.detectors}
        for b in range(self.out_bits):
            if b not in used: return b
        return int(self.rng.integers(0, self.out_bits))

    def build_level(self, lam_d: float = LAM_D, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES,
                    mu_e: float = MU_E_BUILD, attempts: int = DETECT_ATTEMPTS, max_detectors: int | None = DETECT_K,
                    seeds_k: int = SEEDS_TOPK, iou_thr: float = IOU_THR) -> None:
        H, W = self.layout.H, self.layout.W
        seeds = self.rng.permutation(H * W)[:attempts]
        self.layout._ensure_sim()
        for s in tqdm(seeds, desc=f"detectors λ={lam_d:.2f}"):
            iy = int(s // W); ix = int(s % W)
            idx = int(self.layout.grid_idx[iy, ix])
            sims_row = self.layout._sim[idx]
            k = min(int(seeds_k), sims_row.shape[0])
            top = np.argpartition(sims_row, -k)[-k:]
            A = self.activation_from_indices(top, lam_a=lam_d)
            clusters = self._cluster_points(A, mu_e=mu_e, eps=eps, min_samples=min_samples)
            if not clusters: continue
            for P in clusters:
                c_d = self._centroid(P, A)
                r_d = self._optimal_radius(P, c_d)
                y0 = int(max(0, math.floor(c_d[0]-r_d))); y1 = int(min(H, math.ceil(c_d[0]+r_d)+1))
                x0 = int(max(0, math.floor(c_d[1]-r_d))); x1 = int(min(W, math.ceil(c_d[1]+r_d)+1))
                subA = A[y0:y1, x0:x1]; subE = self.E_norm[y0:y1, x0:x1]
                YY, XX = np.meshgrid(np.arange(y0,y1), np.arange(x0,x1), indexing='ij')
                mask = ((YY - c_d[0])**2 + (XX - c_d[1])**2) <= (r_d*r_d)
                e_d = float((subA[mask] * subE[mask]).sum())
                # NMS-подобная вставка
                allow = True; to_remove: List[int] = []
                new_fill = e_d / (math.pi * max(r_d, 1e-6)**2)
                for i, d in enumerate(self.detectors):
                    if abs(d.lam - lam_d) > 1e-9:
                        continue
                    iou = self._circle_iou(c_d, r_d, d.c, d.r)
                    if iou <= iou_thr:
                        continue
                    old_fill = d.energy / (math.pi * max(d.r, 1e-6)**2)
                    if new_fill > old_fill:
                        to_remove.append(i)
                    else:
                        allow = False; break
                if not allow:
                    continue
                for j in sorted(to_remove, reverse=True):
                    self.detectors.pop(j)
                bit = int(self.rng.integers(0, self.out_bits))
                self.detectors.append(Detector(c=c_d, r=float(r_d), lam=float(lam_d),
                                               n_points=int(mask.sum()), energy=e_d, bit_index=bit))
                if max_detectors is not None and len(self.detectors) >= max_detectors:
                    return

    def detect_from_code(self, q64: np.ndarray, lam_a: float = LAM_D, mu_e: float = MU_E_DETECT, mu_d: float = MU_D) -> Tuple[np.ndarray, List[int]]:
        A = self.activation_from_code(q64, lam_a=lam_a)
        H, W = self.layout.H, self.layout.W
        code = np.zeros((self.out_bits,), dtype=bool)
        active: List[int] = []
        for i, d in enumerate(self.detectors):
            y0 = int(max(0, math.floor(d.c[0]-d.r)))
            y1 = int(min(H, math.ceil(d.c[0]+d.r)+1))
            x0 = int(max(0, math.floor(d.c[1]-d.r)))
            x1 = int(min(W, math.ceil(d.c[1]+d.r)+1))
            subA = A[y0:y1, x0:x1]; subE = self.E_norm[y0:y1, x0:x1]
            YY, XX = np.meshgrid(np.arange(y0,y1), np.arange(x0,x1), indexing='ij')
            circle = ((YY - d.c[0])**2 + (XX - d.c[1])**2) <= (d.r*d.r)
            mask = circle & (subE >= mu_e)
            if not np.any(mask):
                continue
            num = float((subA[mask] * subE[mask]).sum())
            den = float(d.energy) if d.energy > 0.0 else max(float(subE[mask].sum()), 1e-12)
            s = num / max(den, 1e-12)
            if s >= mu_d:
                code[d.bit_index] = True
                active.append(i)
        return code, active

# ===================== БАЗОВЫЙ kNN НА БЛОКАХ =====================

def popcount_u64_blocks(arr_u64: np.ndarray) -> np.ndarray:
    v = arr_u64.view(np.uint8)
    return LUT8[v].sum(axis=-1, dtype=np.int32)

def jaccard_blocks(query_blocks: np.ndarray, base_blocks: np.ndarray,
                   base_popcnt: np.ndarray | None = None) -> np.ndarray:
    qa = popcount_u64_blocks(query_blocks).sum(dtype=np.int32)
    if base_popcnt is None:
        bb = popcount_u64_blocks(base_blocks).reshape(base_blocks.shape[0], -1).sum(axis=1, dtype=np.int32)
    else:
        bb = base_popcnt
    inter_blocks = np.bitwise_and(base_blocks, query_blocks)
    inter = popcount_u64_blocks(inter_blocks).reshape(base_blocks.shape[0], -1).sum(axis=1, dtype=np.int32)
    union = qa + bb - inter
    union = np.maximum(union, 1)
    return inter / union

def predict_label_knn(code_blocks: np.ndarray, train_codes: np.ndarray,
                      train_labels: np.ndarray, train_pop: np.ndarray) -> int:
    sims = jaccard_blocks(code_blocks, train_codes, base_popcnt=train_pop)
    idx = np.argpartition(sims, -KNN_K)[-KNN_K:]
    neigh = train_labels[idx]
    vals, counts = np.unique(neigh, return_counts=True)
    y = vals[np.argmax(counts)]
    if np.sum(counts == counts.max()) > 1:
        y = train_labels[idx[np.argmax(sims[idx])]]
    return int(y)

# ===================== ОСНОВНОЙ СЦЕНАРИЙ =====================

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    train_ds = datasets.MNIST(DATA_DIR, train=True,  transform=ToTensor(), download=True)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, transform=ToTensor(), download=True)

    if TRAIN_LIMIT is None: trn_limit = len(train_ds)
    else: trn_limit = min(TRAIN_LIMIT, len(train_ds))

    N = trn_limit
    B = GRID * GRID

    train_imgs = np.empty((N, 28, 28), dtype=np.float32)
    train_lbls = np.empty((N,), dtype=np.int16)
    for i in tqdm(range(N), desc="Load train to NumPy"):
        img, y = train_ds[i]
        train_imgs[i] = img.squeeze(0).numpy()
        train_lbls[i] = int(y)

    T = len(test_ds)
    test_imgs = np.empty((T, 28, 28), dtype=np.float32)
    test_lbls = np.empty((T,), dtype=np.int16)
    for i in tqdm(range(T), desc="Load test to NumPy"):
        img, y = test_ds[i]
        test_imgs[i] = img.squeeze(0).numpy()
        test_lbls[i] = int(y)

    # ----- Часть A: k-NN на блоках -----
    train_codes  = np.empty((N, B, 2), dtype=np.uint64)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = []
        for s in range(0, N, CHUNK_ENCODE):
            e = min(s + CHUNK_ENCODE, N)
            for i in range(s, e):
                futs.append(ex.submit(lambda j: (j, encode_image_blocks(train_imgs[j])), i))
        for f in tqdm(as_completed(futs), total=len(futs), desc="Encode train (parallel)"):
            i, code = f.result(); train_codes[i] = code
    train_pop = popcount_u64_blocks(train_codes).reshape(N, -1).sum(axis=1, dtype=np.int32)

    preds_knn = np.empty((T,), dtype=np.int16)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = []
        for s in range(0, T, CHUNK_TEST):
            e = min(s + CHUNK_TEST, T)
            for i in range(s, e):
                futs.append(ex.submit(lambda j: (j, predict_label_knn(encode_image_blocks(test_imgs[j]), train_codes, train_lbls, train_pop)), i))
        for f in tqdm(as_completed(futs), total=len(futs), desc="Eval k-NN (parallel)"):
            i, p = f.result(); preds_knn[i] = p
    acc_knn = (preds_knn == test_lbls).mean()
    print(f"[A] Accuracy k-NN on blocks (Jaccard, parallel): {acc_knn:.4f}")

    # ----- Часть B: DAMP + детекторы + класс‑память -----
    # 1) Прототипы: случайная выборка P = DAMP_H*DAMP_W обучающих кодов
    P = DAMP_H * DAMP_W
    proto_idx = rng.choice(N, size=P, replace=False)
    proto_codes64 = np.empty((P, NBLK64), dtype=np.uint64)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = []
        for j, i in enumerate(proto_idx):
            futs.append(ex.submit(lambda jj, ii: (jj, blocks_to_u64bits(encode_image_blocks(train_imgs[ii]))), j, i))
        for f in tqdm(as_completed(futs), total=len(futs), desc="Encode prototypes"):
            j, v = f.result(); proto_codes64[j] = v
    proto_pops = popcount_u64_vec(proto_codes64)

    # 2) DAMP‑раскладка прототипов
    damp = DAMPLayoutPacked(codes64=proto_codes64, pops=proto_pops, H=DAMP_H, W=DAMP_W,
                            lam_far=LAM_FAR, lam_near=LAM_NEAR, eta=ETA, r_energy=R_ENERGY, pair_radius=PAIR_RADIUS)
    damp.run(steps_far=8, steps_near=8, p_per_step=16384, min_near_steps=2)

    # 3) Пространство детекторов «по документу»
    space = DetectorSpace(layout=damp, out_bits=DETECT_K)
    space.build_level(lam_d=LAM_D, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                      mu_e=MU_E_BUILD, attempts=DETECT_ATTEMPTS, max_detectors=DETECT_K,
                      seeds_k=SEEDS_TOPK, iou_thr=IOU_THR)
    print("[B] Detectors built:", len(space.detectors))

    # 4) Класс‑память: прогоняем train через детекторы и аккумулируем
    counts = np.zeros((10, DETECT_K), dtype=np.int32)
    def detect_one(img28: np.ndarray) -> np.ndarray:
        q64 = blocks_to_u64bits(encode_image_blocks(img28))
        code, _ = space.detect_from_code(q64, lam_a=LAM_D, mu_e=MU_E_DETECT, mu_d=MU_D)
        return code

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = []
        for s in range(0, N, CHUNK_CLASS):
            e = min(s + CHUNK_CLASS, N)
            for i in range(s, e):
                futs.append(ex.submit(lambda j: (j, detect_one(train_imgs[j])), i))
        for f in tqdm(as_completed(futs), total=len(futs), desc="Class memory (reduce)"):
            i, code = f.result()
            counts[int(train_lbls[i])] += code.astype(np.int32)

    # only-active-bits top‑k
    active_mask = counts.sum(axis=0) > 0
    active_idx = np.where(active_mask)[0]
    num_active = int(active_idx.size)
    k_on = max(1, int(round(TARGET_DENSITY * max(1, num_active))))
    class_hv = np.zeros((10, DETECT_K), dtype=bool)
    for cls in range(10):
        if num_active == 0:
            continue
        cls_counts_active = counts[cls, active_idx]
        if k_on >= num_active:
            sel_rel = np.arange(num_active)
        else:
            sel_rel = np.argpartition(cls_counts_active, -k_on)[-k_on:]
        sel = active_idx[sel_rel]
        class_hv[cls, sel] = True

    # 5) Инференс по детекторному коду
    preds_cls = np.empty((T,), dtype=np.int16)
    for s in tqdm(range(0, T, CHUNK_TEST), desc="Eval class-memory"):
        e = min(s + CHUNK_TEST, T)
        for i in range(s, e):
            q64 = blocks_to_u64bits(encode_image_blocks(test_imgs[i]))
            code, _ = space.detect_from_code(q64, lam_a=LAM_D, mu_e=MU_E_DETECT, mu_d=MU_D)
            # Жаккар к класс‑векторам
            best, arg = -1.0, 0
            for c in range(10):
                inter = np.count_nonzero(code & class_hv[c])
                uni = np.count_nonzero(code | class_hv[c])
                sim = 0.0 if uni == 0 else inter/uni
                if sim > best: best, arg = sim, c
            preds_cls[i] = arg
    acc_cls = (preds_cls == test_lbls).mean()
    print(f"[B] Accuracy class-memory (DAMP detectors): {acc_cls:.4f}")

    # ----- Быстрый self-check: распределение числа активных битов -----
    sample_n = min(256, T)
    bits_on = []
    for i in range(sample_n):
        q64 = blocks_to_u64bits(encode_image_blocks(test_imgs[i]))
        code, _ = space.detect_from_code(q64, lam_a=LAM_D, mu_e=MU_E_DETECT, mu_d=MU_D)
        bits_on.append(int(code.sum()))
    bits_on = np.array(bits_on)
    zero_frac = float((bits_on == 0).mean())
    print(f"[Check] bits_on over {sample_n} tests — min/med/max: {bits_on.min()} / {np.median(bits_on)} / {bits_on.max()} | zero_frac={zero_frac:.2%}")

    # ----- Сохранение -----
    np.savez_compressed(
        OUT_NPZ,
        train_codes=train_codes,
        train_labels=train_lbls,
        train_pop=train_pop,
        proto_codes64=proto_codes64,
        proto_pops=proto_pops,
        damp_grid=damp.grid_idx.astype(np.int32),
        detectors=np.array([(d.c[0], d.c[1], d.r, d.lam, d.n_points, d.energy, d.bit_index) for d in space.detectors], dtype=np.float32),
        class_hv=class_hv.astype(np.uint8),
        counts=counts.astype(np.int32),
    )
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump({
            "GRID": GRID,
            "LEVELS": LEVELS,
            "BITS_PER_CELL": BITS_PER_CELL,
            "K_BITS_PER_LEVEL": K_BITS_PER_LEVEL,
            "KNN_K": KNN_K,
            "DAMP": {
                "H": DAMP_H, "W": DAMP_W,
                "LAM_FAR": LAM_FAR, "LAM_NEAR": LAM_NEAR, "ETA": ETA,
                "R_ENERGY": R_ENERGY, "PAIR_RADIUS": PAIR_RADIUS
            },
            "DETECTORS": {
                "DETECT_K": DETECT_K, "LAM_D": LAM_D,
                "MU_E_BUILD": MU_E_BUILD, "MU_E_DETECT": MU_E_DETECT, "MU_D": MU_D,
                "DBSCAN_EPS": DBSCAN_EPS, "DBSCAN_MIN_SAMPLES": DBSCAN_MIN_SAMPLES,
                "ATTEMPTS": DETECT_ATTEMPTS, "SEEDS_TOPK": SEEDS_TOPK, "IOU_THR": IOU_THR
            },
            "TARGET_DENSITY": TARGET_DENSITY,
            "SEED": SEED,
            "parallel": {
                "workers": WORKERS,
                "chunk_encode": CHUNK_ENCODE,
                "chunk_test": CHUNK_TEST,
                "chunk_class": CHUNK_CLASS
            },
            "pipeline": [
                "28x28 -> 7x7 (avgpool + quantize)",
                "population coding (49 x 128 const‑weight) + concatenation",
                "baseline: Jaccard + k‑NN on blocks",
                "DAMP over P=H*W prototypes (packed Jaccard)",
                "detector space (DBSCAN level, normalized detector energy, IoU‑NMS)",
                "class-memory (top‑k only active bits)",
                "inference by Jaccard to class vectors"
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {OUT_NPZ}, {OUT_META}")


if __name__ == "__main__":
    main()
