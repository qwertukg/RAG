"""
Реализация ключевых алгоритмов из документа
«Дискретный подход к машинному обучению» (rev. _22_) —
в виде самодостаточного Python‑модуля без внешних зависимостей,
кроме numpy и tqdm.

Содержимое:
  • Разреженные битовые коды и операции (гл. 2):
      - популяционное кодирование (2.0.1)
      - объединение/пересечение/конкатенация (2.2.1–2.2.3)
      - меры близости: Жаккар, дискретный косинус (2.2.4)
      - нечеткий поиск k‑NN по Жаккару (2.2.5)
      - «цветовое» объединение (гл. 3.3)
  • Алгоритм раскладки DAMP (гл. 5):
      - энергии точек (5.7)
      - «дальний» и «ближний» варианты энергий пар и обмены (5.5–5.6)
      - пороговая близость с отсечкой sim_λ и плавной «ранней отсечкой» (5.2, 5.4, 5.9.1)
  • Детекторы поверх разложенного пространства (гл. 6):
      - активация пространства кодом/стимулами (6.1, 6.5)
      - построение пространства детекторов (6.4): кластеризация по плотности (минимальная DBSCAN‑реализация),
        центр кластера как взвешенный центроид (6.4.2), оптимальный радиус по критерию заполнения (6.4.3),
        вставка детектора с проверкой неперекрытия и «коэффициента заполнения» (6.4.5)
      - вычисление энергии детектора и формирование выходного кода b_d (6.3)

Примечание.
Документ допускает как битовые, так и нормализованные вещественные коды (см. 2.0.1).
Ниже поддержаны оба типа: бинарные (bool/uint8) и вещественные (float32) векторы.

Авторский акцент — на точном следовании формулам и обозначениям документа; детали,
не конкретизированные в тексте (например, выбор стохастики при выборе пар, или
параметры эвристик), реализованы минимально‑достаточными способами с явными
настройками.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, List, Tuple, Optional, Callable, Sequence

import numpy as np
try:
    from tqdm import tqdm
except Exception:  # tqdm не обязателен
    def tqdm(x, **kwargs):
        return x

# ============================
# Раздел 2: коды и операции
# ============================

# --- утилиты битовых векторов ---

def popcount_uint8(arr: np.ndarray) -> np.ndarray:
    """Подсчет единиц в последнем измерении для uint8/bool массива.
    Возвращает int32.
    """
    if arr.dtype == np.bool_:
        return arr.sum(axis=-1, dtype=np.int32)
    v = np.unpackbits(arr, axis=-1)
    return v.sum(axis=-1, dtype=np.int32)


def jaccard_bool(a: np.ndarray, b: np.ndarray) -> float:
    """Коэффициент Жаккара для двух булевых векторов (2.2.4.2).
    a, b: 1D bool arrays.
    """
    inter = np.count_nonzero(a & b)
    uni = np.count_nonzero(a | b)
    return 0.0 if uni == 0 else inter / uni


def cosine_discrete(a: np.ndarray, b: np.ndarray) -> float:
    """Дискретный аналог косинусной меры (2.2.4): C(a,b) = |a∧b|/sqrt(|a|·|b|).
    a, b: 1D bool arrays.
    """
    pa = np.count_nonzero(a)
    pb = np.count_nonzero(b)
    if pa == 0 or pb == 0:
        return 0.0
    inter = np.count_nonzero(a & b)
    return inter / math.sqrt(pa * pb)


# --- Популяционное кодирование (2.0.1) ---

@dataclass
class PopulationCodebook:
    bits: int                 # длина кода
    k: int                    # константный вес
    rng: np.random.Generator = np.random.default_rng(42)

    def sample_code(self) -> np.ndarray:
        idx = self.rng.choice(self.bits, size=self.k, replace=False)
        v = np.zeros(self.bits, dtype=bool)
        v[idx] = True
        return v

    def sample_many(self, n: int) -> np.ndarray:
        out = np.zeros((n, self.bits), dtype=bool)
        for i in range(n):
            out[i] = self.sample_code()
        return out


# --- Операции (2.2) ---

def union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Побитовое объединение (2.2.1)."""
    return a | b


def intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Побитовое пересечение (2.2.2)."""
    return a & b


def concatenate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Конкатенация (2.2.3)."""
    return np.concatenate([a, b], axis=-1)


# --- «Цветовое» объединение (гл. 3.3) ---

@dataclass
class ChromaticUnion:
    threshold_t: int                 # максимум единиц в результирующем коде
    colors: Optional[np.ndarray] = None  # приоритеты битов в [0,1], той же длины
    prefer: str = "far"              # "far" (дальний порядок) или "near" (ближний)

    def __post_init__(self):
        if self.prefer not in ("far", "near"):
            raise ValueError("prefer должен быть 'far' или 'near'")

    def fuse(self, *codes: np.ndarray) -> np.ndarray:
        """Цветовое объединение 3.3: если |∨ codes| ≤ t — обычное OR;
        иначе — отбор по приоритету цветов (3.3.2).
        """
        total = np.logical_or.reduce(codes)
        cnt = int(total.sum())
        if cnt <= self.threshold_t or self.colors is None:
            return total
        # Отбор по цветам: сортировка по приоритету (направление зависит от порядка)
        on_idx = np.flatnonzero(total)
        weights = self.colors[on_idx]
        # дальний/ближний порядок — это выбор конца спектра (3.3.1–3.3.2)
        rev = (self.prefer == "near")
        sel = on_idx[np.argsort(weights, kind="mergesort")]
        if rev:
            sel = sel[::-1]
        keep = sel[: self.threshold_t]
        out = np.zeros_like(total)
        out[keep] = True
        return out


# --- Близость с отсечкой (5.4) ---

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sim_lambda(base_sim: np.ndarray, lam: float, eta: float = 12.0) -> np.ndarray:
    """Пороговая близость sim_λ(a,b) = τ(S(a,b)), где τ(x)=x·σ(η·(x−λ)).
    Векторная форма: base_sim в [0,1]. См. форм. 5.4 (определение τ).
    """
    return base_sim * sigmoid(eta * (base_sim - lam))


# ============================
# Раздел 5: раскладка DAMP
# ============================

@dataclass
class DAMPLayout:
    """Алгоритм раскладки (гл. 5). Размещает коды V на 2D‑сетке/плоскости.

    Параметры:
      codes: np.ndarray формы [N, D] — бинарные (bool) или вещественные (float32) коды понятий V.
      H, W: размерность 2D‑решетки для раскладки (можно задать число точек n=H*W == N).
      lam_far, lam_near: пороги λ для дальнего и ближнего этапов.
      eta: крутизна сигмоиды в τ (5.4).
      r_energy: радиус для энергий точек (5.7).
      pair_radius: локальный радиус для подбора пар (5.9.4). 0 => весь диапазон.
      metric: Callable(a,b)->float в [0,1] (Жаккар по умолчанию).
    """
    codes: np.ndarray
    H: int
    W: int
    lam_far: float = 0.6
    lam_near: float = 0.6
    eta: float = 12.0
    r_energy: float = 8.0
    pair_radius: float = 12.0
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    rng: np.random.Generator = np.random.default_rng(42)

    def __post_init__(self):
        self.N = self.codes.shape[0]
        assert self.H * self.W == self.N, "H*W должно равняться числу кодов"
        if self.metric is None:
            # по умолчанию Жаккар для бинарных, cos для float
            if self.codes.dtype == np.bool_:
                self.metric = lambda a, b: jaccard_bool(a, b)
            else:
                self.metric = lambda a, b: float(np.dot(a, b) / (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum()) + 1e-9))
        # начальная раскладка — случайная перестановка
        self.grid_idx = self.rng.permutation(self.N).reshape(self.H, self.W)
        # Предвычислим матрицу базовой близости (при больших N используйте блоками)
        self._sim = None

    # --- служебные ---
    def _ensure_sim(self):
        if self._sim is not None:
            return
        N = self.N
        self._sim = np.zeros((N, N), dtype=np.float32)
        for i in tqdm(range(N), desc="pairwise sim"):
            a = self.codes[i]
            for j in range(i, N):
                s = self.metric(a, self.codes[j])
                self._sim[i, j] = self._sim[j, i] = s

    def coords_of(self, idx: int) -> Tuple[int, int]:
        y, x = np.argwhere(self.grid_idx == idx)[0]
        return int(y), int(x)

    def _local_window(self, cy: int, cx: int, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ys = np.arange(max(0, int(cy - r)), min(self.H, int(cy + r) + 1))
        xs = np.arange(max(0, int(cx - r)), min(self.W, int(cx + r) + 1))
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        dy = (Y - cy).astype(np.float32)
        dx = (X - cx).astype(np.float32)
        D = np.sqrt(dy * dy + dx * dx)
        mask = D <= r
        return Y[mask], X[mask], D[mask]

    # --- Энергия точки (5.7) ---
    def point_energy(self, idx: int, r: Optional[float] = None, lam: Optional[float] = None) -> float:
        self._ensure_sim()
        if r is None:
            r = self.r_energy
        if lam is None:
            lam = self.lam_far
        cy, cx = self.coords_of(idx)
        Y, X, D = self._local_window(cy, cx, r)
        # Коды точек в окрестности
        neigh_idx = self.grid_idx[Y, X].ravel()
        base = self._sim[idx, neigh_idx]
        base = np.clip(base, 0.0, 1.0)
        s = sim_lambda(base, lam, self.eta)
        D = np.maximum(D, 1e-6)
        E = (s / D).sum()
        return float(E)

    # --- Энергия пары (5.5) ---
    def _pair_energy(self, i1: int, i2: int, mode: str = "far") -> Tuple[float, float]:
        """Возвращает (phi_c, phi_s) для пары индексов кодов i1,i2.
        mode="far": φ = sum s*d (штраф далеких коррелированных)
        mode="near": φ = sum s/d (поощрение близких коррелированных)
        """
        self._ensure_sim()
        y1, x1 = self.coords_of(i1)
        y2, x2 = self.coords_of(i2)
        r = self.pair_radius if self.pair_radius > 0 else max(self.H, self.W)
        Y1, X1, D1 = self._local_window(y1, x1, r)
        Y2, X2, D2 = self._local_window(y2, x2, r)
        idx1 = self.grid_idx[Y1, X1].ravel()
        idx2 = self.grid_idx[Y2, X2].ravel()
        s1 = sim_lambda(self._sim[i1, idx1], self.lam_far if mode=="far" else self.lam_near, self.eta)
        s2 = sim_lambda(self._sim[i2, idx2], self.lam_far if mode=="far" else self.lam_near, self.eta)
        D1 = np.maximum(D1.ravel(), 1e-6)
        D2 = np.maximum(D2.ravel(), 1e-6)
        if mode == "far":
            phi_c = float((s1 * D1).sum() + (s2 * D2).sum())
            phi_s = float((s2 * D1).sum() + (s1 * D2).sum())
        else:
            phi_c = float((s1 / D1).sum() + (s2 / D2).sum())
            phi_s = float((s2 / D1).sum() + (s1 / D2).sum())
        return phi_c, phi_s

    def _random_pairs(self, p: int) -> List[Tuple[int, int]]:
        pairs = []
        for _ in range(p):
            a, b = self.rng.integers(0, self.N, size=2)
            while b == a:
                b = self.rng.integers(0, self.N)
            pairs.append((int(a), int(b)))
        return pairs

    def step(self, p: int = 256, mode: str = "far") -> int:
        """Один шаг раскладки: протестировать p пар и выполнить обмены (5.6).
        Возвращает число произведённых обменов.
        """
        swapped = 0
        pairs = self._random_pairs(p)
        for (i1, i2) in pairs:
            # ранняя отсечка по базовой близости (5.9.1)
            # если коды почти не связаны, обмен бессмысленен
            if self._sim is None:
                self._ensure_sim()
            # лёгкая эвристика: требуем хотя бы одну заметную связь внутри радиуса
            phi_c, phi_s = self._pair_energy(i1, i2, mode=mode)
            better = (phi_s < phi_c) if mode == "far" else (phi_s > phi_c)
            if better:
                # обмен координат местами
                y1, x1 = self.coords_of(i1)
                y2, x2 = self.coords_of(i2)
                self.grid_idx[y1, x1], self.grid_idx[y2, x2] = self.grid_idx[y2, x2], self.grid_idx[y1, x1]
                swapped += 1
        return swapped

    def run(self, steps_far: int = 20, steps_near: int = 20, p_per_step: int = 2048) -> None:
        """Запуск раскладки: дальний этап (минимизация) + ближний (максимизация)."""
        for _ in tqdm(range(steps_far), desc="DAMP far"):
            n = self.step(p=p_per_step, mode="far")
            if n == 0:
                break
        for _ in tqdm(range(steps_near), desc="DAMP near"):
            n = self.step(p=p_per_step, mode="near")
            if n == 0:
                break


# ============================
# Раздел 6: детекторы
# ============================

@dataclass
class Detector:
    c: Tuple[float, float]  # (cy, cx)
    r: float                # радиус r_d
    lam: float              # λ_d
    n_points: int           # n_d (число точек с e>=μ внутри поля при создании)
    energy: float           # e_d
    bit_index: int          # случайный выходной бит b_d


class SimpleDBSCAN:
    """Минимальная реализация DBSCAN для кластера активированных точек (6.4.1).
    Работает на np.ndarray точек формы [M,2].
    """
    def __init__(self, eps: float = 2.0, min_samples: int = 4):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, P: np.ndarray) -> np.ndarray:
        if len(P) == 0:
            return np.empty((0,), dtype=np.int32)
        M = P.shape[0]
        labels = -np.ones(M, dtype=np.int32)
        visited = np.zeros(M, dtype=bool)
        cluster_id = 0
        # предвычислим расстояния
        # для умеренных M; для больших — блокировать
        D = np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(axis=-1))
        for i in range(M):
            if visited[i]:
                continue
            visited[i] = True
            neigh = np.where(D[i] <= self.eps)[0]
            if neigh.size < self.min_samples:
                labels[i] = -1
                continue
            # новый кластер
            labels[i] = cluster_id
            seeds = list(neigh)
            j = 0
            while j < len(seeds):
                q = seeds[j]
                if not visited[q]:
                    visited[q] = True
                    nq = np.where(D[q] <= self.eps)[0]
                    if nq.size >= self.min_samples:
                        for u in nq:
                            if u not in seeds:
                                seeds.append(int(u))
                if labels[q] == -1:
                    labels[q] = cluster_id
                j += 1
            cluster_id += 1
        return labels


@dataclass
class DetectorSpace:
    """Построение и применение пространства детекторов (гл. 6)."""
    # Разложенное кодовое пространство — сетка HxW точек, каждая точка u имеет:
    #   code index in [0..N-1]
    layout: DAMPLayout
    E_norm: Optional[np.ndarray] = None   # матрица нормализованных энергий Ê (5.7)
    detectors: List[Detector] = None
    out_bits: int = 512
    rng: np.random.Generator = np.random.default_rng(123)

    def __post_init__(self):
        if self.detectors is None:
            self.detectors = []
        if self.E_norm is None:
            # рассчитать Ê по всему пространству
            E = np.zeros((self.layout.H, self.layout.W), dtype=np.float32)
            for idx in range(self.layout.N):
                E.flat[idx] = self.layout.point_energy(idx, r=self.layout.r_energy, lam=self.layout.lam_near)
            Emax = E.max() if E.size else 1.0
            self.E_norm = (E / max(Emax, 1e-9)).astype(np.float32)

    # --- активация пространства кодом/стимулами (6.1, 6.5) ---
    def activation(self, stimuli_idx: Sequence[int], lam_a: float = 0.6) -> np.ndarray:
        """A ≡ A^λ_a(S): объединённая активация по множеству стимулов S (6.5).
        Для каждой точки v_yx: a_yx = max_{s∈S} sim_λ(s, v_yx).
        Возвращает матрицу A формы [H,W].
        """
        self.layout._ensure_sim()
        H, W = self.layout.H, self.layout.W
        A = np.zeros((H, W), dtype=np.float32)
        for s in stimuli_idx:
            # близости от стимула ко всем точкам
            base = self.layout._sim[s]  # [N]
            a = sim_lambda(base, lam=lam_a, eta=self.layout.eta)
            A = np.maximum(A, a.reshape(H, W))
        return A

    # --- построение пространства детекторов (6.4) ---
    def _cluster_points(self, A_loc: np.ndarray, mu_e: float, eps: float, min_samples: int) -> List[np.ndarray]:
        """Выбрать активные точки с Ê>=μ_e и кластеризовать DBSCAN (6.4.1)."""
        H, W = A_loc.shape
        Ys, Xs = np.where(self.E_norm >= mu_e)
        if Ys.size == 0:
            return []
        P = np.stack([Ys.astype(np.float32), Xs.astype(np.float32)], axis=1)
        labels = SimpleDBSCAN(eps=eps, min_samples=min_samples).fit_predict(P)
        clusters = []
        for cid in sorted(set(labels.tolist())):
            if cid < 0:
                continue
            clusters.append(P[labels == cid])
        return clusters

    def _centroid(self, P: np.ndarray, A: np.ndarray) -> Tuple[float, float]:
        """Центроид кластера по (6.4.2): c_d = Σ p_i·w_i / Σ w_i, w_i = A^λ_d · Ê."""
        # извлечём веса для точек P из матрицы A и Ê
        ys = np.clip(P[:, 0].astype(int), 0, self.layout.H-1)
        xs = np.clip(P[:, 1].astype(int), 0, self.layout.W-1)
        W = A[ys, xs] * self.E_norm[ys, xs]
        Wsum = float(W.sum())
        if Wsum <= 1e-12:
            # fallback — геом. центр
            return float(P[:,0].mean()), float(P[:,1].mean())
        cy = float((P[:, 0] * W).sum() / Wsum)
        cx = float((P[:, 1] * W).sum() / Wsum)
        return cy, cx

    def _optimal_radius(self, P: np.ndarray, c: Tuple[float, float]) -> float:
        """Оптимальный радиус по (6.4.3): argmax_p |{q∈P: r(q)≤r(p)}| / (π r(p)^2)."""
        cy, cx = c
        r_all = np.sqrt((P[:,0]-cy)**2 + (P[:,1]-cx)**2)
        order = np.argsort(r_all)
        best_r = 1.0
        best_val = -1.0
        cnt = 0
        for idx in order:
            r = max(float(r_all[idx]), 1e-6)
            cnt += 1
            val = cnt / (math.pi * r * r)
            if val > best_val:
                best_val = val
                best_r = r
        return float(best_r)

    def _can_insert(self, c: Tuple[float,float], r: float, lam: float, n_fill: float) -> Tuple[bool, List[int]]:
        """Проверка неперекрытия/замещения (6.4.5). Возвращает (ok, indices_to_remove)."""
        to_remove = []
        for i, d in enumerate(self.detectors):
            if abs(d.lam - lam) > 1e-9:
                continue
            dy = d.c[0]-c[0]
            dx = d.c[1]-c[1]
            dist = math.hypot(dy, dx)
            if dist <= r or dist <= d.r:
                # сравнить коэффициенты заполнения n/r
                if n_fill > (d.n_points / max(d.r, 1e-6)):
                    to_remove.append(i)
                else:
                    return (False, [])
        return (True, to_remove)

    def build_level(self, lam_d: float = 0.6, r_act: float = 8.0, eps: float = 2.0, min_samples: int = 4,
                    mu_e: float = 0.15, attempts: int = 200, max_detectors: Optional[int] = None) -> None:
        """Построить один «уровень» детекторов при фиксированном λ_d (6.4).
        lam_d — порог активации, r_act — радиус активации (≥ r_d), eps/min_samples — DBSCAN,
        mu_e — порог по норм. энергии точек Ê, attempts — число случайных центров c∈V для проб.
        """
        H, W = self.layout.H, self.layout.W
        for _ in tqdm(range(attempts), desc=f"detectors λ={lam_d:.2f}"):
            # случайный ориентирный центр c (6.4)
            cy = self.rng.uniform(0, H-1)
            cx = self.rng.uniform(0, W-1)
            # активация окрестности (локальная матрица A^λ_d(c, r_a))
            # реализуем через ближайший индекс стимулов: берём код в точке
            idx = int(self.layout.grid_idx[int(round(cy)) % H, int(round(cx)) % W])
            A_loc = self.activation([idx], lam_a=lam_d)
            # кластеризация активных точек согласно Ê и A (6.4.1)
            clusters = self._cluster_points(A_loc, mu_e=mu_e, eps=eps, min_samples=min_samples)
            if not clusters:
                continue
            for P in clusters:
                c_d = self._centroid(P, A_loc)
                r_d = self._optimal_radius(P, c_d)
                # n_d — число точек с энергиeй выше μ в поле детектора в момент создания
                ys = np.clip(P[:,0].astype(int), 0, H-1)
                xs = np.clip(P[:,1].astype(int), 0, W-1)
                n_d = int((self.E_norm[ys, xs] >= mu_e).sum())
                n_fill = n_d / max(r_d, 1e-6)
                ok, rem = self._can_insert(c_d, r_d, lam_d, n_fill)
                if not ok:
                    continue
                # энергия детектора e_d = Σ a·Ê (6.3)
                # ограничим суммирование рецептивным полем
                y0 = int(max(0, math.floor(c_d[0]-r_d)))
                y1 = int(min(H, math.ceil(c_d[0]+r_d)+1))
                x0 = int(max(0, math.floor(c_d[1]-r_d)))
                x1 = int(min(W, math.ceil(c_d[1]+r_d)+1))
                subA = A_loc[y0:y1, x0:x1]
                subE = self.E_norm[y0:y1, x0:x1]
                YY, XX = np.meshgrid(np.arange(y0,y1), np.arange(x0,x1), indexing='ij')
                mask = ((YY - c_d[0])**2 + (XX - c_d[1])**2) <= (r_d*r_d)
                e_d = float((subA[mask] * subE[mask]).sum())
                # выходной бит — случайный индекс
                bit = int(self.rng.integers(0, self.out_bits))
                # удалить перекрывающиеся детекторы уровня
                for j in sorted(rem, reverse=True):
                    self.detectors.pop(j)
                self.detectors.append(Detector(c=c_d, r=float(r_d), lam=float(lam_d),
                                               n_points=int(n_d), energy=e_d, bit_index=bit))
                if max_detectors is not None and len(self.detectors) >= max_detectors:
                    return

    # --- детектирование стимула и эмбеддинг (6.5) ---
    def detect(self, stimuli_idx: Sequence[int], lam_a: float = 0.6, mu_e: float = 0.2,
               energy_coeff: float = 0.0) -> Tuple[np.ndarray, List[int]]:
        """Вернуть бинарный код активности детекторов и список сработавших детекторов.
        A ≡ A^λ_a(S) — объединенная активация (6.5). Далее берём подмножество A_{μ_e}(d, A)
        внутри рецептивного поля и вычисляем e_d = Σ a·Ê (6.3). Детектор активен, если:
            e_d >= max(μ_e·n_d, q⋅E_ref), где q задаётся через energy_coeff (эвристич.)
        Возвращает: (code[bits], active_indices).
        """
        A = self.activation(stimuli_idx, lam_a=lam_a)
        code = np.zeros((self.out_bits,), dtype=bool)
        act_ids = []
        # опорный уровень для адаптивного порога
        E_ref = float((A * self.E_norm).mean())
        for i, d in enumerate(self.detectors):
            y0 = int(max(0, math.floor(d.c[0]-d.r)))
            y1 = int(min(self.layout.H, math.ceil(d.c[0]+d.r)+1))
            x0 = int(max(0, math.floor(d.c[1]-d.r)))
            x1 = int(min(self.layout.W, math.ceil(d.c[1]+d.r)+1))
            subA = A[y0:y1, x0:x1]
            subE = self.E_norm[y0:y1, x0:x1]
            YY, XX = np.meshgrid(np.arange(y0,y1), np.arange(x0,x1), indexing='ij')
            mask = ((YY - d.c[0])**2 + (XX - d.c[1])**2) <= (d.r*d.r)
            e_d = float((subA[mask] * subE[mask]).sum())
            thr = max(mu_e * d.n_points, energy_coeff * E_ref)
            if e_d >= thr:
                code[d.bit_index] = True
                act_ids.append(i)
        return code, act_ids


# ============================
# Нечеткий поиск (2.2.5)
# ============================

def knn_jaccard(query: np.ndarray, base: np.ndarray, labels: Optional[np.ndarray] = None,
                k: int = 7) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """k‑NN по Жаккару для булевых кодов. Возвращает индексы top‑k и (опционально) метки.
    query: [D], base: [N,D].
    """
    # векторизованный Жаккар
    inter = np.count_nonzero(base & query, axis=1)
    uni = np.count_nonzero(base | query, axis=1)
    sim = np.divide(inter, np.maximum(uni, 1), dtype=np.float32)
    idx = np.argpartition(sim, -k)[-k:]
    idx = idx[np.argsort(sim[idx])[::-1]]
    if labels is None:
        return idx, None
    return idx, labels[idx]


# ============================
# Пример использования (микро‑демо)
# ============================

if __name__ == "__main__":
    # 1) Сгенерируем небольшое пространство кодов (битовые коды) и выполним раскладку.
    N = 32 * 32
    codebook = PopulationCodebook(bits=256, k=12, rng=np.random.default_rng(1))
    V = codebook.sample_many(N)

    damp = DAMPLayout(codes=V, H=32, W=32, lam_far=0.6, lam_near=0.6, r_energy=6.0, pair_radius=8.0)
    # В целях демо уменьшим количество шагов
    damp.run(steps_far=2, steps_near=2, p_per_step=512)

    # 2) Построим один уровень детекторов
    space = DetectorSpace(layout=damp, out_bits=512)
    space.build_level(lam_d=0.6, r_act=8.0, eps=1.5, min_samples=3, mu_e=0.10, attempts=30, max_detectors=64)

    # 3) Возьмём произвольный «стимул» — один из кодов — и получим детекторный код
    stimulus_idx = [0]
    code, act = space.detect(stimulus_idx, lam_a=0.6, mu_e=0.1, energy_coeff=0.0)
    print("Active detectors:", len(act), "bits on:", int(code.sum()))
