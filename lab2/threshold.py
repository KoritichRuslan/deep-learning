"""Підрахунок TP/FP/FN/TN, вартості C та точний пошук порога за O(n log n).

Правило рішення скрізь одне: прогноз "шахрайство", якщо s >= t.
Оцінки обробляються як float64 (конвертація float32 -> float64 точна).
"""
from __future__ import annotations

import numpy as np

C_FN = 10
C_FP = 1
THEORY_THRESHOLD = C_FP / (C_FP + C_FN)


def cost(fn, fp):
    """C = 10*FN + FP (Amount у формулу не входить)."""
    return C_FN * fn + C_FP * fp


def confusion(y: np.ndarray, s: np.ndarray, t: float) -> dict[str, int]:
    """Матриця плутанини для одного порога за правилом s >= t (t може бути +inf)."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    pred = s >= t
    pos = y == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    tn = int(np.count_nonzero(~pred & ~pos))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": int(cost(fn, fp))}


def precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """(precision, recall); NaN, якщо знаменник дорівнює нулю (метрика не визначена)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, recall


def candidate_table(y: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray]:
    """Повна таблиця кандидатів через сортування й кумулятивні суми."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y і s мають бути одновимірними масивами однакової довжини")
    if not np.isfinite(s).all():
        raise ValueError("Оцінки мають бути скінченними")
    n = len(y)
    n_pos = int(np.count_nonzero(y == 1))
    n_neg = n - n_pos

    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    pos_sorted = (y[order] == 1).astype(np.int64)
    ctp = np.cumsum(pos_sorted)
    cfp = np.cumsum(1 - pos_sorted)

    is_group_end = np.ones(n, dtype=bool)
    is_group_end[:-1] = s_sorted[:-1] != s_sorted[1:]
    thr = s_sorted[is_group_end]
    tp = ctp[is_group_end]
    fp = cfp[is_group_end]

    thr = np.concatenate(([np.inf], thr))
    tp = np.concatenate(([0], tp))
    fp = np.concatenate(([0], fp))
    fn = n_pos - tp
    tn = n_neg - fp
    return {"threshold": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}


def best_threshold(table: dict[str, np.ndarray]) -> int:
    """Індекс рядка з мінімальною C; за рівності - з НАЙБІЛЬШИМ порогом.

    Рядки впорядковані за спаданням порога, а np.argmin повертає перший мінімум,
    тобто саме найбільший поріг серед рівних за вартістю.
    """
    return int(np.argmin(table["C"]))
