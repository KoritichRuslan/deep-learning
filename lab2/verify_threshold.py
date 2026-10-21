"""Незалежна перевірка пошуку порога на трьох малих наборах (A, B, C) з умови ЛР2.

Еталон - повільний прямий перебір чистим Python: для кожного кандидата правило s >= t
застосовується до кожного об'єкта окремо. Результати швидкого алгоритму (threshold.py) для
побудови еталона не використовуються. Додатково є ручні очікувані значення.

"""
from __future__ import annotations

import math
import sys

import numpy as np

from threshold import best_threshold, candidate_table

SETS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0]),
}
# Очікувані (t*, C(t*)), виведені вручну; незалежні від обох реалізацій.
EXPECTED = {"A": (0.4, 2), "B": (math.inf, 0), "C": (0.8, 10)}


def brute_force(scores: list[float], labels: list[int]):
    """Повільний еталон. Повертає (рядки [(t,TP,FP,FN,TN,C)], індекс обраного рядка)."""
    # Кандидати: унікальні оцінки (за спаданням) та +inf; побудовано без numpy.
    candidates = [math.inf] + sorted(set(scores), reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for s_i, y_i in zip(scores, labels):
            predicted_positive = s_i >= t
            if predicted_positive and y_i == 1:
                tp += 1
            elif predicted_positive and y_i == 0:
                fp += 1
            elif (not predicted_positive) and y_i == 1:
                fn += 1
            else:
                tn += 1
        rows.append((t, tp, fp, fn, tn, 10 * fn + fp))
    best = 0
    for i, row in enumerate(rows):
        # строго менша вартість, або рівна вартість при більшому порозі
        if row[5] < rows[best][5] or (row[5] == rows[best][5] and row[0] > rows[best][0]):
            best = i
    return rows, best


def fmt_t(t: float) -> str:
    return "+inf" if math.isinf(t) else f"{t:g}"


def run_verification() -> tuple[bool, str]:
    """Повертає (усе_пройдено, markdown-звіт)."""
    ok_all = True
    lines: list[str] = []
    for name, (scores, labels) in SETS.items():
        ref_rows, ref_best = brute_force(scores, labels)
        fast = candidate_table(np.array(labels), np.array(scores))
        fast_rows = list(zip(fast["threshold"].tolist(), fast["TP"].tolist(), fast["FP"].tolist(),
                             fast["FN"].tolist(), fast["TN"].tolist(), fast["C"].tolist()))
        fast_best = best_threshold(fast)

        same_rows = len(ref_rows) == len(fast_rows) and all(
            a == b for a, b in zip(ref_rows, fast_rows)  # точний збіг порогів, лічильників і C
        )
        same_choice = ref_rows[ref_best][0] == fast_rows[fast_best][0]
        exp_t, exp_c = EXPECTED[name]
        matches_hand = ref_rows[ref_best][0] == exp_t and ref_rows[ref_best][5] == exp_c
        ok = same_rows and same_choice and matches_hand
        ok_all &= ok

        lines.append(f"### Набір {name}\n")
        lines.append("| Поріг | TP | FP | FN | TN | C | брутфорс = швидкий |")
        lines.append("|---|---|---|---|---|---|---|")
        for r_ref, r_fast in zip(ref_rows, fast_rows):
            lines.append(f"| {fmt_t(r_ref[0])} | {r_ref[1]} | {r_ref[2]} | {r_ref[3]} | {r_ref[4]} | "
                         f"{r_ref[5]} | {'так' if r_ref == r_fast else 'НІ'} |")
        lines.append("")
        lines.append(f"Обраний поріг: брутфорс {fmt_t(ref_rows[ref_best][0])} (C={ref_rows[ref_best][5]}), "
                     f"швидкий {fmt_t(fast_rows[fast_best][0])} (C={fast_rows[fast_best][5]}), "
                     f"ручне очікування {fmt_t(exp_t)} (C={exp_c}). "
                     f"Перевірку {'пройдено' if ok else 'НЕ ПРОЙДЕНО'}.\n")
    lines.append(f"**Усі три перевірки: {'пройдено' if ok_all else 'НЕ ПРОЙДЕНО'}.**")
    return ok_all, "\n".join(lines)


if __name__ == "__main__":
    passed, report = run_verification()
    print(report)
    sys.exit(0 if passed else 1)
