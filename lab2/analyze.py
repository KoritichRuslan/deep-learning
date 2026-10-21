"""ЛР2, пункти 2-4: аналіз порогів зі ЗБЕРЕЖЕНИХ прогнозів.

    uv run python analyze.py --saved results

Порядок (важливий): validation -> зафіксувати t* -> перевірки на наборах A/B/C ->
тільки після їх проходження оцінка на test. Test не впливає на вибір порога.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

from lab2_starter import load_predictions, split_summary  # noqa: E402
from threshold import (THEORY_THRESHOLD, best_threshold, candidate_table, confusion,  # noqa: E402
                       cost, precision_recall)
from verify_threshold import run_verification  # noqa: E402

N_BOOTSTRAP = 200
N_SAMPLE_CHECK = 300
NA = "н/в"  # метрика не визначена (нульовий знаменник)


def ft(t: float) -> str:
    return "+inf" if math.isinf(t) else f"{t:.6g}"


def fm_g(x: float) -> str:
    return NA if math.isnan(x) else f"{x:.4g}"


def fm(x: float) -> str:
    return NA if (x is None or math.isnan(x)) else f"{x:.4f}"


def pct(x: float) -> str:
    return f"{100 * x:.3f}%"


def md_table(header: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def rule_row(label: str, t: float, y: np.ndarray, s: np.ndarray) -> tuple[list[object], dict]:
    c = confusion(y, s, t)
    p, r = precision_recall(c["TP"], c["FP"], c["FN"])
    row = [label, ft(t), c["TP"], c["FP"], c["FN"], c["TN"], c["C"], fm(p), fm(r)]
    return row, {**c, "precision": p, "recall": r}


RULE_HEADER = ["Правило", "Поріг", "TP", "FP", "FN", "TN", "C", "Precision", "Recall"]


def save_curves(y, s, ops: dict[str, float], path: Path) -> None:
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(fpr, tpr, color="C0")
    ax[0].plot([0, 1], [0, 1], ":", color="gray")
    ax[1].plot(rec, prec, color="C1")
    ax[1].axhline(np.mean(y), ls=":", color="gray", label=f"базова лінія π={pct(float(np.mean(y)))}")
    for (label, t), mk in zip(ops.items(), ("o", "s")):
        if math.isinf(t):
            continue
        c = confusion(y, s, t)
        p, r = precision_recall(c["TP"], c["FP"], c["FN"])
        ax[0].plot(c["FP"] / (c["FP"] + c["TN"]), r, mk, label=f"{label}: t={ft(t)}")
        ax[1].plot(r, p, mk, label=f"{label}: t={ft(t)}")
    ax[0].set(title=f"ROC (validation), AUC={roc_auc_score(y, s):.4f}", xlabel="FPR", ylabel="TPR")
    ax[1].set(title=f"Точність-повнота (validation), AP={average_precision_score(y, s):.4f}",
              xlabel="Повнота (recall)", ylabel="Точність (precision)")
    ax[0].legend(loc="lower right")
    ax[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_cost_curve(tab: dict, ops: dict[str, float], path: Path) -> None:
    thr = tab["threshold"][1:][::-1]   # зростання порога, без +inf
    cst = tab["C"][1:][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(thr, cst, where="post", color="C0")
    for label, t in ops.items():
        if not math.isinf(t):
            ax.axvline(t, ls="--", lw=1, label=f"{label}: {ft(t)}")
    ax.axvline(THEORY_THRESHOLD, ls=":", color="gray", label=f"1/11≈{THEORY_THRESHOLD:.4f} (орієнтир)")
    ax.set(xscale="log", yscale="log", xlabel="поріг t", ylabel="C = 10·FN + FP",
           title="Вартість на validation залежно від порога")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def bootstrap_thresholds(y: np.ndarray, s: np.ndarray, seed: int = 0) -> np.ndarray:
    """Стійкість t* до вибірки: бутстреп ЛИШЕ validation (test не використовується)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        tab = candidate_table(y[idx], s[idx])
        out[b] = tab["threshold"][best_threshold(tab)]
    return out


def delta_text(split: str, a: dict, b: dict) -> str:
    c0, c1 = a["C"], b["C"]
    rel = f"; зміна вартості {-100 * (c0 - c1) / c0:+.1f}%" if c0 > 0 else ""
    return (f"- {split}: перехід від t=0.5 до t* змінив FN {a['FN']} → {b['FN']} ({b['FN'] - a['FN']:+d}), "
            f"FP {a['FP']} → {b['FP']} ({b['FP'] - a['FP']:+d}), C {c0} → {c1} ({c1 - c0:+d}){rel}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saved", type=Path, required=True, help="Тека з predictions.npz (результат навчання)")
    ap.add_argument("--report-dir", type=Path, default=Path("analysis"), help="Куди писати результати аналізу")
    ap.add_argument("--readme", type=Path, default=Path("README.md"), help="README для автозаповнення таблиць")
    args = ap.parse_args()
    out_dir: Path = args.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    auto: dict[str, str] = {}

    saved = load_predictions(args.saved)
    y_va = saved["validation_y"].astype(np.int64)
    s_va = saved["validation_scores"].astype(np.float64)  # float32 -> float64 без втрат
    y_te = saved["test_y"].astype(np.int64)
    s_te = saved["test_scores"].astype(np.float64)
    for name, arr in (("validation", s_va), ("test", s_te)):
        if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
            print(f"ПОМИЛКА: оцінки {name} мають бути скінченними та лежати в [0, 1].")
            return 1

    # ---------- Пункт 1: розміри вибірок ----------
    summ = split_summary(saved)
    auto["split_table"] = md_table(
        ["Вибірка", "Об'єктів", "Шахрайських", "Частка шахрайських"],
        [[r.split, f"{r.n:,}".replace(",", " "), r.positives, pct(r.positive_fraction)] for r in summ.itertuples()])

    # ---------- Пункт 3: перевірка пошуку порога (до тесту!) ----------
    passed, ver_md = run_verification()
    (out_dir / "verification.md").write_text(ver_md + "\n", encoding="utf-8")
    auto["verification"] = ver_md
    print(ver_md, "\n")
    if not passed:
        print("ПЕРЕВІРКИ НА НАБОРАХ A/B/C НЕ ПРОЙДЕНО: до оцінки на test не переходимо.")
        return 1

    # ---------- Пункт 2: validation ----------
    tab = candidate_table(y_va, s_va)
    i_best = best_threshold(tab)
    t_star = float(tab["threshold"][i_best])
    pd.DataFrame(tab).to_csv(out_dir / "validation_candidates.csv", index=False,
                             columns=["threshold", "TP", "FP", "FN", "TN", "C"])

    # узгодженість із прямим підрахунком confusion() на вибірці порогів (не повний перебір)
    rng = np.random.default_rng(0)
    pick = set(rng.choice(len(tab["threshold"]), size=min(N_SAMPLE_CHECK, len(tab["threshold"])),
                          replace=False).tolist()) | {0, i_best, len(tab["threshold"]) - 1}
    for i in sorted(pick):
        c = confusion(y_va, s_va, float(tab["threshold"][i]))
        got = {k: int(tab[k][i]) for k in ("TP", "FP", "FN", "TN", "C")}
        if c != got:
            print(f"ПОМИЛКА: розбіжність швидкого пошуку й прямого підрахунку на рядку {i}: {got} vs {c}")
            return 1

    rows, info = [], {}
    for label, t in (("Стандартний поріг", 0.5), ("Мінімум вартості", t_star), ("Усі негативні", math.inf)):
        r, d = rule_row(label, t, y_va, s_va)
        rows.append(r)
        info[label] = d
    auto["val_table"] = md_table(RULE_HEADER, rows)
    ref_row, ref_info = rule_row("Орієнтир 1/11 (не обирався)", THEORY_THRESHOLD, y_va, s_va)

    auc = roc_auc_score(y_va, s_va)
    ap_score = average_precision_score(y_va, s_va)
    n_neg = int(np.count_nonzero(y_va == 0))
    d_star = info["Мінімум вартості"]
    fpr_star = d_star["FP"] / n_neg
    auto["auc"] = (f"ROC-AUC (validation) = {auc:.4f}; середня точність (AP) = {ap_score:.4f}; "
                   f"π = {pct(float(np.mean(y_va)))}. Для t*: FPR = {d_star['FP']}/{n_neg} = {fpr_star:.5f}, "
                   f"TPR = {fm(d_star['recall'])}, precision = {fm(d_star['precision'])}.\n\n"
                   f"Теоретичний орієнтир τ = 1/11 ≈ {THEORY_THRESHOLD:.4f} (лекція 2) дає на validation "
                   f"C = {ref_info['C']} (TP={ref_info['TP']}, FP={ref_info['FP']}, FN={ref_info['FN']}); "
                   f"t* обрано за мінімумом C на validation, а не за цією формулою.")
    save_curves(y_va, s_va, {"стандартний": 0.5, "t*": t_star}, out_dir / "validation_curves.png")
    save_cost_curve(tab, {"стандартний": 0.5, "t*": t_star}, out_dir / "validation_cost_vs_threshold.png")

    boot = bootstrap_thresholds(y_va, s_va)
    finite = boot[np.isfinite(boot)]
    share_inf = 100 * (1 - len(finite) / len(boot))
    if len(finite):
        q = np.percentile(finite, [5, 50, 95])
        q_txt = f"{q[0]:.4g} / {q[1]:.4g} / {q[2]:.4g} (за скінченними t*)"
    else:
        q_txt = "не визначено: усі t* дорівнюють +inf"
    auto["bootstrap"] = (f"Бутстреп validation ({N_BOOTSTRAP} вибірок, seed 0): 5/50/95-й перцентилі t* — {q_txt}; "
                         f"t* = +inf у {share_inf:.0f}% вибірок, t* < 0.5 у {100 * np.mean(boot < 0.5):.0f}%. "
                         f"Спираються лише на validation.")

    # ---------- Фіксуємо поріг ДО тесту ----------
    (out_dir / "threshold.json").write_text(json.dumps({
        "rule": "predict fraud if score >= t",
        "t_star": t_star if math.isfinite(t_star) else "inf",
        "t_star_repr": repr(t_star),
        "selected_on": "validation",
        "tie_break": "largest threshold among equal minimum cost",
        "validation_cost": int(tab["C"][i_best]),
        "cost_FN": 10, "cost_FP": 1,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------- Пункт 4: test (одноразово, з уже зафіксованим t*) ----------
    rows_te, info_te = [], {}
    for label, t in (("Стандартний поріг", 0.5), ("Мінімум вартості (t*)", t_star), ("Усі негативні", math.inf)):
        r, d = rule_row(label, t, y_te, s_te)
        rows_te.append(r)
        info_te[label] = d
    auto["test_table"] = md_table(RULE_HEADER, rows_te)
    d_te = info_te["Мінімум вартості (t*)"]
    notes = []
    if math.isnan(d_te["precision"]):
        notes.append("precision не визначена: TP+FP = 0, жодного позитивного рішення.")
    if math.isnan(d_te["recall"]):
        notes.append("recall не визначений: у test немає позитивних міток.")
    auto["test_notes"] = " ".join(notes) if notes else (
        f"Для t* обидва знаменники ненульові: TP+FP = {d_te['TP'] + d_te['FP']}, "
        f"TP+FN = {d_te['TP'] + d_te['FN']}.")

    pred_te = s_te >= t_star
    err = pd.DataFrame({
        "csv_row": saved["test_row_ids"], "label": y_te, "score": s_te,
        "amount": saved["test_amount"], "pred": pred_te,
    })
    err["type"] = np.where(err["pred"] & (err["label"] == 0), "FP",
                           np.where(~err["pred"] & (err["label"] == 1), "FN", ""))
    err["score_minus_t"] = err["score"] - t_star if math.isfinite(t_star) else np.nan
    err = err[err["type"] != ""].sort_values("csv_row").reset_index(drop=True)
    err[["csv_row", "type", "label", "score", "amount", "score_minus_t"]].to_csv(
        out_dir / "test_errors_all.csv", index=False)
    parts = []
    for typ in ("FP", "FN"):
        parts.append(err[err["type"] == typ].head(5))
    shown = pd.concat(parts)
    auto["errors"] = md_table(
        ["Рядок CSV", "Тип", "Мітка", "Оцінка", "Amount", "s − t*"],
        [[int(r.csv_row), r.type, int(r.label), f"{r.score:.6g}", f"{r.amount:.2f}", NA if math.isnan(r.score_minus_t) else f"{r.score_minus_t:+.4g}"]
         for r in shown.itertuples()])
    shown[["csv_row", "type", "label", "score", "amount", "score_minus_t"]].to_csv(
        out_dir / "test_errors_first5.csv", index=False)

    amt_all = float(np.median(saved["test_amount"]))
    stat_lines = [f"Усього в test: FP = {d_te['FP']}, FN = {d_te['FN']}. Медіана Amount у всьому test: {amt_all:.2f}."]
    for typ in ("FP", "FN"):
        sub = err[err["type"] == typ]
        if len(sub):
            stat_lines.append(
                f"{typ}: медіана Amount = {sub['amount'].median():.2f}, медіана |s − t*| = "
                f"{fm_g(sub['score_minus_t'].abs().median())} (за всіма {len(sub)} помилками цього типу, не лише п'ятьма).")
    auto["error_stats"] = "\n\n".join(stat_lines)

    v0, v1 = info["Стандартний поріг"], info["Мінімум вартості"]
    t0, t1 = info_te["Стандартний поріг"], info_te["Мінімум вартості (t*)"]
    auto["summary_numbers"] = "\n".join([
        delta_text("validation", v0, v1), delta_text("test", t0, t1),
        f"- Базова лінія «усі негативні»: C = 10·P = {info['Усі негативні']['C']} на validation і "
        f"{info_te['Усі негативні']['C']} на test.",
    ])

    # ---------- Звіт ----------
    report = "\n\n".join([
        f"# Результати аналізу\n\nt* = {t_star!r}",
        "## Вибірки\n\n" + auto["split_table"],
        "## Validation\n\n" + auto["val_table"] + "\n\n" + auto["auc"] + "\n\n" + auto["bootstrap"],
        "## Перевірка пошуку порога\n\n" + ver_md,
        "## Test\n\n" + auto["test_table"] + "\n\n" + auto["test_notes"],
        "## Помилки на test\n\n" + auto["errors"] + "\n\n" + auto["error_stats"],
        "## У цифрах\n\n" + auto["summary_numbers"],
    ]) + "\n"
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)

    # ---------- Автозаповнення README ----------
    if args.readme.exists():
        text = args.readme.read_text(encoding="utf-8")
        for name, body in auto.items():
            pat = re.compile(rf"(<!-- AUTO:{name} -->)(.*?)(<!-- /AUTO:{name} -->)", re.S)
            text, n = pat.subn(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
            if n == 0:
                print(f"(увага: у README немає маркера AUTO:{name})")
        args.readme.write_text(text, encoding="utf-8")
        print(f"README оновлено: {args.readme}")
    print(f"Результати: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
