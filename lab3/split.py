"""Побудова поділу (без torch): процедура з умови, контрольна таблиця, перевірки.

Запуск окремо:  python split.py --data <шлях до теки з seg_train/ і seg_test/>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
N_PER_CLASS_TRAIN = 200
PARTS = ["train", "validation", "test"]

# Контрольна таблиця поділу з умови: (клас, частина) -> (кількість, сума числових імен)
CONTROL = {
    ("buildings", "train"): (200, 1_996_761), ("buildings", "validation"): (218, 4_837_323),
    ("buildings", "test"): (219, 4_837_559),
    ("forest", "train"): (200, 2_032_625), ("forest", "validation"): (237, 5_276_962),
    ("forest", "test"): (237, 5_291_326),
    ("glacier", "train"): (200, 2_183_923), ("glacier", "validation"): (276, 6_135_652),
    ("glacier", "test"): (277, 6_157_250),
    ("mountain", "train"): (200, 1_950_345), ("mountain", "validation"): (262, 5_789_535),
    ("mountain", "test"): (263, 5_849_765),
    ("sea", "train"): (200, 1_987_613), ("sea", "validation"): (255, 5_655_068),
    ("sea", "test"): (255, 5_668_011),
    ("street", "train"): (200, 1_841_817), ("street", "validation"): (250, 5_541_845),
    ("street", "test"): (251, 5_547_533),
}
CONTROL_TOTAL = {"train": (1200, 11_993_084), "validation": (1498, 33_236_385),
                 "test": (1502, 33_351_444)}


# --------------------------------------------------------------------------- поділ
def build_split(data_root: Path) -> list[tuple[str, str, str]]:
    """Повертає рядки (file, label, part) за процедурою з умови."""
    train_dir = data_root / "seg_train" / "seg_train"
    test_dir = data_root / "seg_test" / "seg_test"
    rows: dict[str, list[tuple[str, str, str]]] = {p: [] for p in PARTS}

    rng = np.random.default_rng(0)  # один генератор для всіх класів по черзі
    for cls in CLASSES:
        names = sorted(p.name for p in (train_dir / cls).iterdir() if p.suffix == ".jpg")
        n = len(names)
        idx = np.arange(n)
        rng.shuffle(idx)
        for i in idx[:N_PER_CLASS_TRAIN]:
            rows["train"].append((names[i], cls, "train"))

    rng = np.random.default_rng(0)  # новий генератор для seg_test
    for cls in CLASSES:
        names = sorted(p.name for p in (test_dir / cls).iterdir() if p.suffix == ".jpg")
        n = len(names)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = n // 2
        for i in idx[:cut]:
            rows["validation"].append((names[i], cls, "validation"))
        for i in idx[cut:]:
            rows["test"].append((names[i], cls, "test"))

    # порядок рядків у файлі: частина -> клас -> ім'я файлу (рядкове сортування)
    out: list[tuple[str, str, str]] = []
    for part in PARTS:
        out += sorted(rows[part], key=lambda r: (CLASSES.index(r[1]), r[0]))
    return out


def write_split(rows, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "part"])
        w.writerows(rows)


def read_split(path: Path) -> list[tuple[str, str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [(d["file"], d["label"], d["part"]) for d in r]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_control_table(rows) -> tuple[list[dict], bool]:
    """Порівняння з контрольною таблицею. Повертає (рядки таблиці, чи збіглося все)."""
    agg: dict[tuple[str, str], list[int]] = {}
    for fname, label, part in rows:
        c = agg.setdefault((label, part), [0, 0])
        c[0] += 1
        c[1] += int(Path(fname).stem)
    table, ok_all = [], True
    for cls in CLASSES:
        row = {"class": cls}
        for part in PARTS:
            got = tuple(agg.get((cls, part), [0, 0]))
            exp = CONTROL[(cls, part)]
            row[part] = {"got": got, "expected": exp, "ok": got == exp}
            ok_all &= got == exp
        table.append(row)
    tot = {"class": "Разом"}
    for part in PARTS:
        got = (sum(v[0] for (c, p), v in agg.items() if p == part),
               sum(v[1] for (c, p), v in agg.items() if p == part))
        exp = CONTROL_TOTAL[part]
        tot[part] = {"got": got, "expected": exp, "ok": got == exp}
        ok_all &= got == exp
    table.append(tot)
    return table, ok_all


def check_no_overlap(rows) -> bool:
    parts_of: dict[str, set[str]] = {}
    for fname, _, part in rows:
        parts_of.setdefault(fname, set()).add(part)
    return all(len(v) == 1 for v in parts_of.values())




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("split.csv"))
    a = ap.parse_args()
    rows = build_split(a.data)
    write_split(rows, a.out)
    table, ok = split_control_table(rows)
    for r in table:
        print(r["class"].ljust(10), *[f"{r[p]['got'][0]} / {r[p]['got'][1]:,}".replace(",", " ").ljust(24) +
                                      ("OK " if r[p]["ok"] else "ПОМИЛКА ") for p in PARTS])
    print("без перетину частин:", check_no_overlap(rows), "| усе збіглося:", ok)
    print("sha256:", sha256_file(a.out))
    sys.exit(0 if ok else 1)
