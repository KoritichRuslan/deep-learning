"""ЛР3, команда аналізу.

    uv run analyze.py --data <тека з seg_train/ і seg_test/>

1) перевіряє, що SHA-256 split.csv однаковий у всіх семи журналах і збігається з поточним;
2) зі збережених журналів будує таблицю, криві, розклад ефектів, перевіряє правило вибору;
3) єдиний раз обчислює на CPU логіти test для lab3_model.pt, матрицю плутанини, п'ять помилок.
Усі тестові числа в README беруться з виводу цієї команди (analysis/results.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from torch import nn  # noqa: E402

import common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
COLORS = {"scratch": "#6b6b6b", "frozen": "#1f77b4", "unfreeze": "#d95f02"}
STYLES = {1: "-", 2: "--", 3: ":"}


def fail(msg):
    print("ПОМИЛКА:", msg, file=sys.stderr)
    sys.exit(1)


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    return s + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    out = a.out
    adir = out / "analysis"
    adir.mkdir(exist_ok=True)
    md = []

    # ---------------------------------------------------------------- 0. хеш split.csv
    split_path = out / "split.csv"
    cur = C.sha256_file(split_path)
    logs = C.load_run_logs(out)
    hashes = {k: l["split_csv_sha256"] for k, l in logs.items()}
    if len(set(hashes.values())) != 1 or cur not in hashes.values():
        fail(f"хеш split.csv у журналах {sorted(set(hashes.values()))} не збігається з поточним {cur}")
    if any(l["model"].get("debug_no_pretrained_weights") for l in logs.values()):
        fail("журнали містять налагоджувальні запуски без ваг ImageNet")
    print("SHA-256 split.csv однаковий у всіх 7 журналах і збігається з поточним:", cur)
    md.append(f"SHA-256 `split.csv` однаковий у всіх 7 журналах і збігається з поточним: `{cur}`\n")

    rows = C.read_split(split_path)
    val_rows = [r for r in rows if r[2] == "validation"]
    y_val = np.array([C.CLASSES.index(r[1]) for r in val_rows])

    # ---------------------------------------------------------------- 3. таблиця семи запусків
    trows, csv_lines = [], ["n,regime,seed,train_loss_ep8,val_macro_f1_ep8,train_time_s"]
    for n, (reg, seed) in enumerate(C.RUN_PLAN, 1):
        l = logs[(reg, seed)]
        # самоперевірка: F1 з val_logits.npy збігається з журналом
        vl = np.load(out / "runs" / C.run_dirname(reg, seed) / "val_logits.npy")
        assert vl.shape == (1498, 6)
        assert abs(C.macro_f1(y_val, vl) - C.final_f1(l)) < 1e-9, f"val_logits ≠ журнал для {reg} {seed}"
        trows.append([n, reg, seed, f"{l['train_loss'][-1]:.4f}", f"{C.final_f1(l):.4f}", f"{l['train_time_s']:.0f}"])
        csv_lines.append(f"{n},{reg},{seed},{l['train_loss'][-1]:.6f},{C.final_f1(l):.6f},{l['train_time_s']:.2f}")
    (adir / "runs_table.csv").write_text("\n".join(csv_lines) + "\n")
    md.append("## Таблиця семи запусків (п.3)\n\n" + md_table(
        ["#", "Режим", "Зерно", "Втрата train, епоха 8", "macro-F1 validation, епоха 8", "Час навчання, с"], trows))
    md.append("Час навчання - сума часу 8 епох навчання без валідації. "
              "(F1 з `val_logits.npy` збігається з журналом для всіх запусків.)\n")

    # рисунок: криві
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    ep = np.arange(1, C.EPOCHS + 1)
    for (reg, seed), l in logs.items():
        lab = f"{reg} s{seed}"
        ax[0].plot(ep, l["train_loss"], STYLES[seed], color=COLORS[reg], marker="o", ms=3, label=lab)
        ax[1].plot(ep, l["val_macro_f1"], STYLES[seed], color=COLORS[reg], marker="o", ms=3, label=lab)
    ax[0].set_title("Втрата на train (за епоху)")
    ax[1].set_title("macro-F1 на validation")
    for x in ax:
        x.set_xlabel("епоха")
        x.grid(alpha=0.3)
    ax[0].set_ylabel("CrossEntropy")
    ax[1].set_ylabel("macro-F1")
    ax[1].legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_curves.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- 4. розклад ефектів
    F = {k: C.final_f1(l) for k, l in logs.items()}
    def seeds(reg): return [F[(reg, s)] for s in (1, 2, 3)]
    Fbar = {"frozen": float(np.mean(seeds("frozen"))), "unfreeze": float(np.mean(seeds("unfreeze")))}
    Rr = {"frozen": max(seeds("frozen")) - min(seeds("frozen")),
          "unfreeze": max(seeds("unfreeze")) - min(seeds("unfreeze"))}
    R = max(Rr.values())
    Fs = F[("scratch", 1)]
    d_pre = Fbar["frozen"] - Fs
    d_unf = Fbar["unfreeze"] - Fbar["frozen"]
    def verdict(d): return "|Δ| ≤ R: різниця не відрізняється від розкиду, результату немає" if abs(d) <= R \
        else "|Δ| > R: ефект вартий перевірки, але не доведений"
    eff_rows = [
        ["F̄_frozen, R_frozen", f"{Fbar['frozen']:.4f}, {Rr['frozen']:.4f}"],
        ["F̄_unfreeze, R_unfreeze", f"{Fbar['unfreeze']:.4f}, {Rr['unfreeze']:.4f}"],
        ["F_scratch,1", f"{Fs:.4f}"],
        ["R = max(R_frozen, R_unfreeze)", f"{R:.4f}"],
        ["Δ_pre = F̄_frozen − F_scratch,1", f"{d_pre:+.4f} ({d_pre*100:+.1f} п. macro-F1)"],
        ["Δ_unf = F̄_unfreeze − F̄_frozen", f"{d_unf:+.4f} ({d_unf*100:+.1f} п. macro-F1)"],
        ["Δ_pre + Δ_unf = F̄_unfreeze − F_scratch,1", f"{d_pre + d_unf:+.4f} ({(d_pre + d_unf)*100:+.1f} п.)"],
    ]
    md.append("## Розклад ефектів (п.4)\n\n" + md_table(["Величина", "Значення"], eff_rows))
    md.append(f"- Δ_pre: {verdict(d_pre)} (|Δ_pre|={abs(d_pre):.4f}, R={R:.4f}).\n"
              f"- Δ_unf: {verdict(d_unf)} (|Δ_unf|={abs(d_unf):.4f}, R={R:.4f}).\n")
    tt = {reg: [logs[(reg, s)]["train_time_s"] for s in (1, 2, 3)] if reg != "scratch" else [logs[(reg, 1)]["train_time_s"]]
          for reg in ("scratch", "frozen", "unfreeze")}
    md.append("Середній час навчання за режимом, с: " + ", ".join(f"{r}: {np.mean(v):.0f}" for r, v in tt.items()) + "\n")

    # ---------------------------------------------------------------- 5. перевірка вибору
    ch = C.choose_run(logs)
    ckpt = torch.load(out / "lab3_model.pt", map_location="cpu", weights_only=True) \
        if "weights_only" in torch.load.__code__.co_varnames else torch.load(out / "lab3_model.pt", map_location="cpu")
    if set(ckpt) != {"state_dict", "regime", "seed", "classes", "img_size"}:
        fail(f"ключі lab3_model.pt: {sorted(ckpt)}")
    if (ckpt["regime"], ckpt["seed"]) != (ch["regime"], ch["seed"]):
        fail(f"lab3_model.pt ({ckpt['regime']}, {ckpt['seed']}) не відповідає правилу вибору {ch['regime']}, {ch['seed']}")
    if list(ckpt["classes"]) != C.CLASSES or ckpt["img_size"] != 160:
        fail("classes/img_size у lab3_model.pt")
    wp = out / "runs" / C.run_dirname(ch["regime"], ch["seed"]) / "weights.pt"
    if wp.exists():
        w = torch.load(wp, map_location="cpu")
        if not all(torch.equal(w[k], ckpt["state_dict"][k]) for k in w):
            fail("ваги lab3_model.pt не збігаються з вагами вибраного запуску")
    md.append("## Вибір конфігурації (п.5)\n\n" + md_table(
        ["Режим", "F̄_r (для scratch - F_scratch,1)"],
        [[r, f"{ch['means'][r]:.4f}"] for r in ("scratch", "frozen", "unfreeze")]))
    md.append(f"Вибрано режим **{ch['regime']}**, запуск із найбільшим F: **зерно {ch['seed']}** "
              f"(F = {ch['val_f1']:.4f}). `lab3_model.pt` відповідає правилу вибору за журналами.\n")
    if abs(d_unf) <= R:
        md.append("Увага: |Δ_unf| ≤ R, тож вибір між frozen і unfreeze визначив шум.\n")

    # ---------------------------------------------------------------- 5. тест: один раз, CPU
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, 6)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to("cpu").eval()
    test_ds = C.load_split_part(a.data, rows, "test")  # єдине місце в проєкті, де читається test
    logits = C.predict_logits(model, C.make_eval_loader(test_ds), torch.device("cpu"))
    np.save(adir / "test_logits.npy", logits.astype(np.float32))
    y = np.array(test_ds.targets)
    pred = logits.argmax(1)
    f1_test = C.macro_f1(y, logits)
    acc = float((pred == y).mean())
    n = len(y)
    se = float(np.sqrt(acc * (1 - acc) / n))
    cm = np.zeros((6, 6), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    md.append("## Тестова оцінка (п.5)\n\n"
              f"- macro-F1 на test: **{f1_test:.4f}**; accuracy: {acc:.4f}; n = {n}\n"
              f"- F вибраного запуску на validation: {ch['val_f1']:.4f}; різниця test − validation = "
              f"{f1_test - ch['val_f1']:+.4f}\n"
              f"- стандартна похибка частки правильних √(p(1−p)/n) = {se:.4f} "
              f"(≈ {se*100:.2f} п.)\n")
    md.append("Матриця плутанини (рядки - справжній клас, стовпці - передбачений):\n\n" +
              md_table(["справжній \\ передбачений"] + C.CLASSES,
                       [[C.CLASSES[i]] + [int(v) for v in cm[i]] for i in range(6)]))

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), C.CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(6), C.CLASSES)
    ax.set_xlabel("передбачений клас")
    ax.set_ylabel("справжній клас")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_title(f"Матриця плутанини на test ({ch['regime']}, зерно {ch['seed']}); macro-F1 = {f1_test:.3f}",
                 fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "fig_confusion.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- п'ять помилок
    pairs = [(cm[t, p], t, p) for t in range(6) for p in range(6) if t != p and cm[t, p] > 0]
    pairs.sort(key=lambda r: (-r[0], r[1], r[2]))
    soft = torch.softmax(torch.from_numpy(logits), 1).numpy()
    files = [r[0] for r in test_ds.rows]
    def err_files(t, p):  # імена, відсортовані як рядки
        return sorted((files[i], i) for i in range(n) if y[i] == t and pred[i] == p)
    chosen = []
    k = 0
    while len(chosen) < 5 and any(len(err_files(t, p)) > k for _, t, p in pairs):
        for cnt, t, p in pairs:
            ef = err_files(t, p)
            if len(ef) > k and len(chosen) < 5:
                fname, i = ef[k]
                chosen.append({"file": fname, "true": C.CLASSES[t], "pred": C.CLASSES[p],
                               "prob": float(soft[i, p]), "count": int(cnt), "idx": i})
        k += 1
    md.append("## П'ять характерних помилок (п.5)\n\n" + md_table(
        ["Файл", "Справжній клас", "Передбачений клас", "Імовірність передбаченого класу", "Кількість таких помилок у test"],
        [[c["file"], c["true"], c["pred"], f"{c['prob']:.3f}", c["count"]] for c in chosen]))
    md.append("Топ-10 пар помилок: " + ", ".join(f"({C.CLASSES[t]}→{C.CLASSES[p]}: {c})" for c, t, p in pairs[:10]) + "\n")
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.6))
    for ax, c in zip(axes, chosen):
        ax.imshow(test_ds.images[c["idx"]])
        ax.set_title(f"{c['file']}\nсправжній: {c['true']}\nпередбачено: {c['pred']} (p={c['prob']:.2f})", fontsize=9)
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "fig_errors.png", dpi=140)
    plt.close(fig)

    res = {"split_sha256": cur, "F": {f"{k[0]}_s{k[1]}": v for k, v in F.items()}, "Fbar": Fbar, "R_r": Rr, "R": R,
           "F_scratch1": Fs, "delta_pre": d_pre, "delta_unf": d_unf, "selected": ch,
           "test_macro_f1": f1_test, "test_accuracy": acc, "test_se_accuracy": se,
           "confusion_matrix": cm.tolist(), "five_errors": chosen}
    C.save_json(res, adir / "results.json")
    text = "\n".join(md)
    (adir / "results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
