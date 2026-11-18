"""ЛР3, команда навчання.

    uv run train.py --data <тека з seg_train/ і seg_test/>

Виконує: поділ (+ контрольна таблиця) -> перевірки п.1-2 -> сім запусків (п.3) ->
вибір за validation і збереження lab3_model.pt (п.5, без test).
Тестові зображення тут НЕ завантажуються.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import common as C  # noqa: E402

HERE = Path(__file__).resolve().parent

# ---- Прогноз до досліду з навмисною помилкою (записано до запуску) ----
BN_PREDICTION = (
    "Режим frozen з model.train() замість set_train_mode. Параметри всього кістяка (conv1, bn1, "
    "layer1-4) НЕ зміняться: вони з requires_grad=False й не в оптимізаторі. Ковзні статистики "
    "BN (running_mean, running_var, num_batches_tracked) у bn1 та layer1-layer4 ЗМІНЯТЬСЯ: "
    "їх оновлює прямий прохід у режимі train(), градієнти тут ні до чого. Змінюватиметься лише fc. "
    "Із кривої втрати на train це майже не видно: кістяк і далі нормує пакетними статистиками, "
    "втрата просто трохи інша; наслідок проявиться на validation/test, де BN бере "
    "накопичені (зсунуті) статистики замість статистик ImageNet."
)


def device_sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


# ============================================================ п.1: перевірки до навчання
def save_batch_figure(train_ds, path: Path):
    torch.manual_seed(1)
    x, y = next(iter(C.make_train_loader(train_ds)))
    mean = torch.tensor(C.MEAN).view(3, 1, 1)
    std = torch.tensor(C.STD).view(3, 1, 1)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9.6))
    for i, ax in enumerate(axes.flat):
        img = (x[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(C.CLASSES[y[i]], fontsize=10)
        ax.axis("off")
    fig.suptitle("Перші 16 зображень навчального пакета після аугментацій (зерно 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def initial_loss(regime, train_ds, dev, weights_file, debug):
    model = C.build_model(regime, 1, weights_file, debug).to(dev)
    loader = C.make_train_loader(train_ds)
    x, y = next(iter(loader))
    C.set_train_mode(model, regime)
    with torch.no_grad():
        loss = nn.CrossEntropyLoss()(model(x.to(dev)), y.to(dev)).item()
    return loss


# ============================================================ п.2: перевірки режимів
def param_report(regime, dev, weights_file, debug):
    model = C.build_model(regime, 1, weights_file, debug).to(dev)
    opt = C.make_optimizer(model, regime)
    total = C.count_params(model.parameters())
    trainable = C.count_params(p for p in model.parameters() if p.requires_grad)
    groups = [{"n_params": C.count_params(g["params"]), "lr": g["lr"]} for g in opt.param_groups]
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    leak = [n for n, p in model.named_parameters() if not p.requires_grad and id(p) in in_opt]
    ok = (total == C.EXPECTED_N_TOTAL and trainable == C.EXPECTED_N_TRAINABLE[regime]
          and not leak and sum(g["n_params"] for g in groups) == trainable)
    return {"regime": regime, "total": total, "trainable": trainable, "groups": groups,
            "frozen_params_in_optimizer": leak, "ok": ok}


def state_change_experiment(regime, train_ds, dev, weights_file, debug, use_set_train_mode=True):
    model = C.build_model(regime, 1, weights_file, debug).to(dev)
    opt = C.make_optimizer(model, regime)
    loader = C.make_train_loader(train_ds)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if use_set_train_mode:
        C.set_train_mode(model, regime)
    else:
        model.train()
    crit = nn.CrossEntropyLoss()
    for i, (x, y) in enumerate(loader):
        if i == 5:
            break
        opt.zero_grad()
        crit(model(x.to(dev)), y.to(dev)).backward()
        opt.step()
    after = model.state_dict()
    buf_names = ("running_mean", "running_var", "num_batches_tracked")
    rows = []
    for name in C.TOP_LEVEL:
        p_changed, b_changed, has_bn = False, False, False
        for k in before:
            if k.split(".")[0] != name:
                continue
            same = torch.equal(before[k], after[k])
            if k.endswith(buf_names):
                has_bn = True
                b_changed |= not same
            else:
                p_changed |= not same
        if regime == "scratch":
            expect_frozen = False
        else:
            expect_frozen = name in C.EVAL_MODULES[regime]
        exp_p = not expect_frozen
        exp_b = (not expect_frozen) if has_bn else None
        ok = (p_changed == exp_p) and ((b_changed == exp_b) if has_bn else (not b_changed))
        rows.append({"module": name, "params_changed": p_changed,
                     "bn_stats_changed": (b_changed if has_bn else "немає BN"), "ok": ok})
    key = "layer1.0.bn1"
    drift = {
        "max_abs_dRunningMean": float((after[f"{key}.running_mean"].cpu() - before[f"{key}.running_mean"].cpu()).abs().max()),
        "max_abs_dRunningVar": float((after[f"{key}.running_var"].cpu() - before[f"{key}.running_var"].cpu()).abs().max()),
    }
    return {"regime": regime, "use_set_train_mode": use_set_train_mode, "rows": rows,
            "layer1.0.bn1": drift, "ok": all(r["ok"] for r in rows)}


def bn_fault_prediction_holds(res) -> bool:
    """Зіставлення результату досліду з прогнозом BN_PREDICTION."""
    ok = True
    for r in res["rows"]:
        if r["module"] == "fc":
            ok &= r["params_changed"] is True
        elif r["module"] == "conv1":
            ok &= r["params_changed"] is False and r["bn_stats_changed"] == "немає BN"
        else:  # bn1, layer1..layer4
            ok &= r["params_changed"] is False and r["bn_stats_changed"] is True
    return ok


def run_checks(rows, train_ds, dev, weights_file, debug, out: Path):
    cdir = out / "checks"
    cdir.mkdir(exist_ok=True)
    res = {}
    table, ok = C.split_control_table(rows)
    res["split"] = {"table": table, "all_match": ok, "no_overlap": C.check_no_overlap(rows)}
    assert ok and res["split"]["no_overlap"], "Поділ не збігається з контрольною таблицею!"

    save_batch_figure(train_ds, out / "fig_batch.png")

    res["L0"] = {r: initial_loss(r, train_ds, dev, weights_file, debug) for r in ("scratch", "frozen")}
    res["L0"]["ln6"] = float(np.log(6))

    res["params"] = [param_report(r, dev, weights_file, debug) for r in ("scratch", "frozen", "unfreeze")]
    assert all(p["ok"] for p in res["params"]), "Кількості параметрів не збіглися!"

    res["state_change"] = [state_change_experiment(r, train_ds, dev, weights_file, debug)
                           for r in ("frozen", "unfreeze")]
    assert all(s["ok"] for s in res["state_change"]), "Перевірка заморожування не збіглася з очікуваним!"

    fault = state_change_experiment("frozen", train_ds, dev, weights_file, debug, use_set_train_mode=False)
    res["bn_fault"] = {"prediction": BN_PREDICTION, "result": fault,
                       "prediction_holds": bn_fault_prediction_holds(fault)}
    assert res["bn_fault"]["prediction_holds"], "Результат досліду з BN не збігся з прогнозом!"
    C.save_json(res, out / "checks.json")
    return res


# ============================================================ п.3: запуск
def run_config(regime, seed, opt, weights_file, debug):
    return {
        "seed": seed, "regime": regime,
        "transforms": {
            "train": "RandomResizedCrop(160, scale=(0.7,1.0)) -> RandomHorizontalFlip() -> ToTensor() -> Normalize",
            "validation_test": "Resize((160,160)) -> ToTensor() -> Normalize",
            "mean": C.MEAN, "std": C.STD, "img_size": C.IMG_SIZE},
        "model": {"arch": "resnet18", "weights": None if regime == "scratch" else "IMAGENET1K_V1",
                  "weights_loaded_from_file": (weights_file.name if (weights_file and regime != "scratch") else None),
                  "debug_no_pretrained_weights": bool(debug),
                  "fc": "Linear(512, 6) default init",
                  "bn": {"momentum": 0.1, "eps": 1e-5},
                  "n_params_total": C.EXPECTED_N_TOTAL, "n_params_trainable": C.EXPECTED_N_TRAINABLE[regime],
                  "trainable_modules": C.TRAINABLE_MODULES[regime] or "all",
                  "eval_modules_during_training": C.EVAL_MODULES[regime]},
        "optimizer": {"name": "torch.optim.AdamW", "weight_decay": C.WEIGHT_DECAY,
                      "param_groups": [{"n_params": C.count_params(g["params"]), "lr": g["lr"]} for g in opt.param_groups],
                      "defaults": {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.defaults.items()
                                   if k not in ("params",) and not callable(v)}},
        "loop": {"epochs": C.EPOCHS, "train_batch_size": C.TRAIN_BS, "eval_batch_size": C.EVAL_BS,
                 "shuffle_train": True, "drop_last": False, "num_workers": 0,
                 "steps_per_epoch": 38, "total_steps": 304, "loss": "CrossEntropyLoss(mean)",
                 "lr_schedule": None, "grad_clip": None, "amp": None, "label_smoothing": 0,
                 "class_weights": None, "early_stopping": None, "checkpoint_selection": None,
                 "dtype": "float32"},
    }


def run_one(regime, seed, train_ds, val_ds, dev, out, split_hash, weights_file, debug):
    rdir = out / "runs" / C.run_dirname(regime, seed)
    rdir.mkdir(parents=True, exist_ok=True)
    model = C.build_model(regime, seed, weights_file, debug).to(dev)  # крок 5
    opt = C.make_optimizer(model, regime)
    train_loader, val_loader = C.make_train_loader(train_ds), C.make_eval_loader(val_ds)
    y_val = np.array(val_ds.targets)
    cfg = run_config(regime, seed, opt, weights_file, debug)
    losses, f1s, ep_times = [], [], []
    t_total0 = time.perf_counter()
    for ep in range(1, C.EPOCHS + 1):
        device_sync(dev)
        t0 = time.perf_counter()
        loss, steps = C.train_one_epoch(model, regime, train_loader, opt, dev, len(train_ds))
        device_sync(dev)
        ep_times.append(time.perf_counter() - t0)
        assert steps == 38
        logits = C.predict_logits(model, val_loader, dev)
        f1 = C.macro_f1(y_val, logits)
        losses.append(loss)
        f1s.append(f1)
        print(f"[{regime} s{seed}] epoch {ep}: train_loss={loss:.4f} val_macroF1={f1:.4f} "
              f"({ep_times[-1]:.0f}s)", flush=True)
    total_wall = time.perf_counter() - t_total0
    np.save(rdir / "val_logits.npy", logits.astype(np.float32))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, rdir / "weights.pt")
    log = {**cfg,
           "git": C.git_info(HERE), "split_csv_sha256": split_hash,
           "pretrained_weights_sha256": (C.sha256_file(weights_file) if (weights_file and regime != "scratch") else None),
           "env": C.env_info(dev),
           "train_loss": losses, "val_macro_f1": f1s,
           "epoch_train_time_s": ep_times,
           "train_time_s": float(sum(ep_times)),
           "total_wall_time_s_with_validation": total_wall}
    C.save_json(log, rdir / "run.json")
    return log


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="тека, що містить seg_train/ і seg_test/")
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--weights-file", type=Path, default=None,
                    help="локальний файл resnet18-f37072fd.pth (якщо download.pytorch.org недоступний)")
    ap.add_argument("--resume", action="store_true", help="пропустити запуски, для яких уже є run.json")
    ap.add_argument("--only-checks", action="store_true")
    ap.add_argument("--skip-checks", action="store_true", help="лише для поетапного/відновлюваного запуску")
    ap.add_argument("--runs", nargs="*", default=None, help="підмножина запусків, напр. scratch_seed1")
    ap.add_argument("--debug-no-pretrained", action="store_true",
                    help="ЛИШЕ ДЛЯ НАЛАГОДЖЕННЯ: випадкові ваги замість ImageNet; результати не годяться")
    a = ap.parse_args()
    out, debug = a.out, a.debug_no_pretrained
    out.mkdir(parents=True, exist_ok=True)

    wf = a.weights_file
    if wf is None and (out / "resnet18-f37072fd.pth").exists():
        wf = out / "resnet18-f37072fd.pth"
    dev = C.pick_device()
    print("device:", C.device_name(dev), "| torch", torch.__version__, flush=True)

    # --- поділ
    split_path = out / "split.csv"
    rows = C.build_split(a.data)
    new_tmp = out / "split.csv.new"
    C.write_split(rows, new_tmp)
    new_hash, old_hash = C.sha256_file(new_tmp), (C.sha256_file(split_path) if split_path.exists() else None)
    if old_hash is not None and old_hash != new_hash:
        print("УВАГА: split.csv змінився; усі запуски треба перенавчити.")
        shutil.rmtree(out / "runs", ignore_errors=True)
    shutil.move(new_tmp, split_path)
    split_hash = C.sha256_file(split_path)
    rows = C.read_split(split_path)  # далі дані - лише за split.csv
    print("split.csv sha256:", split_hash)

    train_ds = C.load_split_part(a.data, rows, "train")
    val_ds = C.load_split_part(a.data, rows, "validation")  # test тут не завантажується

    # --- перевірки
    if not a.skip_checks:
        res = run_checks(rows, train_ds, dev, wf, debug, out)
        print("L0:", res["L0"])
        print("Усі перевірки п.1-2 збіглися з очікуваним.", flush=True)
    if a.only_checks:
        return

    # --- сім запусків
    for regime, seed in C.RUN_PLAN:
        if a.runs is not None and C.run_dirname(regime, seed) not in a.runs:
            continue
        rj = out / "runs" / C.run_dirname(regime, seed) / "run.json"
        if a.resume and rj.exists() and json.loads(rj.read_text())["split_csv_sha256"] == split_hash:
            print(f"пропуск {regime} seed{seed} (є журнал)")
            continue
        run_one(regime, seed, train_ds, val_ds, dev, out, split_hash, wf, debug)

    if a.runs is not None:
        return
    # --- вибір за validation
    logs = C.load_run_logs(out)
    ch = C.choose_run(logs)
    wpath = out / "runs" / C.run_dirname(ch["regime"], ch["seed"]) / "weights.pt"
    torch.save({"state_dict": torch.load(wpath, map_location="cpu"), "regime": ch["regime"],
                "seed": ch["seed"], "classes": C.CLASSES, "img_size": C.IMG_SIZE}, out / "lab3_model.pt")
    print("Вибрано:", ch)


if __name__ == "__main__":
    main()
