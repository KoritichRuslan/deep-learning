"""Спільний код ЛР3: поділ, перетворення, датасет, модель, режими, цикл навчання.

Усі константи зафіксовано умовою завдання. Тестові зображення тут НЕ завантажуються
автоматично: тест читається лише функцією load_split_part("test"), яку викликає
виключно analyze.py (train.py її з частиною "test" не викликає).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from split import (CLASSES, CONTROL, CONTROL_TOTAL, N_PER_CLASS_TRAIN, PARTS, build_split,  # noqa: F401
                   check_no_overlap, read_split, sha256_file, split_control_table, write_split)

# --------------------------------------------------------------------------- константи
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EPOCHS = 8
TRAIN_BS = 32
EVAL_BS = 64
WEIGHT_DECAY = 1e-4
LR_HEAD = 3e-3
LR_BACKBONE = 3e-4
LR_SCRATCH = 3e-3
RESNET18_WEIGHTS_SHA256_PREFIX = "f37072fd"  # префікс SHA-256 файлу ваг (частина його імені)

# «У eval() під час навчання» для кожного режиму (імена модулів верхнього рівня)
EVAL_MODULES = {
    "scratch": [],
    "frozen": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    "unfreeze": ["conv1", "bn1", "layer1", "layer2", "layer3"],
}
# що навчається (імена модулів верхнього рівня)
TRAINABLE_MODULES = {
    "scratch": None,  # уся мережа
    "frozen": ["fc"],
    "unfreeze": ["layer4", "fc"],
}
EXPECTED_N_TRAINABLE = {"scratch": 11_179_590, "frozen": 3_078, "unfreeze": 8_396_806}
EXPECTED_N_TOTAL = 11_179_590
TOP_LEVEL = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]

# --------------------------------------------------------------------------- дані
def make_transforms(train: bool):
    norm = transforms.Normalize(MEAN, STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        norm,
    ])


class SplitDataset(Dataset):
    """Датасет за рядками split.csv. Зображення читаються через Pillow (.convert('RGB'))
    і кешуються в пам'яті у вигляді PIL-об'єктів (до перетворень), що не впливає на результат."""

    def __init__(self, data_root: Path, rows, part: str, train: bool):
        assert all(r[2] == part for r in rows)
        self.rows = rows
        self.tf = make_transforms(train)
        self.targets = [CLASSES.index(r[1]) for r in rows]
        base = data_root / ("seg_train" if part == "train" else "seg_test")
        base = base / base.name
        self.images = []
        for fname, label, _ in rows:
            with Image.open(base / label / fname) as im:
                self.images.append(im.convert("RGB"))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.targets[i]


def load_split_part(data_root: Path, split_rows, part: str) -> SplitDataset:
    rows = [r for r in split_rows if r[2] == part]
    return SplitDataset(data_root, rows, part, train=(part == "train"))


def make_train_loader(ds):
    # без власного generator/sampler: випадковість із глобального генератора torch
    return DataLoader(ds, batch_size=TRAIN_BS, shuffle=True, drop_last=False, num_workers=0)


def make_eval_loader(ds):
    return DataLoader(ds, batch_size=EVAL_BS, shuffle=False, num_workers=0)


# --------------------------------------------------------------------------- модель
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if dev.type == "cpu":
        return f"cpu ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)"
    return dev.type


def build_model(regime: str, seed: int, weights_file: Path | None = None,
                debug_no_weights: bool = False) -> nn.Module:
    """Кроки 1–4 з умови (крок 5, перенесення на пристрій й оптимізатор, робить викликач)."""
    assert regime in EVAL_MODULES
    torch.manual_seed(seed)                      # крок 1
    np.random.seed(seed)
    if regime == "scratch" or debug_no_weights:  # крок 2
        model = torchvision.models.resnet18(weights=None)
    elif weights_file is None:
        model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        # Те саме, що weights=IMAGENET1K_V1, але файл ваг узято локально
        # (download.pytorch.org може бути недоступним). Конструктор ініціалізує випадкові
        # ваги так само, як і при weights=W, тож споживання RNG збігається.
        digest = sha256_file(weights_file)
        assert digest.startswith(RESNET18_WEIGHTS_SHA256_PREFIX), \
            f"SHA-256 файлу ваг {digest[:8]} не збігається з очікуваним {RESNET18_WEIGHTS_SHA256_PREFIX}"
        model = torchvision.models.resnet18(weights=None)
        model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
    model.fc = nn.Linear(512, 6)                 # крок 3
    train_mods = TRAINABLE_MODULES[regime]       # крок 4
    if train_mods is not None:
        for p in model.parameters():
            p.requires_grad = False
        for name in train_mods:
            for p in getattr(model, name).parameters():
                p.requires_grad = True
    return model


def make_optimizer(model: nn.Module, regime: str) -> torch.optim.Optimizer:
    if regime == "scratch":
        groups = [{"params": list(model.parameters()), "lr": LR_SCRATCH}]
    elif regime == "frozen":
        groups = [{"params": list(model.fc.parameters()), "lr": LR_HEAD}]
    else:
        groups = [{"params": list(model.fc.parameters()), "lr": LR_HEAD},
                  {"params": list(model.layer4.parameters()), "lr": LR_BACKBONE}]
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def set_train_mode(model: nn.Module, regime: str) -> None:
    """model.train(), потім .eval() для модулів зі стовпця «У eval()» таблиці режимів.
    Заморожені модулі з BN мусять бути в eval(): requires_grad=False НЕ захищає ковзні
    статистики BN, їх оновлює сам прямий прохід у режимі train()."""
    model.train()
    for name in EVAL_MODULES[regime]:
        getattr(model, name).eval()


def count_params(params) -> int:
    return sum(p.numel() for p in params)


# --------------------------------------------------------------------------- навчання/оцінка
@torch.no_grad()
def predict_logits(model: nn.Module, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        out.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def macro_f1(y_true, logits) -> float:
    pred = np.asarray(logits).argmax(1)
    return float(f1_score(y_true, pred, average="macro", labels=list(range(6))))


def train_one_epoch(model, regime, loader, optimizer, device, n_total):
    set_train_mode(model, regime)
    crit = nn.CrossEntropyLoss()
    loss_sum = 0.0
    steps = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = crit(model(x), y)
        loss_sum += loss.item() * x.shape[0]  # L_b ДО кроку оптимізатора
        loss.backward()
        optimizer.step()
        steps += 1
    return loss_sum / n_total, steps


# --------------------------------------------------------------------------- журнал
def git_info(repo: Path) -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "uncommitted_changes": bool(run("status", "--porcelain", "--", "*.py", "pyproject.toml"))}
    except Exception:
        return {"commit": None, "uncommitted_changes": None}


def env_info(dev) -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "device": device_name(dev)}


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- вибір конфігурації
RUN_PLAN = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
            ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]


def run_dirname(regime: str, seed: int) -> str:
    return f"{regime}_seed{seed}"


def load_run_logs(out: Path) -> dict:
    logs = {}
    for regime, seed in RUN_PLAN:
        p = Path(out) / "runs" / run_dirname(regime, seed) / "run.json"
        logs[(regime, seed)] = json.loads(p.read_text(encoding="utf-8"))
    return logs


def final_f1(log: dict) -> float:
    return log["val_macro_f1"][-1]


def choose_run(logs: dict) -> dict:
    """Правило вибору (лише за validation): режим із найбільшим F̄_r (для scratch - єдине
    значення), у ньому - запуск із найбільшим F_{r,s} (за рівності - менше зерно)."""
    means = {}
    for regime in ("scratch", "frozen", "unfreeze"):
        vals = [final_f1(l) for (r, s), l in logs.items() if r == regime]
        means[regime] = sum(vals) / len(vals)
    best_regime = max(("scratch", "frozen", "unfreeze"), key=lambda r: (means[r]))
    cand = sorted(((s, final_f1(l)) for (r, s), l in logs.items() if r == best_regime),
                  key=lambda t: (-t[1], t[0]))
    return {"regime": best_regime, "seed": cand[0][0], "val_f1": cand[0][1], "means": means}
