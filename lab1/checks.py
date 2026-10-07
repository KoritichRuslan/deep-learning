import numpy as np
import torch
import torch.nn.functional as F

from mlp import PARAM_NAMES, loss_only

TOL_TORCH = 1e-12
TOL_NUM = 1e-7
EPS = 1e-6
NUM_PARAMS = (("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,)))


def torch_reference(params: dict, X: np.ndarray, y: np.ndarray):
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.from_numpy(params["W1"].T.copy()))
        net[0].bias.copy_(torch.from_numpy(params["b1"].copy()))
        net[2].weight.copy_(torch.from_numpy(params["W2"].T.copy()))
        net[2].bias.copy_(torch.from_numpy(params["b2"].copy()))
    xt = torch.from_numpy(X.copy())
    yt = torch.from_numpy(y.copy())
    loss = F.cross_entropy(net(xt), yt)
    loss.backward()
    grads = {
        "W1": net[0].weight.grad.numpy().T.copy(),
        "b1": net[0].bias.grad.numpy().copy(),
        "W2": net[2].weight.grad.numpy().T.copy(),
        "b2": net[2].bias.grad.numpy().copy(),
    }
    return float(loss.item()), grads


def compare_with_torch(loss_np, grads_np, loss_t, grads_t):
    rows = []
    d = abs(loss_np - loss_t)
    ok = bool(np.isfinite(loss_np) and np.isfinite(loss_t) and d <= TOL_TORCH)
    rows.append(("Втрата", d, ok))
    for name in PARAM_NAMES:
        a, b = grads_np[name], grads_t[name]
        assert a.shape == b.shape, (name, a.shape, b.shape)
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        d = float(np.max(np.abs(a - b)))
        rows.append((f"Градієнт {name}", d, bool(finite and d <= TOL_TORCH)))
    return rows


def numeric_derivative(params: dict, X, y, name: str, idx: tuple, eps: float = EPS) -> float:
    p = {k: v.copy() for k, v in params.items()}
    theta0 = p[name][idx]
    p[name][idx] = theta0 + eps
    L_plus = loss_only(p, X, y)
    p[name][idx] = theta0 - eps
    L_minus = loss_only(p, X, y)
    p[name][idx] = theta0
    return (L_plus - L_minus) / (2.0 * eps)


def numeric_check(params: dict, X, y, grads_manual: dict, items=NUM_PARAMS):
    rows = []
    for name, idx in items:
        g_man = float(grads_manual[name][idx])
        g_num = numeric_derivative(params, X, y, name, idx)
        d = abs(g_num - g_man)
        label = f"{name}[{', '.join(map(str, idx))}]"
        rows.append((label, g_man, g_num, d, bool(np.isfinite(d) and d <= TOL_NUM)))
    return rows