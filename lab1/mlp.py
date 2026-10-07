import numpy as np

PARAM_NAMES = ("W1", "b1", "W2", "b2")


def init_params(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, np.sqrt(2.0 / 4), size=(4, 8))
    W2 = rng.normal(0.0, np.sqrt(2.0 / (8 + 3)), size=(8, 3))
    return {"W1": W1, "b1": np.zeros(8), "W2": W2, "b2": np.zeros(3)}


def log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=1, keepdims=True)
    s = z - m
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


def cross_entropy_from_logits(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    N = logits.shape[0]
    assert y.shape == (N,), f"очікувались мітки форми ({N},), отримано {y.shape}"
    logp = log_softmax(logits)
    loss = -logp[np.arange(N), y].mean()
    return float(loss), logp


def forward(params: dict, X: np.ndarray, y: np.ndarray):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = np.maximum(Z1, 0.0)
    Z2 = A1 @ params["W2"] + params["b2"]
    loss, logp = cross_entropy_from_logits(Z2, y)
    cache = {"X": X, "Z1": Z1, "A1": A1, "logp": logp}
    return loss, cache


def loss_only(params: dict, X: np.ndarray, y: np.ndarray) -> float:
    return forward(params, X, y)[0]


def backward(params: dict, cache: dict, y: np.ndarray, buggy: bool = False) -> dict:
    X, Z1, A1, logp = cache["X"], cache["Z1"], cache["A1"], cache["logp"]
    N = X.shape[0]

    dZ2 = np.exp(logp)
    dZ2[np.arange(N), y] -= 1.0
    if not buggy:
        dZ2 /= N

    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0.0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}