import numpy as np
from sklearn.datasets import load_iris

N_PER_CLASS_TRAIN = 35


def load_split_standardize():
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    rng = np.random.default_rng(0)
    train_idx, test_idx = [], []
    for c in (0, 1, 2):
        idx = np.where(y == c)[0]
        idx = rng.permutation(idx)
        train_idx.append(idx[:N_PER_CLASS_TRAIN])
        test_idx.append(idx[N_PER_CLASS_TRAIN:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0, ddof=0)
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    assert X_tr.shape == (105, 4) and X_te.shape == (45, 4)
    assert np.bincount(y_tr).tolist() == [35, 35, 35]
    assert np.bincount(y_te).tolist() == [15, 15, 15]
    return X_tr, y_tr, X_te, y_te