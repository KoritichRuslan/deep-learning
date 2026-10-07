import argparse
import sys

import numpy as np

from checks import (EPS, NUM_PARAMS, TOL_NUM, TOL_TORCH, compare_with_torch,
                    numeric_check, numeric_derivative, torch_reference)
from data import load_split_standardize
from mlp import (PARAM_NAMES, backward, cross_entropy_from_logits, forward,
                 init_params, log_softmax)

PREDICTION = """\
ПРОГНОЗ (записано до запуску):
  1. Втрата на тих самих параметрах НЕ зміниться: у досліді змінюється лише backward,
     а forward і формула втрати лишаються без змін.
  2. dL/dZ2 стане (q - y) замість (q - y)/N, тобто в N = 105 разів більшим. Решта
     зворотного проходу лінійна за dL/dZ2 (матричні добутки, маска ReLU, суми), тому
     КОЖЕН ненульовий елемент усіх чотирьох градієнтів зросте рівно в 105 разів.
     Нулі лишаться нулями.
  3. Втрата порівняно з PyTorch пройде, а порівняння градієнтів W1, b1, W2, b2 - ні.
     Чисельна перевірка впіймає помилку для тих параметрів, градієнт яких ненульовий.
     Якщо градієнт параметра дорівнює нулю (мертвий ReLU), помилка на ньому невидима.
"""


def fmt(x):
    return f"{x:.3e}"


def run_main_checks(params, X, y, buggy):
    loss, cache = forward(params, X, y)
    grads = backward(params, cache, y, buggy=buggy)
    loss_t, grads_t = torch_reference(params, X, y)

    print(f"Втрата NumPy  : {loss:.15f}")
    print(f"Втрата PyTorch: {loss_t:.15f}\n")

    rows_t = compare_with_torch(loss, grads, loss_t, grads_t)
    print(f"### Звірка з PyTorch (критерій |a-b| <= {TOL_TORCH:g})\n")
    print("| Величина | Макс. абс. різниця NumPy / PyTorch | Перевірку пройдено |")
    print("|---|---|---|")
    for name, d, ok in rows_t:
        print(f"| {name} | {fmt(d)} | {'так' if ok else 'НІ'} |")

    rows_n = numeric_check(params, X, y, grads)
    print(f"\n### Чисельна перевірка (eps = {EPS:g}, критерій |g_num - g_manual| <= {TOL_NUM:g})\n")
    print("| Параметр | Градієнт backward() | Чисельна похідна | Абсолютна різниця | Перевірку пройдено |")
    print("|---|---|---|---|---|")
    for label, g_man, g_num, d, ok in rows_n:
        print(f"| {label} | {g_man:.12e} | {g_num:.12e} | {fmt(d)} | {'так' if ok else 'НІ'} |")
    all_ok = all(r[2] for r in rows_t) and all(r[4] for r in rows_n)
    return loss, grads, rows_t, rows_n, all_ok


def run_bug(params, X, y):
    print(PREDICTION)
    print("=" * 70 + "\nФАКТИЧНИЙ РЕЗУЛЬТАТ (buggy=True)\n" + "=" * 70 + "\n")
    loss_ref, cache = forward(params, X, y)
    good = backward(params, cache, y, buggy=False)
    _, _, rows_t, rows_n, all_ok = run_main_checks(params, X, y, buggy=True)
    bad = backward(params, cache, y, buggy=True)

    print("\n### Відношення градієнтів bug / correct (лише ненульові елементи)\n")
    N = X.shape[0]
    for k in PARAM_NAMES:
        nz = np.abs(good[k]) > 0
        r = bad[k][nz] / good[k][nz]
        print(f"- {k}: ненульових {nz.sum()}/{nz.size}, відношення min={r.min():.12f}, max={r.max():.12f} (N={N})")

    loss_ok = rows_t[0][2]
    grads_flagged = [not r[2] for r in rows_t[1:]]
    num_flagged = [not r[4] for r in rows_n]
    print("\n### Зіставлення з прогнозом\n")
    print(f"- Втрата змінилась відносно правильної реалізації: {'ні' if loss_ok else 'так'} (прогноз: ні)")
    print(f"- PyTorch-звірка виявила помилку в градієнтах: {grads_flagged} (прогноз: усі True)")
    print(f"- Чисельна перевірка виявила помилку для [W1,b1,W2,b2]: {num_flagged}")
    detected = (not all_ok)
    print("\nРезультат досліду:", "ПОМИЛКУ ВИЯВЛЕНО (це правильний результат досліду)" if detected
          else "помилку НЕ виявлено - це провал досліду")
    return 0 if detected else 1


def run_extras(params, X, y):
    print("\n" + "=" * 70 + "\nДОДАТКОВІ ПЕРЕВІРКИ\n" + "=" * 70)

    z = np.array([[1000.0, 1001.0, 1002.0]])
    with np.errstate(all="ignore"):
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        naive = -np.log(p[0, 2])
    stable, _ = cross_entropy_from_logits(z, np.array([2]))
    shifted, _ = cross_entropy_from_logits(z - 1000.0, np.array([2]))
    import torch
    lib = torch.nn.functional.cross_entropy(torch.from_numpy(z), torch.tensor([2])).item()
    print("\n### 1. Стабільність при логітах ~1000 (клас 2)")
    print(f"- наївний softmax -> -log p: {naive}")
    print(f"- стабільний (зсув на max):  {stable:.15f}")
    print(f"- ті самі логіти мінус 1000: {shifted:.15f}  (інваріантність до зсуву)")
    print(f"- torch cross_entropy:       {lib:.15f}")
    assert np.isfinite(stable) and abs(stable - shifted) <= 1e-12 and abs(stable - lib) <= 1e-12
    assert np.allclose(np.exp(log_softmax(z)).sum(axis=1), 1.0)

    loss, cache = forward(params, X, y)
    grads = backward(params, cache, y)
    worst = {}
    for name in PARAM_NAMES:
        w = 0.0
        for idx in np.ndindex(*params[name].shape):
            w = max(w, abs(numeric_derivative(params, X, y, name, idx) - grads[name][idx]))
        worst[name] = w
    print("\n### 2. Повна чисельна перевірка (усі елементи), макс. |різниця|")
    for k, v in worst.items():
        print(f"- {k}: {fmt(v)} {'OK' if v <= TOL_NUM else 'FAIL'}")

    Z1 = cache["Z1"]
    active = (Z1 > 0).mean(axis=0)
    print("\n### 3. Діагностика ReLU і градієнтів (лекція 2)")
    print("- частка об'єктів з Z1>0 для кожного з 8 нейронів:", np.round(active, 3).tolist())
    print("- мертві нейрони (Z1<=0 на всіх 105 об'єктах):", np.where(active == 0)[0].tolist())
    print("- точні нулі в Z1:", int((Z1 == 0).sum()), "(рішення в нулі не впливає на перевірку)")
    for name, idx in NUM_PARAMS:
        print(f"- grad {name}{list(idx)} = {grads[name][idx]:.6e}")
    print("- норми градієнтів: " + ", ".join(f"||{k}||={np.linalg.norm(grads[k]):.4f}" for k in PARAM_NAMES))
    print(f"- sum(db2) = {grads['b2'].sum():.3e}  (рядки q-y сумуються в 0; ця інваріанта НЕ ловить помилку масштабу)")
    print(f"- початкова втрата {loss:.6f} (ln 3 = {np.log(3):.6f})")

    name, idx = "W2", (0, 0)
    print(f"\n### 4. Чисельна похідна {name}{list(idx)} залежно від eps")
    print("| eps | |g_num - g_manual| |")
    print("|---|---|")
    for e in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10):
        d = abs(numeric_derivative(params, X, y, name, idx, eps=e) - grads[name][idx])
        print(f"| {e:g} | {fmt(d)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bug", action="store_true", help="дослід із навмисною помилкою")
    ap.add_argument("--extras", action="store_true", help="додаткові перевірки")
    args = ap.parse_args()

    X, y, _, _ = load_split_standardize()
    params = init_params(0)
    assert all(p.dtype == np.float64 for p in params.values()) and X.dtype == np.float64

    if args.bug:
        return run_bug(params, X, y)
    *_, all_ok = run_main_checks(params, X, y, buggy=False)
    if args.extras:
        run_extras(params, X, y)
    print("\nПІДСУМОК:", "усі перевірки пройдено" if all_ok else "є перевірки, що не пройшли")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())