from __future__ import annotations

from pathlib import Path
import random
import time
import matplotlib.pyplot as plt
from typing import List, Tuple

try:
    import seaborn as sns
    _HAS_SEABORN = True
except:
    _HAS_SEABORN = False

from .backtracking import rezolva_tsp_backtracking
from .nearest_neighbor import rezolva_tsp_nn, rezolva_tsp_nn_multistart

Matrix = List[List[int]]


def genereaza_matrice_aleatorie(n: int, seed: int = 42) -> Matrix:
    "Genereaza matrice simetrica [1,100]."
    random.seed(seed)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = random.randint(1, 100)
            matrix[i][j] = matrix[j][i] = d
    return matrix


def _time_call(func, *args, **kwargs) -> Tuple[float, object]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return time.perf_counter() - start, result


def ruleaza_experiment_timpi() -> Path:
    "Grafic 1 - Timp de rulare (a/c BT vs base/multi NN)."
    N_BT = [5, 8, 10, 12]
    N_NN = N_BT + [15, 20, 30, 50]
    seed_base = 42

    times_a = []
    times_c = []
    times_nn = []
    times_nn_multi = []

    for n in N_BT:
        matrix = genereaza_matrice_aleatorie(n, seed_base + n)
        d_a, _ = _time_call(rezolva_tsp_backtracking, n, matrix, mod='prima')
        d_c, _ = _time_call(rezolva_tsp_backtracking, n, matrix, mod='y_solutii', Y=n)
        times_a.append(d_a)
        times_c.append(d_c)

    for n in N_NN:
        matrix = genereaza_matrice_aleatorie(n, seed_base + n)
        d_nn, _ = _time_call(rezolva_tsp_nn, n, matrix)
        d_multi, _ = _time_call(rezolva_tsp_nn_multistart, n, matrix)
        times_nn.append(d_nn)
        times_nn_multi.append(d_multi)

    if _HAS_SEABORN:
        sns.set_theme()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(N_BT, times_a, 'o-', label='BT a) prima')
    ax1.plot(N_BT, times_c, 's-', label='BT c) Y solutii')
    ax1.plot(N_NN, times_nn, '^-', label='NN base')
    ax1.plot(N_NN, times_nn_multi, 'v-', label='NN multistart')
    ax1.set_xlabel('N')
    ax1.set_ylabel('Timp (s)')
    ax1.set_title('Timp linear')
    ax1.legend()
    ax1.grid(True)

    ax2.semilogy(N_BT, times_a, 'o-', label='BT a) prima')
    ax2.semilogy(N_BT, times_c, 's-', label='BT c) Y solutii')
    ax2.semilogy(N_NN, times_nn, '^-', label='NN base')
    ax2.semilogy(N_NN, times_nn_multi, 'v-', label='NN multistart')
    ax2.set_xlabel('N')
    ax2.set_ylabel('Timp (s, log)')
    ax2.set_title('Timp log')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    out = Path('timp_performanta.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    return out


def ruleaza_experiment_calitate() -> Path:
    "Grafic 2 - Calitate pentru timp fix T."
    N = 18
    T_s = [1, 2, 5]
    matrix = genereaza_matrice_aleatorie(N, 42)

    costs_bt = []
    costs_nn = []
    for t in T_s:
        _, ( _, cost_bt ) = _time_call(rezolva_tsp_backtracking, N, matrix, mod='exhaustiv', time_limit_s=t)
        costs_bt.append(cost_bt)
        Y_est = max(1, int(t * 1000))
        _, ( _, cost_nn ) = _time_call(rezolva_tsp_nn_multistart, N, matrix, Y=Y_est)
        costs_nn.append(cost_nn)

    fig, ax = plt.subplots()
    x_pos = range(len(T_s))
    width = 0.35
    ax.bar([p - width/2 for p in x_pos], costs_bt, width, label='BT exhaustiv')
    ax.bar([p + width/2 for p in x_pos], costs_nn, width, label='NN multistart')
    ax.set_xlabel('T (s)')
    ax.set_ylabel('Cost')
    ax.set_title('Calitate la timp fix T')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(T_s)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path('calitate_timp_fix.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    return out


def ruleaza_experiment_gap() -> Path:
    "Grafic 3 - Gap % fata de optim."
    Ns = [5, 8, 10, 12]
    gaps = []
    seed_base = 42

    for n in Ns:
        matrix = genereaza_matrice_aleatorie(n, seed_base + n)
        _, cost_opt = rezolva_tsp_backtracking(n, matrix, mod='exhaustiv')
        _, cost_nn = rezolva_tsp_nn_multistart(n, matrix)
        gap = 100 * (cost_nn - cost_opt) / cost_opt if cost_opt > 0 else 0
        gaps.append(gap)

    fig, ax = plt.subplots()
    ax.plot(Ns, gaps, 'o-')
    ax.set_xlabel('N')
    ax.set_ylabel('Gap %')
    ax.set_title('Gap NN multistart vs BT optim')
    ax.grid(True)
    plt.tight_layout()
    out = Path('gap_optimal.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    return out


def ruleaza_experiment():
    "Ruleaza toate 3 grafice."
    p1 = ruleaza_experiment_timpi()
    p2 = ruleaza_experiment_calitate()
    p3 = ruleaza_experiment_gap()
    print(f'Grafice generate: {p1}, {p2}, {p3}')
    return [p1, p2, p3]
