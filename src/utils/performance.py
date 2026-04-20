from __future__ import annotations

"""Experimente de performanta pentru TSP (Lab3/Lab4).

Acest modul este folosit de CLI (src/main.py) prin:
- ruleaza_experiment()      -> Lab3 (BT vs Hill Climbing)
- ruleaza_experiment_lab4() -> Lab4 (BT moduri vs NN / NN AIMA)

Nota: modulul a fost curatat pentru a evita duplicate si probleme de indentare.
"""

from pathlib import Path
import random
import time
from typing import List, Tuple

import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore

    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

from .backtracking import rezolva_tsp_backtracking, rezolva_tsp_backtracking_extins
from .hill_climbing_tsp import rezolva_tsp_hc
from .nearest_neighbor import rezolva_tsp_nn, rezolva_tsp_nn_multistart
from .nn_aima import rezolva_tsp_nn_aima, rezolva_tsp_nn_aima_multistart


Matrix = List[List[int]]


def genereaza_instanta_tsp(n: int, rng: random.Random) -> Matrix:
    """Genereaza o matrice de distante TSP simetrica (int in [1, 100])."""
    matrix: Matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = int(rng.randint(1, 100))
            matrix[i][j] = matrix[j][i] = d
    return matrix


def _time_call(func, *args, **kwargs) -> Tuple[float, object]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return time.perf_counter() - start, result


def ruleaza_experiment(
    output_png: str | Path = "comparare_performanta.png",
    seed: int = 42,
    reporniri_hc: int = 30,
    iteratii_hc: int = 2000,
    bt_time_limit_s: float = 30.0,
) -> Path:
    """Ruleaza experimentul comparativ Lab3 (BT vs Hill Climbing) si salveaza PNG.

    Protocol (conform README):
    - N pentru backtracking: 5, 7, 8, 10, 12
    - N pentru hill climbing: 5, 7, 8, 10, 12, 15, 20, 30, 50
    - distante intregi in [1, 100], matrice simetrica, seed fix
    - grafic: 2 subploturi (liniar + semilogy)
    """

    valori_n_bt = [5, 7, 8, 10, 12]
    valori_n_hc = [5, 7, 8, 10, 12, 15, 20, 30, 50]

    times_bt: List[float] = []
    times_hc: List[float] = []

    max_n_bt_sub_prag: int | None = None

    for n in valori_n_bt:
        rng = random.Random(seed + n)
        matrix = genereaza_instanta_tsp(n, rng)
        duration, _ = _time_call(rezolva_tsp_backtracking, n, matrix)
        times_bt.append(float(duration))
        if duration <= bt_time_limit_s:
            max_n_bt_sub_prag = n

    for n in valori_n_hc:
        rng = random.Random(seed + n)
        matrix = genereaza_instanta_tsp(n, rng)
        duration, _ = _time_call(
            rezolva_tsp_hc,
            n,
            matrix,
            reporniri=reporniri_hc,
            iteratii=iteratii_hc,
            seed=seed,
        )
        times_hc.append(float(duration))

    if _HAS_SEABORN:
        sns.set_theme()

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax_lin.plot(valori_n_bt, times_bt, marker="o", label="Backtracking")
    ax_lin.plot(valori_n_hc, times_hc, marker="o", label="Hill Climbing (RR)")
    ax_lin.set_title("Timp executie (scala liniara)")
    ax_lin.set_xlabel("N (orase)")
    ax_lin.set_ylabel("Timp (secunde)")
    ax_lin.grid(True, which="both", alpha=0.3)
    ax_lin.legend()

    ax_log.semilogy(valori_n_bt, times_bt, marker="o", label="Backtracking")
    ax_log.semilogy(valori_n_hc, times_hc, marker="o", label="Hill Climbing (RR)")
    ax_log.set_title("Timp executie (scala log)")
    ax_log.set_xlabel("N (orase)")
    ax_log.set_ylabel("Timp (secunde, log)")
    ax_log.grid(True, which="both", alpha=0.3)
    ax_log.legend()

    if max_n_bt_sub_prag is not None:
        fig.suptitle(f"Prag backtracking {bt_time_limit_s:.0f}s: max N = {max_n_bt_sub_prag}")

    fig.tight_layout()
    out_path = Path(output_png)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def ruleaza_experiment_lab4(
    output_png: str | Path = "comparare_performanta_lab4.png",
    seed: int = 42,
    timp_nn_s: float = 1.0,
) -> Path:
    """Ruleaza experimentul comparativ cerut in Lab #04.

    Grafic minim: compara timpii de executie pentru 4 valori N (5, 8, 10, 12)
    pentru cazul a) prima solutie si cazul c) Y solutii (Y=N), pentru:
    - backtracking (modurile prima / y_solutii)
    - nearest neighbor manual
    - nearest neighbor "aima" (wrapper)

    Parametrul timp_nn_s e pastrat pentru compatibilitate CLI.
    """

    valori_n = [5, 8, 10, 12]
    valori_n_nn_extra = [15, 20, 30, 50]

    bt_prima: List[float] = []
    bt_y: List[float] = []
    nn_prima: List[float] = []
    nn_y: List[float] = []
    aima_prima: List[float] = []
    aima_y: List[float] = []

    for n in valori_n:
        matrix = genereaza_instanta_tsp(n, random.Random(seed + n))

        d, _ = _time_call(rezolva_tsp_backtracking_extins, n, matrix, mod="prima")
        bt_prima.append(float(d))
        d, _ = _time_call(rezolva_tsp_backtracking_extins, n, matrix, mod="y_solutii", y_max=n)
        bt_y.append(float(d))

        d, _ = _time_call(rezolva_tsp_nn, n, matrix, 0)
        nn_prima.append(float(d))
        d, _ = _time_call(rezolva_tsp_nn_multistart, n, matrix)
        nn_y.append(float(d))

        d, _ = _time_call(rezolva_tsp_nn_aima, n, matrix, 0)
        aima_prima.append(float(d))
        d, _ = _time_call(rezolva_tsp_nn_aima_multistart, n, matrix)
        aima_y.append(float(d))

    nn_prima_extra: List[float] = []
    nn_y_extra: List[float] = []
    aima_prima_extra: List[float] = []
    aima_y_extra: List[float] = []

    for n in valori_n_nn_extra:
        matrix = genereaza_instanta_tsp(n, random.Random(seed + n))

        d, _ = _time_call(rezolva_tsp_nn, n, matrix, 0)
        nn_prima_extra.append(float(d))
        d, _ = _time_call(rezolva_tsp_nn_multistart, n, matrix)
        nn_y_extra.append(float(d))

        d, _ = _time_call(rezolva_tsp_nn_aima, n, matrix, 0)
        aima_prima_extra.append(float(d))
        d, _ = _time_call(rezolva_tsp_nn_aima_multistart, n, matrix)
        aima_y_extra.append(float(d))

    if _HAS_SEABORN:
        sns.set_theme()

    fig, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax_a.plot(valori_n, bt_prima, marker="o", label="BT (prima)")
    ax_a.plot(valori_n, nn_prima, marker="o", label="NN manual (start=0)")
    ax_a.plot(valori_n, aima_prima, marker="o", label="NN aima (start=0)")
    ax_a.plot(valori_n_nn_extra, nn_prima_extra, marker="x", linestyle="--", label="NN manual (extra N)")
    ax_a.plot(valori_n_nn_extra, aima_prima_extra, marker="x", linestyle="--", label="NN aima (extra N)")
    ax_a.set_title("Caz a) prima solutie / un start")
    ax_a.set_xlabel("N (orase)")
    ax_a.set_ylabel("Timp (secunde)")
    ax_a.grid(True, which="both", alpha=0.3)
    ax_a.legend(fontsize=8)

    ax_c.plot(valori_n, bt_y, marker="o", label="BT (Y=N)")
    ax_c.plot(valori_n, nn_y, marker="o", label="NN manual (multistart)")
    ax_c.plot(valori_n, aima_y, marker="o", label="NN aima (multistart)")
    ax_c.plot(valori_n_nn_extra, nn_y_extra, marker="x", linestyle="--", label="NN manual (extra N)")
    ax_c.plot(valori_n_nn_extra, aima_y_extra, marker="x", linestyle="--", label="NN aima (extra N)")
    ax_c.set_title("Caz c) Y solutii / multistart (Y=N)")
    ax_c.set_xlabel("N (orase)")
    ax_c.set_ylabel("Timp (secunde)")
    ax_c.grid(True, which="both", alpha=0.3)
    ax_c.legend(fontsize=8)

    fig.tight_layout()
    out_path = Path(output_png)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
