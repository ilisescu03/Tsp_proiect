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
from .backtracking import rezolva_tsp_backtracking, rezolva_tsp_backtracking_extins
from .hill_climbing_tsp import rezolva_tsp_hc
from .nearest_neighbor import rezolva_tsp_nn, rezolva_tsp_nn_multistart
from .nn_aima import rezolva_tsp_nn_aima, rezolva_tsp_nn_aima_multistart


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
	start = time.perf_counter()
	result = func(*args, **kwargs)
	duration = time.perf_counter() - start
	return duration, result


def ruleaza_experiment(
	output_png: str | Path = "comparare_performanta.png",
	seed: int = 42,
	reporniri_hc: int = 30,
	iteratii_hc: int = 2000,
	bt_time_limit_s: float = 30.0,
) -> Path:
	"""Ruleaza experimentul comparativ si genereaza graficul de performanta.

	Protocol (conform laborator):
		- N pentru backtracking: 5, 7, 8, 10, 12
		- N pentru hill climbing: 5, 7, 8, 10, 12, 15, 20, 30, 50
		- distante intregi in [1, 100], matrice simetrica, seed fix
		- timp masurat cu time.perf_counter
		- grafic: 2 subploturi (liniar + semilogy), salvat ca PNG

	Args:
		output_png: Calea fisierului PNG.
		seed: Seed de baza pentru generarea instanțelor.
		reporniri_hc: Numar reporniri pentru hill climbing.
		iteratii_hc: Limita iteratii per repornire.
		bt_time_limit_s: Daca backtracking depaseste pragul, notam limita si oprim.

	Returns:
		Path catre imaginea PNG generata.
	"""
	valori_n_bt = [5, 7, 8, 10, 12]
	valori_n_hc = [5, 7, 8, 10, 12, 15, 20, 30, 50]

	times_bt: List[float] = []
	times_hc: List[float] = []

	max_n_bt_sub_prag = None

	for n in valori_n_bt:
		rng = random.Random(seed + n)
		matrix = genereaza_instanta_tsp(n, rng)
		duration, _ = _time_call(rezolva_tsp_backtracking, n, matrix)
		times_bt.append(duration)
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
		times_hc.append(duration)

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

	Grafic minim (cerinta): compara timpii de executie pentru 4 valori ale lui N
	(5, 8, 10, 12) pentru:
		- cazul a) prima solutie / un singur start
		- cazul c) Y solutii / multistart cu Y=N

	Include 3 implementari pe acelasi grafic:
		- backtracking (modurile prima / y_solutii)
		- nearest neighbor manual
		- nearest neighbor "aima" (wrapper; poate face fallback)

	Args:
		output_png: Calea fisierului PNG.
		seed: Seed pentru generarea instantelor.
		timp_nn_s: Rezervat pentru extensii (NN timp), pastrat pentru compatibilitate CLI.

	Returns:
		Path catre imaginea PNG generata.
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
		bt_prima.append(d)
		d, _ = _time_call(rezolva_tsp_backtracking_extins, n, matrix, mod="y_solutii", y_max=n)
		bt_y.append(d)

		d, _ = _time_call(rezolva_tsp_nn, n, matrix, 0)
		nn_prima.append(d)
		d, _ = _time_call(rezolva_tsp_nn_multistart, n, matrix)
		nn_y.append(d)

		d, _ = _time_call(rezolva_tsp_nn_aima, n, matrix, 0)
		aima_prima.append(d)
		d, _ = _time_call(rezolva_tsp_nn_aima_multistart, n, matrix)
		aima_y.append(d)

	# Extra N doar pentru NN (protocol lab): il afisam doar pe graficul NN.
	nn_prima_extra: List[float] = []
	nn_y_extra: List[float] = []
	aima_prima_extra: List[float] = []
	aima_y_extra: List[float] = []

	for n in valori_n_nn_extra:
		matrix = genereaza_instanta_tsp(n, random.Random(seed + n))
		d, _ = _time_call(rezolva_tsp_nn, n, matrix, 0)
		nn_prima_extra.append(d)
		d, _ = _time_call(rezolva_tsp_nn_multistart, n, matrix)
		nn_y_extra.append(d)
		d, _ = _time_call(rezolva_tsp_nn_aima, n, matrix, 0)
		aima_prima_extra.append(d)
		d, _ = _time_call(rezolva_tsp_nn_aima_multistart, n, matrix)
		aima_y_extra.append(d)

	if _HAS_SEABORN:
		sns.set_theme()

	fig, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(12, 4.8))

	# Caz a) prima solutie / un start
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

	# Caz c) Y solutii / multistart (Y=N)
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
