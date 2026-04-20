"""Vizualizari cerute in Lab #08 pentru SA + TSP.

V1 - Traseul TSP (2D)
V2 - Evolutia costului (curent + best)
V3 - Programul de racire (temperatura vs iteratie, scala log)
V4 - Probabilitatea de acceptare Metropolis
V5 - Comparație NN vs SA (doua subploturi)
V6 - Timp de execuție vs N (simanneal vs implementare proprie)
V7 - Heatmap distante

Acestea sunt folosite de subcomanda CLI `lab8`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

try:
	import seaborn as sns  # type: ignore
	_HAS_SEABORN = True
except Exception:
	_HAS_SEABORN = False

from .simulated_annealing_tsp import (
	Matrix,
	SAResult,
	SimulatedAnnealingTSP,
	Tour,
	nearest_neighbor_tour,
	solve_with_simanneal,
	tour_cost,
)


Point = Tuple[float, float]


def generate_cities(n: int, *, seed: int = 42, low: float = 0.0, high: float = 100.0) -> List[Point]:
	rng = random.Random(seed)
	return [(rng.uniform(low, high), rng.uniform(low, high)) for _ in range(n)]


def build_distance_matrix(cities: Sequence[Point]) -> Matrix:
	n = len(cities)
	dist: Matrix = [[0.0] * n for _ in range(n)]
	for i in range(n):
		x1, y1 = cities[i]
		for j in range(i + 1, n):
			x2, y2 = cities[j]
			d = math.hypot(x1 - x2, y1 - y2)
			dist[i][j] = dist[j][i] = float(d)
	return dist


def plot_tour(cities: Sequence[Point], tour: Sequence[int], *, title: str, output_png: Path) -> Path:
	coords = [cities[i] for i in tour] + [cities[tour[0]]] if tour else []
	xs, ys = zip(*coords) if coords else ([], [])

	fig, ax = plt.subplots(figsize=(8, 6))
	ax.plot(xs, ys, "b-o", markersize=6, linewidth=1.5)
	for idx, (x, y) in enumerate(cities):
		ax.annotate(str(idx), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
	ax.set_title(title)
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(output_png, dpi=150)
	plt.close(fig)
	return output_png


def plot_cost_history(
	cost_history: Sequence[float],
	best_history: Sequence[float],
	*,
	title: str,
	output_png: Path,
) -> Path:
	fig, ax = plt.subplots(figsize=(9, 4.5))
	x = list(range(1, len(cost_history) + 1))
	ax.plot(x, cost_history, color="tab:blue", alpha=0.35, linewidth=1.0, label="cost curent")
	ax.plot(x, best_history, color="tab:green", linewidth=2.0, label="best")
	ax.set_title(title)
	ax.set_xlabel("Iteratie")
	ax.set_ylabel("Cost")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_png, dpi=150)
	plt.close(fig)
	return output_png


def plot_temperature_schedule(temp_history: Sequence[float], *, title: str, output_png: Path) -> Path:
	fig, ax = plt.subplots(figsize=(9, 4.5))
	x = list(range(1, len(temp_history) + 1))
	ax.semilogy(x, temp_history, color="tab:orange", linewidth=1.5)
	ax.set_title(title)
	ax.set_xlabel("Iteratie")
	ax.set_ylabel("Temperatura (log)")
	ax.grid(True, which="both", alpha=0.3)
	fig.tight_layout()
	fig.savefig(output_png, dpi=150)
	plt.close(fig)
	return output_png


def plot_acceptance_probability(
	*,
	temps: Sequence[float],
	delta_max: float = 200.0,
	points: int = 200,
	title: str,
	output_png: Path,
) -> Path:
	fig, ax = plt.subplots(figsize=(9, 4.5))
	deltas = [delta_max * i / (points - 1) for i in range(points)]
	for T in temps:
		ps = [math.exp(-d / T) if T > 0 else 0.0 for d in deltas]
		ax.plot(deltas, ps, label=f"T={T}")
	ax.set_title(title)
	ax.set_xlabel("ΔE (deteriorare)")
	ax.set_ylabel("P(acceptare)")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_png, dpi=150)
	plt.close(fig)
	return output_png


def plot_comparison(
	cities: Sequence[Point],
	tour_nn: Sequence[int],
	tour_sa: Sequence[int],
	*,
	cost_nn: float,
	cost_sa: float,
	output_png: Path,
) -> Path:
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

	def _draw(ax, tour: Sequence[int], title: str) -> None:
		coords = [cities[i] for i in tour] + [cities[tour[0]]] if tour else []
		xs, ys = zip(*coords) if coords else ([], [])
		ax.plot(xs, ys, "b-o", markersize=5, linewidth=1.2)
		for idx, (x, y) in enumerate(cities):
			ax.annotate(str(idx), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
		ax.set_title(title)
		ax.grid(True, alpha=0.3)

	_draw(ax1, tour_nn, f"Nearest Neighbor\nCost={cost_nn:.2f}")
	_draw(ax2, tour_sa, f"Simulated Annealing\nCost={cost_sa:.2f}")

	fig.tight_layout()
	fig.savefig(output_png, dpi=150)
	plt.close(fig)
	return output_png


def plot_distance_heatmap(dist: Matrix, *, title: str, output_png: Path) -> Path:
	if _HAS_SEABORN:
		sns.set_theme()
		fig, ax = plt.subplots(figsize=(7, 6))
		annot = len(dist) <= 15
		sns.heatmap(dist, annot=annot, fmt=".0f", cmap="YlOrRd", ax=ax)
		ax.set_title(title)
		fig.tight_layout()
		fig.savefig(output_png, dpi=150)
		plt.close(fig)
		return output_png

	# Fallback fara seaborn
	fig, ax = plt.subplots(figsize=(7, 6))
	im = ax.imshow(dist, cmap="YlOrRd")
	ax.set_title(title)
	fig.colorbar(im, ax=ax)
	fig.tight_layout()
	fig.savefig(output_png, dpi=150)
	plt.close(fig)
	return output_png


@dataclass
class BenchmarkResult:
	sizes: List[int]
	times_lib: List[float]
	times_own: List[float]
	costs_lib: List[float]
	costs_own: List[float]


def benchmark_simanneal_vs_own(
	*,
	sizes: Sequence[int] = (8, 12, 20, 25),
	seed: int = 42,
	# simanneal
	sim_tmax: float = 10000.0,
	sim_tmin: float = 1.0,
	sim_steps: int = 50000,
	# own
	own_tmax: float = 10000.0,
	own_tmin: float = 1.0,
	own_alpha: float = 0.995,
	own_iters_per_temp: int = 100,
) -> BenchmarkResult:
	times_lib: List[float] = []
	times_own: List[float] = []
	costs_lib: List[float] = []
	costs_own: List[float] = []

	for n in sizes:
		cities = generate_cities(int(n), seed=seed + int(n))
		dist = build_distance_matrix(cities)
		init = nearest_neighbor_tour(dist, start=0)

		# simanneal
		start = time.perf_counter()
		tour_lib, cost_lib = solve_with_simanneal(
			dist,
			init_tour=init,
			t_max=sim_tmax,
			t_min=sim_tmin,
			steps=sim_steps,
			seed=seed,
		)
		times_lib.append(time.perf_counter() - start)
		costs_lib.append(float(cost_lib))

		# own
		start = time.perf_counter()
		sa = SimulatedAnnealingTSP(
			dist,
			t_max=own_tmax,
			t_min=own_tmin,
			alpha=own_alpha,
			iterations_per_temp=own_iters_per_temp,
			seed=seed,
			fix_start=True,
		)
		res = sa.solve(init="nn")
		times_own.append(time.perf_counter() - start)
		costs_own.append(float(res.best_cost))

	return BenchmarkResult(list(map(int, sizes)), times_lib, times_own, costs_lib, costs_own)


def plot_benchmark(result: BenchmarkResult, *, title: str, output_png: Path) -> Path:
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
	x = list(range(len(result.sizes)))
	width = 0.35

	ax1.bar([i - width / 2 for i in x], result.times_lib, width, label="simanneal")
	ax1.bar([i + width / 2 for i in x], result.times_own, width, label="Python pur")
	ax1.set_xticks(x)
	ax1.set_xticklabels([str(n) for n in result.sizes])
	ax1.set_xlabel("N (numar de orase)")
	ax1.set_ylabel("Timp (s)")
	ax1.set_title("Timp executie vs N")
	ax1.legend()
	ax1.grid(axis="y", alpha=0.3)

	ax2.plot(result.sizes, result.costs_lib, "o-", label="simanneal")
	ax2.plot(result.sizes, result.costs_own, "s--", label="Python pur")
	ax2.set_xlabel("N")
	ax2.set_ylabel("Cost tur")
	ax2.set_title("Calitatea solutiei vs N")
	ax2.legend()
	ax2.grid(True, alpha=0.3)

	fig.suptitle(title, fontweight="bold")
	fig.tight_layout()
	fig.savefig(output_png, dpi=150)
	plt.close(fig)
	return output_png


def run_lab8_bundle(
	*,
	n: int = 20,
	seed: int = 42,
	outdir: Path,
	# SA own params
	t_max: float = 10000.0,
	t_min: float = 1.0,
	alpha: float = 0.995,
	iterations_per_temp: int = 100,
	# simanneal
	sim_steps: int = 50000,
) -> List[Path]:
	"""Genereaza toate graficele V1–V7 intr-un folder."""
	outdir.mkdir(parents=True, exist_ok=True)
	if _HAS_SEABORN:
		sns.set_theme()

	cities = generate_cities(n, seed=seed)
	dist = build_distance_matrix(cities)

	# NN (init + baseline)
	tour_nn = nearest_neighbor_tour(dist, start=0)
	cost_nn = float(tour_cost(tour_nn, dist))

	# SA own
	sa = SimulatedAnnealingTSP(
		dist,
		t_max=t_max,
		t_min=t_min,
		alpha=alpha,
		iterations_per_temp=iterations_per_temp,
		seed=seed,
		fix_start=True,
	)
	res = sa.solve(init="nn")
	tour_sa = res.best_tour
	cost_sa = float(res.best_cost)

	artifacts: List[Path] = []

	# V1 - traseu (tur optim SA)
	artifacts.append(plot_tour(cities, tour_sa, title=f"V1 - Tur optim SA (cost={cost_sa:.2f})", output_png=outdir / "v1_tour_sa.png"))

	# V2 - evolutie cost
	artifacts.append(
		plot_cost_history(
			list(map(float, res.cost_history)),
			list(map(float, res.best_history)),
			title="V2 - Evolutia costului (curent vs best)",
			output_png=outdir / "v2_cost_history.png",
		)
	)

	# V3 - racire
	artifacts.append(
		plot_temperature_schedule(
			list(map(float, res.temp_history)),
			title="V3 - Programul de racire (temperatura)",
			output_png=outdir / "v3_temperature_schedule.png",
		)
	)

	# V4 - acceptare Metropolis
	artifacts.append(
		plot_acceptance_probability(
			temps=[5000, 1000, 200, 50, 10],
			delta_max=200.0,
			title="V4 - Probabilitatea de acceptare Metropolis",
			output_png=outdir / "v4_acceptance_probability.png",
		)
	)

	# V5 - comparatie NN vs SA
	artifacts.append(
		plot_comparison(
			cities,
			tour_nn,
			tour_sa,
			cost_nn=cost_nn,
			cost_sa=cost_sa,
			output_png=outdir / "v5_nn_vs_sa.png",
		)
	)

	# V6 - benchmark simanneal vs own
	bench = benchmark_simanneal_vs_own(
		sizes=(8, 12, 20, 25),
		seed=seed,
		sim_tmax=t_max,
		sim_tmin=t_min,
		sim_steps=sim_steps,
		own_tmax=t_max,
		own_tmin=t_min,
		own_alpha=alpha,
		own_iters_per_temp=iterations_per_temp,
	)
	artifacts.append(
		plot_benchmark(bench, title="V6 - Comparație simanneal vs Python pur", output_png=outdir / "v6_benchmark.png")
	)

	# V7 - heatmap distante
	artifacts.append(
		plot_distance_heatmap(
			dist,
			title="V7 - Heatmap distante",
			output_png=outdir / "v7_distance_heatmap.png",
		)
	)

	return artifacts
