"""Vizualizari si experimente cerute in Lab #09 (Algoritmi genetici pentru TSP).

Genereaza PNG-uri (fara a deschide ferestre interactive), ca sa fie usor de rulat din CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from .genetic_algorithm_tsp import CitySet, GAResult, default_cityset_ro, random_cityset, run_ga, tour_distance, build_distance_matrix


def plot_convergence(distances_best: Sequence[float], *, title: str, output_png: Path) -> Path:
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(list(distances_best), color="steelblue", linewidth=1.5, label="Best")
	ax.set_xlabel("Generație")
	ax.set_ylabel("Distanță totală")
	ax.set_title(title)
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_png, dpi=150)
	plt.close(fig)
	return output_png


def plot_route(cityset: CitySet, tour: Sequence[int], *, title: str, output_png: Path) -> Path:
	coords = cityset.coords
	route = list(map(int, tour))
	if route:
		route = route + [route[0]]

	fig, ax = plt.subplots(figsize=(12, 9))

	# Draw arrows between consecutive cities
	for i in range(len(route) - 1):
		a = route[i]
		b = route[i + 1]
		x1, y1 = float(coords[a][0]), float(coords[a][1])
		x2, y2 = float(coords[b][0]), float(coords[b][1])
		ax.annotate(
			"",
			xy=(x2, y2),
			xytext=(x1, y1),
			arrowprops=dict(arrowstyle="->", color="steelblue", lw=2),
		)
		mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
		ax.text(mx, my, str(i + 1), fontsize=7, color="gray", ha="center")

	ax.scatter(coords[:, 0], coords[:, 1], s=150, c="tomato", zorder=5)
	for idx, name in enumerate(cityset.names):
		x, y = float(coords[idx][0]), float(coords[idx][1])
		ax.annotate(name, (x, y), textcoords="offset points", xytext=(10, 5), fontsize=9)

	ax.set_title(title)
	ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(output_png, dpi=150)
	plt.close(fig)
	return output_png


@dataclass
class TaskOutputs:
	artifacts: List[Path]
	metrics: Dict[str, float]


def task1_primary_run(*, outdir: Path, seed: int = 42) -> TaskOutputs:
	"""Sarcina 1: rulare primara + ruta + convergenta."""
	outdir.mkdir(parents=True, exist_ok=True)
	cityset = default_cityset_ro()
	res = run_ga(cityset, pop_size=100, n_generations=500, mutation_rate_percent=50, seed=seed)

	artifacts: List[Path] = []
	artifacts.append(plot_route(cityset, res.best_tour, title=f"Lab9 Task1 - Ruta GA (dist={res.best_distance:.2f})", output_png=outdir / "task1_route.png"))
	artifacts.append(plot_convergence(res.best_distances_by_generation, title="Lab9 Task1 - Curba de convergenta", output_png=outdir / "task1_convergence.png"))

	route_str = " -> ".join(cityset.names[i] for i in res.best_tour) + f" -> {cityset.names[res.best_tour[0]]}"
	(outdir / "task1_report.txt").write_text(
		"\n".join(
			[
				"Sarcina 1 - Rulare primara (PyGAD)",
				f"Ruta: {route_str}",
				f"Distanta: {res.best_distance:.4f}",
				f"Timp (s): {res.duration_s:.4f}",
			]
		)
		+ "\n",
		encoding="utf-8",
	)

	metrics = {"best_distance": float(res.best_distance), "duration_s": float(res.duration_s)}
	return TaskOutputs(artifacts=artifacts, metrics=metrics)


def task2_population_study(*, outdir: Path, seed: int = 42) -> TaskOutputs:
	"""Sarcina 2: sol_per_pop in [20, 50, 100, 200] (gen=300, mut=40)."""
	outdir.mkdir(parents=True, exist_ok=True)
	cityset = default_cityset_ro()
	pops = [20, 50, 100, 200]
	colors = ["tomato", "steelblue", "seagreen", "darkorange"]

	results: List[Tuple[int, GAResult]] = []
	for pop in pops:
		res = run_ga(cityset, pop_size=pop, n_generations=300, mutation_rate_percent=40, seed=seed)
		results.append((pop, res))

	# Plot 1: curbe suprapuse
	fig, ax = plt.subplots(figsize=(12, 6))
	for (pop, res), color in zip(results, colors):
		ax.plot(res.best_distances_by_generation, color=color, linewidth=1.5, label=f"pop={pop}")
	ax.set_xlabel("Generație")
	ax.set_ylabel("Distanță totală")
	ax.set_title("Task2 - Curbe de convergenta (populatie)")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	p_conv = outdir / "task2_convergence_overlay.png"
	fig.savefig(p_conv, dpi=150)
	plt.close(fig)

	# Plot 2/3: bar distance + bar time
	dist_final = [r.best_distance for _, r in results]
	times = [r.duration_s for _, r in results]
	labels = [str(pop) for pop, _ in results]

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
	ax1.bar(labels, dist_final, color=colors, alpha=0.85)
	ax1.set_xlabel("sol_per_pop")
	ax1.set_ylabel("Distanță finală")
	ax1.set_title("Task2 - Distanță finală vs populație")
	ax1.grid(axis="y", alpha=0.3)

	ax2.bar(labels, times, color=colors, alpha=0.85)
	ax2.set_xlabel("sol_per_pop")
	ax2.set_ylabel("Timp (s)")
	ax2.set_title("Task2 - Timp execuție vs populație")
	ax2.grid(axis="y", alpha=0.3)

	fig.tight_layout()
	p_bars = outdir / "task2_bars.png"
	fig.savefig(p_bars, dpi=150)
	plt.close(fig)

	artifacts = [p_conv, p_bars]
	metrics = {
		"best_distance_min": float(min(dist_final)),
		"duration_s_min": float(min(times)),
		"duration_s_max": float(max(times)),
	}
	return TaskOutputs(artifacts=artifacts, metrics=metrics)


def task3_mutation_study(*, outdir: Path, seed: int = 42) -> TaskOutputs:
	"""Sarcina 3: mutatie in [5, 20, 40, 60, 80, 95] (pop=100, gen=300)."""
	outdir.mkdir(parents=True, exist_ok=True)
	cityset = default_cityset_ro()
	muts = [5, 20, 40, 60, 80, 95]
	colors = ["navy", "royalblue", "steelblue", "seagreen", "darkorange", "tomato"]

	results: List[Tuple[int, GAResult]] = []
	for mut in muts:
		res = run_ga(cityset, pop_size=100, n_generations=300, mutation_rate_percent=mut, seed=seed)
		results.append((mut, res))

	# Overlay convergence
	fig, ax = plt.subplots(figsize=(12, 6))
	for (mut, res), color in zip(results, colors):
		ax.plot(res.best_distances_by_generation, color=color, linewidth=1.5, label=f"mut={mut}%")
	ax.set_xlabel("Generație")
	ax.set_ylabel("Distanță totală")
	ax.set_title("Task3 - Curbe de convergenta (mutație)")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	p_conv = outdir / "task3_convergence_overlay.png"
	fig.savefig(p_conv, dpi=150)
	plt.close(fig)

	# Distance vs mutation
	x = [mut for mut, _ in results]
	y = [res.best_distance for _, res in results]
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(x, y, "o-", color="steelblue")
	ax.set_xlabel("Rata de mutație (%)")
	ax.set_ylabel("Distanță finală")
	ax.set_title("Task3 - Distanță finală vs mutație")
	ax.grid(True, alpha=0.3)
	fig.tight_layout()
	p_dist = outdir / "task3_distance_vs_mutation.png"
	fig.savefig(p_dist, dpi=150)
	plt.close(fig)

	artifacts = [p_conv, p_dist]
	metrics = {"best_distance_min": float(min(y)), "best_distance_max": float(max(y))}
	return TaskOutputs(artifacts=artifacts, metrics=metrics)


def task4_selection_study(*, outdir: Path, seed: int = 42) -> TaskOutputs:
	"""Sarcina 4: compară strategii selecție [tournament, rws, rank, sus]."""
	outdir.mkdir(parents=True, exist_ok=True)
	cityset = default_cityset_ro()
	strategies = ["tournament", "rws", "rank", "sus"]
	colors = ["steelblue", "tomato", "seagreen", "darkorange"]

	results: List[Tuple[str, GAResult]] = []
	for s in strategies:
		res = run_ga(
			cityset,
			pop_size=100,
			n_generations=300,
			mutation_rate_percent=40,
			parent_selection_type=s,
			seed=seed,
		)
		results.append((s, res))

	# Overlay convergence
	fig, ax = plt.subplots(figsize=(12, 6))
	for (s, res), color in zip(results, colors):
		ax.plot(res.best_distances_by_generation, color=color, linewidth=1.5, label=s)
	ax.set_xlabel("Generație")
	ax.set_ylabel("Distanță totală")
	ax.set_title("Task4 - Curbe de convergenta (selecție părinți)")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	p_conv = outdir / "task4_convergence_overlay.png"
	fig.savefig(p_conv, dpi=150)
	plt.close(fig)

	# Table report
	lines = ["Strategie\tDistanta_finala\tTimp_s"]
	for s, res in results:
		lines.append(f"{s}\t{res.best_distance:.4f}\t{res.duration_s:.4f}")
	(outdir / "task4_results.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

	artifacts = [p_conv, outdir / "task4_results.tsv"]
	metrics = {"best_distance_min": float(min(r.best_distance for _, r in results))}
	return TaskOutputs(artifacts=artifacts, metrics=metrics)


def task5_scalability(*, outdir: Path, seed: int = 42, pop_size: int = 150, n_generations: int = 400, mutation_rate_percent: int = 40) -> TaskOutputs:
	"""Sarcina 5: scalabilitate pe N in {15, 20, 25} (orase random)."""
	outdir.mkdir(parents=True, exist_ok=True)
	sizes = [15, 20, 25]

	final_distances: List[float] = []
	times: List[float] = []

	for n in sizes:
		cityset = random_cityset(n, seed=seed + n)
		res = run_ga(
			cityset,
			pop_size=pop_size,
			n_generations=n_generations,
			mutation_rate_percent=mutation_rate_percent,
			seed=seed,
		)
		final_distances.append(float(res.best_distance))
		times.append(float(res.duration_s))

	# Plot distance vs N
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(sizes, final_distances, "o-", color="steelblue")
	ax.set_xlabel("N (număr de orașe)")
	ax.set_ylabel("Distanță finală")
	ax.set_title("Task5 - Distanță finală vs N")
	ax.grid(True, alpha=0.3)
	fig.tight_layout()
	p_dist = outdir / "task5_distance_vs_n.png"
	fig.savefig(p_dist, dpi=150)
	plt.close(fig)

	# Plot time vs N
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(sizes, times, "o-", color="darkorange")
	ax.set_xlabel("N (număr de orașe)")
	ax.set_ylabel("Timp (s)")
	ax.set_title("Task5 - Timp execuție vs N")
	ax.grid(True, alpha=0.3)
	fig.tight_layout()
	p_time = outdir / "task5_time_vs_n.png"
	fig.savefig(p_time, dpi=150)
	plt.close(fig)

	# Report
	lines = ["N\tDistanta_finala\tTimp_s"]
	for n, d, t in zip(sizes, final_distances, times):
		lines.append(f"{n}\t{d:.4f}\t{t:.4f}")
	(outdir / "task5_results.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

	artifacts = [p_dist, p_time, outdir / "task5_results.tsv"]
	metrics = {"duration_s_max": float(max(times))}
	return TaskOutputs(artifacts=artifacts, metrics=metrics)


def run_lab9(*, mode: str, outdir: Path, seed: int = 42) -> TaskOutputs:
	mode = mode.lower().strip()
	if mode == "task1":
		return task1_primary_run(outdir=outdir, seed=seed)
	if mode == "task2":
		return task2_population_study(outdir=outdir, seed=seed)
	if mode == "task3":
		return task3_mutation_study(outdir=outdir, seed=seed)
	if mode == "task4":
		return task4_selection_study(outdir=outdir, seed=seed)
	if mode == "task5":
		return task5_scalability(outdir=outdir, seed=seed)
	raise ValueError("mode invalid. Alege: task1/task2/task3/task4/task5")
