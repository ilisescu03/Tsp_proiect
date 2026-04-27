"""Algoritm Genetic (PyGAD) pentru problema comis-voiajorului (TSP).

Lab #09 cere rezolvarea TSP folosind:
- reprezentare: permutare a oraselor
- fitness: PyGAD maximizeaza => fitness = -distanta_totala
- crossover recomandat: Order Crossover (OX) (custom)
- mutatie recomandata: swap (custom)

Acest modul implementeaza solverul + seturi de date (ex. un set mic de orase 2D)
si functii utilitare folosite de vizualizari/studii.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
	import pygad
except Exception as exc:  # pragma: no cover
	raise ImportError("PyGAD nu este instalat. Ruleaza: pip install pygad") from exc


Point = Tuple[float, float]
Tour = List[int]


@dataclass(frozen=True)
class CitySet:
	"""Set de orase cu nume + coordonate 2D."""

	names: List[str]
	coords: np.ndarray  # shape (n, 2)

	@property
	def n(self) -> int:
		return int(self.coords.shape[0])


def default_cityset_ro() -> CitySet:
	"""Un set mic de orase (nume + coordonate) pentru demo/Task1.

	Coordonatele sunt in unitati relative (nu neaparat km) si sunt suficiente
	pentru a vizualiza ruta si a rula GA.
	"""
	data: List[Tuple[str, Point]] = [
		("Cluj-Napoca", (0.0, 0.0)),
		("Brasov", (220.0, -130.0)),
		("Bucuresti", (330.0, -175.0)),
		("Timisoara", (-175.0, -75.0)),
		("Iasi", (380.0, 55.0)),
		("Constanta", (450.0, -225.0)),
		("Craiova", (160.0, -230.0)),
		("Galati", (430.0, -55.0)),
		("Oradea", (-95.0, 45.0)),
		("Sibiu", (95.0, -95.0)),
	]
	_names = [n for n, _ in data]
	_coords = np.array([p for _, p in data], dtype=float)
	return CitySet(names=_names, coords=_coords)


def random_cityset(n: int, *, seed: int = 42, x_range: Tuple[float, float] = (-500, 500), y_range: Tuple[float, float] = (-300, 300)) -> CitySet:
	rng = random.Random(seed)
	names = [f"Oras_{i}" for i in range(n)]
	coords = np.array(
		[(rng.uniform(*x_range), rng.uniform(*y_range)) for _ in range(n)],
		dtype=float,
	)
	return CitySet(names=names, coords=coords)


def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
	"""Matrice dist Euclidiana, shape (n, n)."""
	n = int(coords.shape[0])
	dist = np.zeros((n, n), dtype=float)
	for i in range(n):
		x1, y1 = float(coords[i][0]), float(coords[i][1])
		for j in range(i + 1, n):
			x2, y2 = float(coords[j][0]), float(coords[j][1])
			d = math.hypot(x1 - x2, y1 - y2)
			dist[i][j] = dist[j][i] = float(d)
	return dist


def tour_distance(tour: Sequence[int], dist: np.ndarray) -> float:
	if len(tour) == 0:
		return 0.0
	total = 0.0
	n = len(tour)
	for i in range(n):
		a = int(tour[i])
		b = int(tour[(i + 1) % n])
		total += float(dist[a][b])
	return float(total)


def generate_initial_population(pop_size: int, n_cities: int, rng: random.Random) -> np.ndarray:
	"""Populatie initiala: permutari aleatoare."""
	pop: List[List[int]] = []
	base = list(range(n_cities))
	for _ in range(int(pop_size)):
		perm = base[:]
		rng.shuffle(perm)
		pop.append(perm)
	return np.array(pop, dtype=int)


def ox_crossover(parents: np.ndarray, offspring_size: Tuple[int, int], ga_instance: "pygad.GA") -> np.ndarray:
	"""Order Crossover (OX) pentru permutari.

	Returneaza un numpy array shape offspring_size.
	"""
	offspring: List[List[int]] = []
	idx = 0
	while len(offspring) < int(offspring_size[0]):
		p1 = parents[idx % parents.shape[0]].astype(int).tolist()
		p2 = parents[(idx + 1) % parents.shape[0]].astype(int).tolist()
		n = len(p1)
		cx1, cx2 = sorted(random.sample(range(n), 2))

		child = [-1] * n
		child[cx1 : cx2 + 1] = p1[cx1 : cx2 + 1]
		segment = set(child[cx1 : cx2 + 1])

		remaining = [g for g in p2 if g not in segment]
		free_positions = [i for i in range(n) if child[i] == -1]
		for pos, gene in zip(free_positions, remaining):
			child[pos] = int(gene)

		offspring.append(child)
		idx += 1

	return np.array(offspring, dtype=int)


def swap_mutation(offspring: np.ndarray, ga_instance: "pygad.GA") -> np.ndarray:
	"""Swap mutation pe cromozomi permutare.

	In PyGAD, mutation_percent_genes este folosit aici ca probabilitatea (0-100)
	ca un cromozom sa fie mutat (asa cum e descris in cerinta).
	"""
	rate = float(getattr(ga_instance, "mutation_percent_genes", 0)) / 100.0
	for i in range(int(offspring.shape[0])):
		if random.random() < rate:
			n = int(offspring.shape[1])
			idx1, idx2 = random.sample(range(n), 2)
			offspring[i][idx1], offspring[i][idx2] = offspring[i][idx2], offspring[i][idx1]
	return offspring


@dataclass
class GAResult:
	best_tour: Tour
	best_distance: float
	duration_s: float
	best_distances_by_generation: List[float]
	ga_instance: "pygad.GA"


def run_ga(
	cityset: CitySet,
	*,
	pop_size: int = 100,
	n_generations: int = 500,
	mutation_rate_percent: int = 50,
	parent_selection_type: str = "tournament",
	k_tournament: int = 3,
	keep_elitism: int = 2,
	crossover_probability: float = 1.0,
	seed: int = 42,
	verbose: bool = False,
) -> GAResult:
	"""Configureaza si ruleaza GA pentru un CitySet."""

	rng = random.Random(seed)
	np.random.seed(seed)
	random.seed(seed)

	dist = build_distance_matrix(cityset.coords)

	def fitness_func(ga_instance: "pygad.GA", solution: np.ndarray, solution_idx: int) -> float:
		return -tour_distance(solution.astype(int).tolist(), dist)

	initial_population = generate_initial_population(pop_size, cityset.n, rng)

	ga = pygad.GA(
		num_generations=int(n_generations),
		num_parents_mating=max(2, int(pop_size) // 2),
		fitness_func=fitness_func,
		initial_population=initial_population,
		crossover_type=ox_crossover,
		mutation_type=swap_mutation,
		mutation_percent_genes=int(mutation_rate_percent),
		parent_selection_type=str(parent_selection_type),
		K_tournament=int(k_tournament),
		keep_elitism=int(keep_elitism),
		keep_parents=0,
		crossover_probability=float(crossover_probability),
		suppress_warnings=True,
	)

	start = time.perf_counter()
	ga.run()
	duration = time.perf_counter() - start

	solution, fitness, _ = ga.best_solution()
	best_distance = float(-fitness)
	best_tour = [int(x) for x in solution]

	# Convergenta: PyGAD pastreaza fitness-ul best per generatie; convertim in distante.
	best_distances = [float(-f) for f in ga.best_solutions_fitness]

	if verbose:
		route_names = " -> ".join(cityset.names[int(i)] for i in best_tour)
		print(route_names + f" -> {cityset.names[int(best_tour[0])]}")
		print(f"Distanta: {best_distance:.2f}")
		print(f"Timp: {duration:.3f}s")

	return GAResult(
		best_tour=best_tour,
		best_distance=best_distance,
		duration_s=float(duration),
		best_distances_by_generation=best_distances,
		ga_instance=ga,
	)
