"""Simulated Annealing (Python pur) pentru TSP + helper pentru simanneal.

Implementarea e aliniata cu cerintele Lab #08:
- SA clasic cu criteriul Metropolis
- program de racire geometric (T <- alpha * T)
- vecinatate 2-opt
- istorice: cost curent, best, temperatura

Acest modul lucreaza pe matrice de distante NxN (int/float).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple


Number = float
Matrix = List[List[Number]]
Tour = List[int]


def tour_cost(tour: Sequence[int], dist: Matrix) -> Number:
	"""Costul total al turului (incl. revenire la start)."""
	if not tour:
		return 0.0
	cost: Number = 0.0
	n = len(tour)
	for i in range(n - 1):
		cost += dist[tour[i]][tour[i + 1]]
	cost += dist[tour[-1]][tour[0]]
	return cost


def random_tour(n: int, rng: random.Random, *, fix_start: bool = True, start_city: int = 0) -> Tour:
	"""Genereaza un tur aleator (permutare).

	Daca fix_start=True, orasul `start_city` ramane pe prima pozitie (elimina rotatii echivalente).
	"""
	if n <= 0:
		return []
	if n == 1:
		return [start_city]
	cities = list(range(n))
	if fix_start:
		cities.remove(start_city)
		rng.shuffle(cities)
		return [start_city] + cities
	rng.shuffle(cities)
	return cities


def nearest_neighbor_tour(dist: Matrix, *, start: int = 0) -> Tour:
	"""Construieste tur folosind Nearest Neighbor (greedy)."""
	n = len(dist)
	if n == 0:
		return []
	if not (0 <= start < n):
		raise ValueError("start trebuie sa fie in [0, n)")
	if n == 1:
		return [start]

	visited = [False] * n
	visited[start] = True
	tour: Tour = [start]
	cur = start

	for _ in range(n - 1):
		best_j = None
		best_d = None
		for j in range(n):
			if visited[j]:
				continue
			d = dist[cur][j]
			if best_d is None or d < best_d:
				best_d = d
				best_j = j
		if best_j is None:
			break
		visited[int(best_j)] = True
		tour.append(int(best_j))
		cur = int(best_j)

	return tour


def two_opt_neighbor(tour: Sequence[int], rng: random.Random, *, fix_start: bool = True) -> Tour:
	"""Genereaza un vecin 2-opt prin inversarea unui subsegment."""
	n = len(tour)
	if n <= 3:
		return list(tour)

	# Daca fix_start, nu mutam pozitia 0.
	lo = 1 if fix_start else 0
	i, j = sorted(rng.sample(range(lo, n), 2))
	if i == j:
		return list(tour)
	new_tour = list(tour)
	new_tour[i : j + 1] = reversed(new_tour[i : j + 1])
	return new_tour


@dataclass
class SAResult:
	best_tour: Tour
	best_cost: Number
	current_tour: Tour
	current_cost: Number
	cost_history: List[Number] = field(default_factory=list)
	best_history: List[Number] = field(default_factory=list)
	temp_history: List[Number] = field(default_factory=list)
	accepted_history: List[bool] = field(default_factory=list)


@dataclass
class SimulatedAnnealingTSP:
	"""Solver SA (Python pur) pentru TSP pe matrice de distante."""

	dist: Matrix
	t_max: Number = 10000.0
	t_min: Number = 1.0
	alpha: Number = 0.995
	iterations_per_temp: int = 100
	seed: int = 42
	fix_start: bool = True

	def __post_init__(self) -> None:
		if self.t_max <= 0 or self.t_min <= 0:
			raise ValueError("t_max si t_min trebuie sa fie > 0")
		if self.alpha <= 0 or self.alpha >= 1:
			raise ValueError("alpha trebuie sa fie in (0, 1)")
		if self.iterations_per_temp <= 0:
			raise ValueError("iterations_per_temp trebuie sa fie > 0")
		self._rng = random.Random(self.seed)

	@property
	def n(self) -> int:
		return len(self.dist)

	def solve(
		self,
		*,
		init: str = "nn",
		start_city: int = 0,
		max_steps: Optional[int] = None,
	) -> SAResult:
		"""Ruleaza SA si intoarce cea mai buna solutie gasita.

		Args:
			init: "nn" (recomandat) sau "random".
			start_city: orasul de start (folosit cand fix_start=True).
			max_steps: limita hard pe numarul total de mutari evaluate (optional).
		"""
		n = self.n
		if n == 0:
			return SAResult(best_tour=[], best_cost=0.0, current_tour=[], current_cost=0.0)
		if not (0 <= start_city < n):
			raise ValueError("start_city invalid")

		if init == "nn":
			current = nearest_neighbor_tour(self.dist, start=start_city)
		elif init == "random":
			current = random_tour(n, self._rng, fix_start=self.fix_start, start_city=start_city)
		else:
			raise ValueError("init trebuie sa fie 'nn' sau 'random'")

		if self.fix_start and current and current[0] != start_city:
			# normalizeaza
			if start_city in current:
				idx = current.index(start_city)
				current = current[idx:] + current[:idx]

		cur_cost = tour_cost(current, self.dist)
		best = list(current)
		best_cost = cur_cost

		T = float(self.t_max)
		steps_done = 0

		res = SAResult(
			best_tour=best,
			best_cost=float(best_cost),
			current_tour=list(current),
			current_cost=float(cur_cost),
		)

		while T > float(self.t_min):
			for _ in range(self.iterations_per_temp):
				if max_steps is not None and steps_done >= int(max_steps):
					T = 0.0
					break

				candidate = two_opt_neighbor(current, self._rng, fix_start=self.fix_start)
				cand_cost = tour_cost(candidate, self.dist)
				delta = float(cand_cost - cur_cost)

				accepted = False
				if delta <= 0:
					accepted = True
				else:
					# Metropolis
					p = math.exp(-delta / float(T)) if T > 0 else 0.0
					accepted = self._rng.random() < p

				if accepted:
					current = candidate
					cur_cost = float(cand_cost)
					if cur_cost < best_cost:
						best_cost = cur_cost
						best = list(current)

				res.cost_history.append(float(cur_cost))
				res.best_history.append(float(best_cost))
				res.temp_history.append(float(T))
				res.accepted_history.append(bool(accepted))
				steps_done += 1

			T *= float(self.alpha)

		res.best_tour = best
		res.best_cost = float(best_cost)
		res.current_tour = list(current)
		res.current_cost = float(cur_cost)
		return res


# --- Optional: simanneal wrapper -------------------------------------------------

try:
	from simanneal import Annealer  # type: ignore

	class _TSPSimAnneal(Annealer):
		def __init__(self, state: Tour, dist: Matrix):
			self._dist = dist
			super().__init__(state)

		def move(self) -> None:
			# 2-opt in place
			n = len(self.state)
			if n <= 3:
				return
			i, j = sorted(random.sample(range(1, n), 2))
			self.state[i : j + 1] = self.state[i : j + 1][::-1]

		def energy(self) -> float:
			return float(tour_cost(self.state, self._dist))


	def solve_with_simanneal(
		dist: Matrix,
		*,
		init_tour: Optional[Tour] = None,
		t_max: float = 10000.0,
		t_min: float = 1.0,
		steps: int = 50000,
		updates: int = 100,
		seed: int = 42,
	) -> Tuple[Tour, float]:
		"""Rezolva TSP folosind biblioteca `simanneal` (daca e instalata)."""
		n = len(dist)
		random.seed(seed)
		if init_tour is None:
			init_tour = nearest_neighbor_tour(dist, start=0)
		if not init_tour:
			return [], 0.0
		solver = _TSPSimAnneal(list(init_tour), dist)
		solver.Tmax = float(t_max)
		solver.Tmin = float(t_min)
		solver.steps = int(steps)
		solver.updates = int(updates)
		best_tour, best_cost = solver.anneal()
		return list(best_tour), float(best_cost)

except Exception:
	Annealer = None  # type: ignore

	def solve_with_simanneal(*args, **kwargs):  # type: ignore
		raise ImportError("simanneal nu este instalat. Ruleaza: pip install simanneal")
