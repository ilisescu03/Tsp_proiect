from __future__ import annotations

import random
from typing import List, Optional

Matrix = List[List[int]]

def _cost_tur(route: List[int], matrice: Matrix) -> int:
    "Cost total tur (incl. return to start)."
    n = len(route)
    if n == 0:
        return 0
    cost = 0
    for i in range(n - 1):
        cost += matrice[route[i]][route[i + 1]]
    cost += matrice[route[-1]][route[0]]
    return cost

def genereaza_tur_initial_aleator(n: int, seed: Optional[int] = None) -> List[int]:
    "Random perm, fix city 0 at start."
    if seed is not None:
        random.seed(seed)
    cities = list(range(1, n))
    random.shuffle(cities)
    return [0] + cities

def rezolva_tsp_nn(n: int, matrice: Matrix, seed: Optional[int] = None) -> tuple[List[int], int]:
    "Base Nearest Neighbor (single random start)."
    if seed is not None:
        random.seed(seed)
    route = genereaza_tur_initial_aleator(n, seed)
    visited = [False] * n
    visited[0] = True
    current = 0
    
    for _ in range(1, n):
        min_dist = float('inf')
        next_city = -1
        for city in range(n):
            if not visited[city] and matrice[current][city] < min_dist:
                min_dist = matrice[current][city]
                next_city = city
        if next_city == -1:
            break
        route.append(next_city)
        visited[next_city] = True
        current = next_city
    
    cost = _cost_tur(route, matrice)
    return route, cost

def rezolva_tsp_nn_multistart(n: int, matrice: Matrix, Y: Optional[int] = None, seed: Optional[int] = None) -> tuple[List[int], int]:
    "Multistart NN: Y=N random starts, return best."
    if Y is None:
        Y = n
    if seed is not None:
        random.seed(seed)
    
    best_route: List[int] = []
    best_cost = float('inf')
    
    for i in range(Y):
        sub_seed = seed + i if seed else None
        route, cost = rezolva_tsp_nn(n, matrice, sub_seed)
        if cost < best_cost:
            best_cost = cost
            best_route = route
    
    return best_route, int(best_cost)
"""Euristica celui mai apropiat vecin (Nearest Neighbor) pentru TSP.

Algoritmul NN este o cautare informata constructiva (greedy): construieste turul
pas cu pas, alegand la fiecare iteratie cel mai apropiat oras nevizitat.

Functiile din acest modul sunt cerute in Lab #04 si folosesc acelasi format
(traseu, cost) ca celelalte implementari pentru a permite comparatii directe.
"""

from __future__ import annotations

import random
import time
from typing import List, Sequence, Tuple


Matrix = List[List[int]]


def _cost_tur(traseu: Sequence[int], matrice: Matrix) -> int:
	"""Calculeaza costul turului, incluzand revenirea la orasul de start."""
	if not traseu:
		return 0
	cost = 0
	for i in range(len(traseu) - 1):
		cost += matrice[traseu[i]][traseu[i + 1]]
	cost += matrice[traseu[-1]][traseu[0]]
	return cost


def rezolva_tsp_nn(n: int, matrice: Matrix, start: int = 0) -> Tuple[List[int], int]:
	"""Construieste un tur TSP folosind NN, pornind dintr-un oras dat.

	Args:
		n: Numarul de orase.
		matrice: Matricea de distante NxN.
		start: Orasul de start.

	Returns:
		(traseu, cost) unde `traseu` este ordinea vizitarii (incepe cu `start`).

	Raises:
		ValueError: Daca n/matrice/start sunt invalide.
	"""
	if n <= 0:
		raise ValueError("n trebuie sa fie > 0")
	if len(matrice) != n or any(len(row) != n for row in matrice):
		raise ValueError("matrice trebuie sa fie NxN")
	if not (0 <= start < n):
		raise ValueError("start trebuie sa fie in [0, n)")
	if n == 1:
		return [start], 0

	vizitat = [False] * n
	vizitat[start] = True
	traseu = [start]
	oras_curent = start

	for _ in range(n - 1):
		best_next = None
		best_dist = None
		for j in range(n):
			if vizitat[j]:
				continue
			d = matrice[oras_curent][j]
			if best_dist is None or d < best_dist or (d == best_dist and j < int(best_next)):
				best_dist = d
				best_next = j

		if best_next is None:
			break

		vizitat[int(best_next)] = True
		traseu.append(int(best_next))
		oras_curent = int(best_next)

	return traseu, _cost_tur(traseu, matrice)


def rezolva_tsp_nn_multistart(
	n: int, matrice: Matrix
) -> Tuple[List[int], int, List[Tuple[int, List[int], int]]]:
	"""Ruleaza NN din toate punctele de start si pastreaza cel mai bun tur.

	Args:
		n: Numarul de orase.
		matrice: Matricea NxN.

	Returns:
		(best_traseu, best_cost, rezultate)

	Unde `rezultate` este o lista de tuple (start, traseu, cost) pentru fiecare start.
	"""
	best_route: List[int] = []
	best_cost = None
	results: List[Tuple[int, List[int], int]] = []

	for start in range(n):
		route, cost = rezolva_tsp_nn(n, matrice, start=start)
		results.append((start, route, cost))
		if best_cost is None or cost < best_cost:
			best_cost = cost
			best_route = route

	return best_route, int(best_cost if best_cost is not None else 0), results


def rezolva_tsp_nn_timp(
	n: int,
	matrice: Matrix,
	timp_max: float,
	*,
	seed: int = 42,
) -> Tuple[List[int], int, int, float]:
	"""Cauta o solutie NN mai buna in limita de timp (multistart aleator).

	Ruleaza NN repetat din starturi alese aleator (uniform) pana expira `timp_max`.
	Pastreaza cel mai bun tur gasit.

	Args:
		n: Numarul de orase.
		matrice: Matrice NxN.
		timp_max: Limita de timp in secunde.
		seed: Seed pentru reproductibilitate.

	Returns:
		(best_traseu, best_cost, rulari, durata_secunde)
	"""
	if timp_max <= 0:
		raise ValueError("timp_max trebuie sa fie > 0")

	start_time = time.perf_counter()
	deadline = start_time + float(timp_max)
	rng = random.Random(seed)

	# Initializam cu o solutie determinista (start=0) ca fallback.
	best_route, best_cost = rezolva_tsp_nn(n, matrice, start=0)
	runs = 1

	while True:
		if time.perf_counter() >= deadline:
			break
		start_city = rng.randrange(0, n)
		route, cost = rezolva_tsp_nn(n, matrice, start=start_city)
		runs += 1
		if cost < best_cost:
			best_cost = cost
			best_route = route

	duration = time.perf_counter() - start_time
	return best_route, int(best_cost), int(runs), duration
