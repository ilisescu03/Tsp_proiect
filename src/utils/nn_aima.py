"""Nearest Neighbor pentru TSP folosind biblioteca AIMA (cand este disponibila).

In enuntul Lab #04 se recomanda utilizarea `aima3` pentru NN. In practica,
versiunile PyPI ale `aima3` pot sa nu includa o functie NN dedicata pentru TSP.
Acest modul expune o interfata stabila (wrapper) si:
  - foloseste `aima3` daca exista o functie compatibila;
  - altfel face fallback la implementarea manuala NN.

Scop: proiectul sa fie executabil si comparabil cu backtracking/NN manual.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .nearest_neighbor import rezolva_tsp_nn, rezolva_tsp_nn_multistart


Matrix = List[List[int]]


def _incearca_gaseste_nn_aima() -> Optional[Callable]:
	"""Incearca sa localizeze o functie NN TSP in `aima3`.

	Returns:
		Callable daca exista ceva compatibil, altfel None.
	"""
	try:
		from aima3 import search  # type: ignore
	except Exception:
		return None

	# Unele materiale mentioneaza `nearest_neighbor_tsp`, dar poate lipsi.
	cand = getattr(search, "nearest_neighbor_tsp", None)
	if callable(cand):
		return cand
	return None


_NEAREST_NEIGHBOR_TSP = _incearca_gaseste_nn_aima()


def rezolva_tsp_nn_aima(n: int, matrice: Matrix, start: int = 0) -> Tuple[List[int], int]:
	"""Wrapper NN TSP bazat pe AIMA (sau fallback manual).

	Args:
		n: Numarul de orase.
		matrice: Matricea de distante NxN.
		start: Orasul de start.

	Returns:
		(traseu, cost)
	"""
	# Daca exista backend AIMA, am avea nevoie si de un adapter pentru parametri.
	# In absenta unei semnaturi stabile in pachetul instalat, folosim fallback.
	if _NEAREST_NEIGHBOR_TSP is None:
		return rezolva_tsp_nn(n, matrice, start=start)

	# Best-effort adapter: daca proiectul tau are o versiune AIMA care expune
	# nearest_neighbor_tsp(start, cities, distances) unde distances(i, j) -> cost.
	cities = list(range(n))

	def distances(i: int, j: int) -> int:
		return matrice[i][j]

	try:
		route = list(_NEAREST_NEIGHBOR_TSP(start, cities, distances))
		# Ne asiguram ca ruta incepe cu start.
		if route and route[0] != start and start in route:
			k = route.index(start)
			route = route[k:] + route[:k]
		# Costul turului.
		cost = 0
		for idx in range(len(route) - 1):
			cost += matrice[route[idx]][route[idx + 1]]
		cost += matrice[route[-1]][route[0]]
		return route, int(cost)
	except Exception:
		# Fallback robust.
		return rezolva_tsp_nn(n, matrice, start=start)


def rezolva_tsp_nn_aima_multistart(
	n: int, matrice: Matrix
) -> Tuple[List[int], int, List[Tuple[int, List[int], int]]]:
	"""Ruleaza NN AIMA (sau fallback) din toate punctele de start."""
	if _NEAREST_NEIGHBOR_TSP is None:
		return rezolva_tsp_nn_multistart(n, matrice)

	best_route: List[int] = []
	best_cost: int | None = None
	results: List[Tuple[int, List[int], int]] = []

	for start in range(n):
		route, cost = rezolva_tsp_nn_aima(n, matrice, start=start)
		results.append((start, route, cost))
		if best_cost is None or cost < best_cost:
			best_cost = cost
			best_route = route

	return best_route, int(best_cost if best_cost is not None else 0), results
