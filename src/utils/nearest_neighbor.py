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
