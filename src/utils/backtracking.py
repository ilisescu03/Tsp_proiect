from __future__ import annotations

import time
import heapq
from typing import List, Optional, Tuple

Matrix = List[List[int]]


def rezolva_tsp_backtracking(n: int, matrice: Matrix, mod: str = 'exhaustiv', Y: Optional[int] = None, time_limit_s: Optional[float] = None) -> Tuple[List[int], int]:
    "Rezolva TSP cu backtracking: 'prima', 'y_solutii', 'exhaustiv'."
    if n <= 0:
        raise ValueError("n > 0 required")
    if len(matrice) != n or any(len(row) != n for row in matrice):
        raise ValueError("matrice NxN required")

    if n == 1:
        return [0], 0

    if mod not in ['prima', 'y_solutii', 'exhaustiv']:
        raise ValueError("Invalid mod")

    if Y is None:
        Y = n if mod == 'y_solutii' else 1

    top_solutions = []
    found_first = None
    visited = [False] * n
    route = [0]
    visited[0] = True
    start_time = time.perf_counter() if time_limit_s else 0

    def backtrack(current, cost_cur):
        nonlocal found_first
        if time_limit_s and time.perf_counter() - start_time > time_limit_s:
            return

        if len(route) == n:
            cost_total = cost_cur + matrice[current][0]
            if mod == 'prima':
                if found_first is None:
                    found_first = (route[:], cost_total)
                return
            tup = (cost_total, route[:])
            heapq.heappush(top_solutions, tup)
            if len(top_solutions) > Y:
                heapq.heappop(top_solutions)
            return

        for next_city in range(n):
            if visited[next_city]:
                continue
            cost_new = cost_cur + matrice[current][next_city]
            if top_solutions and cost_new >= top_solutions[0][0]:
                continue
            visited[next_city] = True
            route.append(next_city)
            backtrack(next_city, cost_new)
            route.pop()
            visited[next_city] = False

    backtrack(0, 0)

    if mod == 'prima' and found_first:
        return found_first
    if top_solutions:
        _, route_best = top_solutions[0]
        return route_best, top_solutions[0][0]
    return [0], 0
