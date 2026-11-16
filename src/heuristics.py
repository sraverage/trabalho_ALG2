from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from src.io_tsplib import compute_tour_cost
from src.profiler import profile_block


def cold_start_tsp(
    coords: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple[int, list[int], dict[str, float]]:
    """Estratégia 1: Inicialização trivial na ordem natural dos índices [0, 1, ..., N-1]."""
    _ = coords
    telemetry: dict[str, float] = {}
    with profile_block(telemetry, "heuristic_total"):
        n = dist_matrix.shape[0]
        tour = list(range(n))
        cost = compute_tour_cost(dist_matrix, tour)
    return cost, tour, telemetry


def nearest_neighbor_tsp(
    coords: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple[int, list[int], dict[str, float]]:
    """Estratégia 2: Heurística construtiva gulosa do Vizinho Mais Próximo."""
    _ = coords
    telemetry: dict[str, float] = {}
    with profile_block(telemetry, "heuristic_total"):
        n = dist_matrix.shape[0]
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        tour = [0]
        curr = 0
        inf_sentinel = int(np.max(dist_matrix)) + 1

        for _ in range(n - 1):
            row = np.where(visited, inf_sentinel, dist_matrix[curr])
            nxt = int(np.argmin(row))
            visited[nxt] = True
            tour.append(nxt)
            curr = nxt

        cost = compute_tour_cost(dist_matrix, tour)
    return cost, tour, telemetry


def convex_hull_insertion_tsp(
    coords: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple[int, list[int], dict[str, float]]:
    """Estratégia 3: Sub-rota inicial pelo Fecho Convexo e inserção gulosa vetorizada."""
    telemetry: dict[str, float] = {}
    with profile_block(telemetry, "heuristic_total"):
        n = dist_matrix.shape[0]
        with profile_block(telemetry, "convex_hull"):
            hull = ConvexHull(coords)
            subtour: list[int] = [int(v) for v in hull.vertices]

        in_subtour = np.zeros(n, dtype=bool)
        in_subtour[subtour] = True
        free_nodes = [v for v in range(n) if not in_subtour[v]]

        with profile_block(telemetry, "hull_insertion"):
            while free_nodes:
                u_arr = np.asarray(subtour, dtype=np.int64)
                v_arr = np.roll(u_arr, -1)
                free_arr = np.asarray(free_nodes, dtype=np.int64)

                delta = (
                    dist_matrix[np.ix_(u_arr, free_arr)]
                    + dist_matrix[np.ix_(v_arr, free_arr)]
                    - dist_matrix[u_arr, v_arr][:, None]
                )
                flat_best = int(np.argmin(delta))
                edge_pos, free_pos = divmod(flat_best, len(free_arr))
                chosen_node = int(free_arr[free_pos])

                subtour.insert(edge_pos + 1, chosen_node)
                free_nodes.pop(free_pos)

        zero_idx = subtour.index(0)
        tour = subtour[zero_idx:] + subtour[:zero_idx]
        cost = compute_tour_cost(dist_matrix, tour)

    return cost, tour, telemetry
