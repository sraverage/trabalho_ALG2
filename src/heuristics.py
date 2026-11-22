from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from src.io_tsplib import compute_tour_cost
from src.profiler import profile_block


def cold_start_tsp(
    coords: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple[int, list[int], dict[str, float]]:
    """Inicialização trivial seguindo a ordem natural dos vértices [0, 1, ..., N-1]."""
    del coords
    telemetry: dict[str, float] = {}
    with profile_block(telemetry, "heuristic_total"):
        tour = list(range(dist_matrix.shape[0]))
        cost = compute_tour_cost(dist_matrix, tour)
    return cost, tour, telemetry


def nearest_neighbor_tsp(
    coords: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple[int, list[int], dict[str, float]]:
    """Heurística gulosa do Vizinho Mais Próximo a partir do vértice 0."""
    del coords
    telemetry: dict[str, float] = {}
    with profile_block(telemetry, "heuristic_total"):
        n = dist_matrix.shape[0]
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        tour = [0]
        current = 0
        sentinel = int(np.max(dist_matrix)) + 1

        for _ in range(n - 1):
            candidates = np.where(visited, sentinel, dist_matrix[current])
            nxt = int(np.argmin(candidates))
            visited[nxt] = True
            tour.append(nxt)
            current = nxt

        cost = compute_tour_cost(dist_matrix, tour)
    return cost, tour, telemetry


def convex_hull_insertion_tsp(
    coords: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple[int, list[int], dict[str, float]]:
    """Constrói o ciclo inicial pelo fecho convexo e insere os demais vértices gulosamente."""
    telemetry: dict[str, float] = {}
    with profile_block(telemetry, "heuristic_total"):
        n = dist_matrix.shape[0]
        with profile_block(telemetry, "convex_hull"):
            hull = ConvexHull(coords)
            subtour = [int(v) for v in hull.vertices]

        in_subtour = np.zeros(n, dtype=bool)
        in_subtour[subtour] = True
        free_nodes = [v for v in range(n) if not in_subtour[v]]

        with profile_block(telemetry, "hull_insertion"):
            while free_nodes:
                u_arr = np.asarray(subtour, dtype=np.int64)
                v_arr = np.roll(u_arr, -1)
                free_arr = np.asarray(free_nodes, dtype=np.int64)

                # Avalia d(u, k) + d(k, v) - d(u, v) de forma matricial para todas as arestas e nós livres.
                insertion_cost = (
                    dist_matrix[np.ix_(u_arr, free_arr)]
                    + dist_matrix[np.ix_(v_arr, free_arr)]
                    - dist_matrix[u_arr, v_arr][:, None]
                )
                best_flat = int(np.argmin(insertion_cost))
                edge_pos, free_pos = divmod(best_flat, len(free_arr))
                chosen = int(free_arr[free_pos])

                subtour.insert(edge_pos + 1, chosen)
                free_nodes.pop(free_pos)

        zero_pos = subtour.index(0)
        tour = subtour[zero_pos:] + subtour[:zero_pos]
        cost = compute_tour_cost(dist_matrix, tour)

    return cost, tour, telemetry
