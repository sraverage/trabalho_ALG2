from __future__ import annotations

import networkx as nx
import numpy as np

from src.io_tsplib import compute_tour_cost
from src.profiler import profile_block


def _build_graph_and_mst(
    dist_matrix: np.ndarray,
    telemetry: dict[str, float],
) -> tuple[nx.Graph, nx.Graph]:
    """Monta o grafo completo em NetworkX e extrai a MST via Prim com profiling isolado."""
    n = dist_matrix.shape[0]
    with profile_block(telemetry, "mst"):
        graph = nx.from_numpy_array(dist_matrix)
        # from_numpy_array trata peso 0 como ausência de aresta; restaura pares coincidentes.
        if graph.number_of_edges() < n * (n - 1) // 2:
            zero_pairs = np.argwhere((dist_matrix == 0) & (~np.eye(n, dtype=bool)))
            graph.add_edges_from(
                (int(u), int(v), {"weight": 0}) for u, v in zero_pairs if u < v
            )
        mst = nx.minimum_spanning_tree(graph, algorithm="prim", weight="weight")
    return graph, mst


def twice_around_the_tree_tsp(
    coords: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple[int, list[int], dict[str, float]]:
    """Algoritmo 2-aproximado Twice-around-the-tree."""
    del coords
    telemetry: dict[str, float] = {}
    with profile_block(telemetry, "heuristic_total"):
        _, mst = _build_graph_and_mst(dist_matrix, telemetry)
        with profile_block(telemetry, "dfs_preorder"):
            tour = [int(v) for v in nx.dfs_preorder_nodes(mst, source=0)]
        cost = compute_tour_cost(dist_matrix, tour)
    return cost, tour, telemetry


def christofides_tsp(
    coords: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple[int, list[int], dict[str, float]]:
    """Algoritmo 3/2-aproximado de Christofides-Serdyukov."""
    del coords
    telemetry: dict[str, float] = {}
    with profile_block(telemetry, "heuristic_total"):
        graph, mst = _build_graph_and_mst(dist_matrix, telemetry)

        odd_vertices = [int(v) for v, deg in mst.degree() if deg % 2 == 1]

        with profile_block(telemetry, "matching"):
            odd_subgraph = graph.subgraph(odd_vertices)
            matching = nx.algorithms.matching.min_weight_matching(
                odd_subgraph,
                weight="weight",
            )

        with profile_block(telemetry, "eulerian"):
            multigraph = nx.MultiGraph(mst)
            multigraph.add_edges_from(
                (int(u), int(v), {"weight": int(dist_matrix[int(u), int(v)])})
                for u, v in matching
            )
            euler_edges = list(nx.eulerian_circuit(multigraph, source=0))

        with profile_block(telemetry, "shortcutting"):
            visited = {0}
            tour = [0]
            for _, v in euler_edges:
                vertex = int(v)
                if vertex not in visited:
                    visited.add(vertex)
                    tour.append(vertex)

        cost = compute_tour_cost(dist_matrix, tour)
    return cost, tour, telemetry
