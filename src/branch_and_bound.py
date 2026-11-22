from __future__ import annotations

import time
import tracemalloc
from typing import TypedDict

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from src.io_tsplib import compute_tour_cost


# Escala inteira fixa para os potenciais de Held-Karp; evita erros de ponto flutuante no teto.
HK_SCALE = 256
HK_TWO_SCALE = 2 * HK_SCALE


class BnBResult(TypedDict):
    optimal_cost: int | None
    best_ub_found_so_far: int
    best_tour: list[int]
    aborted: bool
    nodes_expanded: int
    nodes_pruned: int
    nodes_pruned_fast: int
    nodes_pruned_geom: int
    nodes_pruned_mst: int
    nodes_pruned_symmetry: int
    max_depth_reached: int
    bb_total_time: float
    bb_mst_bound_time: float
    peak_memory_kb: float


def _reduce_prim_edges(
    min_edge: list[int],
    remaining: list[int],
    weight_rows: list[list[int]],
) -> int:
    """Executa Prim denso em listas Python puras para evitar overhead de arrays pequenos na DFS."""
    total = 0
    for _ in range(len(remaining)):
        best_val = 10**15
        best_idx = 0
        for idx, val in enumerate(min_edge):
            if val < best_val:
                best_val = val
                best_idx = idx

        total += best_val
        u = remaining[best_idx]
        min_edge.pop(best_idx)
        remaining.pop(best_idx)

        row_u = weight_rows[u]
        for idx, v in enumerate(remaining):
            d = row_u[v]
            if d < min_edge[idx]:
                min_edge[idx] = d

    return total


def _mst_weight_numpy_prim(sub_mat: np.ndarray) -> int:
    """Prim denso em NumPy para comparar com a conversão CSR do SciPy em submatrizes pequenas."""
    k = sub_mat.shape[0]
    if k <= 1:
        return 0
    if k == 2:
        return int(sub_mat[0, 1])

    in_mst = np.zeros(k, dtype=bool)
    in_mst[0] = True
    min_edge = sub_mat[0].copy()
    inf_val = int(np.max(sub_mat)) + 1
    total = 0

    for _ in range(k - 1):
        masked = np.where(in_mst, inf_val, min_edge)
        u = int(np.argmin(masked))
        in_mst[u] = True
        total += int(min_edge[u])
        np.minimum(min_edge, sub_mat[u], out=min_edge)

    return total


def _mst_weight_scipy(sub_mat: np.ndarray) -> int:
    """Peso da MST via scipy.sparse.csgraph para benchmark de overhead."""
    if sub_mat.shape[0] <= 1:
        return 0
    return int(round(float(minimum_spanning_tree(sub_mat).sum())))


def benchmark_mst_subroutines(
    dist_matrix: np.ndarray,
    sub_size: int = 20,
    repetitions: int = 500,
) -> dict[str, float]:
    """Compara tempo e memória do Prim denso em NumPy contra scipy.sparse.csgraph."""
    k = min(sub_size, dist_matrix.shape[0])
    sub_mat = dist_matrix[:k, :k].copy()
    sub_mat_float = sub_mat.astype(np.float64)

    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(repetitions):
        _mst_weight_numpy_prim(sub_mat)
    prim_time = time.perf_counter() - t0
    _, prim_peak = tracemalloc.get_traced_memory()

    tracemalloc.reset_peak()
    t1 = time.perf_counter()
    for _ in range(repetitions):
        _mst_weight_scipy(sub_mat_float)
    scipy_time = time.perf_counter() - t1
    _, scipy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "submatrix_size": float(k),
        "repetitions": float(repetitions),
        "numpy_prim_total_sec": prim_time,
        "scipy_csgraph_total_sec": scipy_time,
        "speedup_ratio": scipy_time / max(prim_time, 1e-12),
        "numpy_prim_peak_kb": prim_peak / 1024.0,
        "scipy_csgraph_peak_kb": scipy_peak / 1024.0,
    }


def _segments_strictly_intersect(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> bool:
    """Testa interseção própria entre os segmentos (a, b) e (c, d) via orientação CCW."""
    ccw_acd = (dy - ay) * (cx - ax) > (cy - ay) * (dx - ax)
    ccw_bcd = (dy - by) * (cx - bx) > (cy - by) * (dx - bx)
    if ccw_acd == ccw_bcd:
        return False

    ccw_abc = (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
    ccw_abd = (dy - ay) * (bx - ax) > (by - ay) * (dx - ax)
    return ccw_abc != ccw_abd


def _compute_hk_potentials(dist: np.ndarray, root: int, iters: int = 180) -> np.ndarray:
    """Estima multiplicadores lagrangianos de Held-Karp (em ponto fixo) por ascensão de subgradiente."""
    n = dist.shape[0]
    if n < 4:
        return np.zeros(n, dtype=np.int64)

    pi = np.zeros(n, dtype=np.float64)
    d_float = dist.astype(np.float64)
    others = [v for v in range(n) if v != root]
    others_arr = np.asarray(others, dtype=np.int64)
    k = n - 1

    best_lb = -1e18
    best_pi = pi.copy()
    step = 2.0

    for it in range(iters):
        penalized = d_float + pi[:, None] + pi[None, :]
        np.fill_diagonal(penalized, 1e15)
        sub = penalized[np.ix_(others_arr, others_arr)]

        in_mst = np.zeros(k, dtype=bool)
        in_mst[0] = True
        min_edge = sub[0].copy()
        parent = np.zeros(k, dtype=np.int64)
        degree = np.zeros(n, dtype=np.int64)
        mst_weight = 0.0

        for _ in range(k - 1):
            u = int(np.argmin(np.where(in_mst, 1e15, min_edge)))
            in_mst[u] = True
            mst_weight += float(min_edge[u])

            p = int(parent[u])
            degree[others[u]] += 1
            degree[others[p]] += 1

            better = sub[u] < min_edge
            min_edge = np.where(better, sub[u], min_edge)
            parent = np.where(better, u, parent)

        root_edges = penalized[root, others_arr]
        e1, e2 = (int(i) for i in np.argsort(root_edges)[:2])
        degree[root] = 2
        degree[others[e1]] += 1
        degree[others[e2]] += 1

        lb = mst_weight + float(root_edges[e1] + root_edges[e2]) - 2.0 * float(np.sum(pi))
        if lb > best_lb:
            best_lb = lb
            best_pi = pi.copy()

        subgradient = degree - 2
        norm_sq = int(np.sum(subgradient * subgradient))
        if norm_sq == 0:
            break

        # Passo tipo Polyak amortecido e centrado para manter soma(pi) == 0.
        gap = max(best_lb * 0.12 * (0.90 ** (it // 5)), 0.5) + (best_lb - lb)
        pi = pi + (step * gap / norm_sq) * subgradient
        pi -= np.mean(pi)
        if (it + 1) % 20 == 0:
            step *= 0.75

    return np.round(best_pi * HK_SCALE).astype(np.int64)


class BranchAndBoundTSP:
    """Resolvedor exato do TSP usando DFS in-place, bitmask e funil de poda em 3 níveis."""

    def __init__(
        self,
        coords: np.ndarray,
        dist_matrix: np.ndarray,
        time_limit: float = 1800.0,
        use_geom_pruning: bool = True,
        mst_mode: str = "numpy_prim",
    ) -> None:
        self.coords = np.asarray(coords, dtype=np.float64)
        self.dist = np.asarray(dist_matrix, dtype=np.int64)
        self.n = int(self.dist.shape[0])
        self.time_limit = float(time_limit)
        self.use_geom_pruning = bool(use_geom_pruning)
        self.mst_mode = mst_mode

        self.dist_list: list[list[int]] = self.dist.tolist()
        sorted_dists = np.sort(self.dist, axis=1)
        self.min1 = sorted_dists[:, 1].astype(np.int64)
        self.min1_list: list[int] = self.min1.tolist()
        self.two_min_sum = (sorted_dists[:, 1] + sorted_dists[:, 2]).astype(np.int64)
        self.two_min_list: list[int] = self.two_min_sum.tolist()

        # Fixar a raiz no vértice mais isolado antecipa a decisão das arestas mais caras.
        self.root_vertex = int(np.argmax(self.two_min_sum))

        self.pi_int = _compute_hk_potentials(self.dist, self.root_vertex)
        self.pi_list: list[int] = self.pi_int.tolist()

        mod_dist = (self.dist * HK_SCALE + self.pi_int[:, None] + self.pi_int[None, :]).astype(
            np.int64
        )
        np.fill_diagonal(mod_dist, 10**15)
        self.mod_list: list[list[int]] = mod_dist.tolist()

        mod_sorted = np.sort(mod_dist, axis=1)
        self.mod_two_adj: list[int] = [
            int(mod_sorted[u, 0] + mod_sorted[u, 1] - 4 * self.pi_int[u])
            for u in range(self.n)
        ]
        self.mod_end_adj: list[int] = [
            int(mod_sorted[u, 0] - 2 * self.pi_int[u]) for u in range(self.n)
        ]

        reduced_cost = mod_dist - (mod_sorted[:, 0][None, :] // 2)
        np.fill_diagonal(reduced_cost, 10**15)
        self.sorted_neighbors = np.argsort(reduced_cost, axis=1).astype(np.int64)
        self.neighbors_list: list[list[int]] = self.sorted_neighbors.tolist()

        self.best_cost = 0
        self.best_tour: list[int] = []
        self.path: list[int] = [0] * self.n

        self.nodes_expanded = 0
        self.nodes_pruned_fast = 0
        self.nodes_pruned_geom = 0
        self.nodes_pruned_mst = 0
        self.nodes_pruned_symmetry = 0
        self.max_depth_reached = 1
        self.aborted = False

        self.start_time = 0.0
        self.bb_mst_bound_time = 0.0

    def _violates_2opt_geometry(self, count: int, curr: int, nxt: int) -> bool:
        """Retorna True se (curr, nxt) cruza uma aresta anterior com ganho inteiro estrito.

        Exigir ganho estrito evita descartar empates criados pelo arredondamento inteiro EUC_2D.
        """
        if count < 3:
            return False

        cx = float(self.coords[curr, 0])
        cy = float(self.coords[curr, 1])
        dx = float(self.coords[nxt, 0])
        dy = float(self.coords[nxt, 1])
        d_cd = self.dist_list[curr][nxt]

        for i in range(count - 2):
            u = self.path[i]
            v = self.path[i + 1]
            if self.dist_list[u][v] + d_cd <= self.dist_list[u][curr] + self.dist_list[v][nxt]:
                continue
            ax = float(self.coords[u, 0])
            ay = float(self.coords[u, 1])
            bx = float(self.coords[v, 0])
            by = float(self.coords[v, 1])
            if _segments_strictly_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                return True
        return False

    def _compute_mst_lower_bound(
        self,
        curr: int,
        curr_cost: int,
        visited_mask: int,
        rem_pi_sum: int,
    ) -> int:
        """Calcula o limite inferior de 1-árvore sobre os vértices não visitados U."""
        t0 = time.perf_counter()
        root = self.root_vertex
        unvisited = [u for u in range(self.n) if not (visited_mask & (1 << u))]

        if self.mst_mode == "scipy_csgraph":
            idx_arr = np.asarray(unvisited, dtype=np.int64)
            sub_mat = self.dist[np.ix_(idx_arr, idx_arr)]
            mst_weight = _mst_weight_scipy(sub_mat.astype(np.float64))
            min_root = int(self.dist[root, idx_arr].min())
            min_curr = int(self.dist[curr, idx_arr].min())
            self.bb_mst_bound_time += time.perf_counter() - t0
            return curr_cost + mst_weight + min_root + min_curr

        mod_list = self.mod_list
        pi_list = self.pi_list
        mr = mod_list[root]
        mc = mod_list[curr]

        u0 = unvisited[0]
        row0 = mod_list[u0]
        min_edge = [row0[unvisited[j]] for j in range(1, len(unvisited))]
        mst_weight = _reduce_prim_edges(min_edge, unvisited[1:], mod_list)

        min_r = min(mr[u] for u in unvisited)
        min_c = min(mc[u] for u in unvisited)
        scaled_rem = mst_weight + min_r + min_c - pi_list[root] - pi_list[curr] - 2 * rem_pi_sum

        self.bb_mst_bound_time += time.perf_counter() - t0
        return curr_cost + (scaled_rem + HK_SCALE - 1) // HK_SCALE

    def _dfs(
        self,
        curr: int,
        count: int,
        curr_cost: int,
        visited_mask: int,
        rem_two_min: int,
        rem_mod_adj: int,
        rem_pi_sum: int,
    ) -> None:
        self.nodes_expanded += 1
        if count > self.max_depth_reached:
            self.max_depth_reached = count

        # Checa o timeout a cada 1024 nós para evitar custo de syscall em cada chamada.
        if (self.nodes_expanded & 1023) == 0:
            if time.perf_counter() - self.start_time > self.time_limit:
                self.aborted = True
                return

        root = self.root_vertex

        # Nível 1: limites rápidos de grau 2 (original e com potenciais de Held-Karp).
        fast_lb1 = curr_cost + (
            rem_two_min + self.min1_list[root] + self.min1_list[curr] + 1
        ) // 2
        if fast_lb1 >= self.best_cost:
            self.nodes_pruned_fast += 1
            return

        fast_lb2 = curr_cost + (
            rem_mod_adj
            + self.mod_end_adj[root]
            + self.mod_end_adj[curr]
            + HK_TWO_SCALE
            - 1
        ) // HK_TWO_SCALE
        if fast_lb2 >= self.best_cost:
            self.nodes_pruned_fast += 1
            return

        # Nível 3: 1-árvore de Held-Karp quando ainda restam ao menos 3 vértices livres.
        if self.n - count > 2:
            mst_lb = self._compute_mst_lower_bound(curr, curr_cost, visited_mask, rem_pi_sum)
            if mst_lb >= self.best_cost:
                self.nodes_pruned_mst += 1
                return

        n = self.n
        neighbors = self.neighbors_list[curr]
        dist_curr = self.dist_list[curr]
        dist_root = self.dist_list[root]
        two_min_l = self.two_min_list
        mod_two_l = self.mod_two_adj
        pi_l = self.pi_list

        for nxt in neighbors:
            if nxt == curr or (visited_mask & (1 << nxt)):
                continue

            # Quebra de simetria em grafos não-direcionados: descarta a direção reversa na folha.
            if count == n - 1 and self.path[1] > nxt:
                self.nodes_pruned_symmetry += 1
                continue

            step_cost = curr_cost + dist_curr[nxt]
            if step_cost >= self.best_cost:
                self.nodes_pruned_fast += 1
                continue

            # Nível 2: filtro geométrico 2-Opt sobre a aresta candidata (curr, nxt).
            if self.use_geom_pruning and self._violates_2opt_geometry(count, curr, nxt):
                self.nodes_pruned_geom += 1
                continue

            self.path[count] = nxt
            if count + 1 == n:
                total_cost = step_cost + dist_root[nxt]
                if total_cost < self.best_cost:
                    self.best_cost = total_cost
                    self.best_tour = self.path.copy()
            else:
                self._dfs(
                    nxt,
                    count + 1,
                    step_cost,
                    visited_mask | (1 << nxt),
                    rem_two_min - two_min_l[nxt],
                    rem_mod_adj - mod_two_l[nxt],
                    rem_pi_sum - pi_l[nxt],
                )
                if self.aborted:
                    return

    def solve(
        self,
        initial_ub: int | None = None,
        initial_tour: list[int] | None = None,
    ) -> BnBResult:
        """Executa o Branch-and-Bound a partir do limite superior inicial fornecido."""
        if initial_ub is None or initial_tour is None:
            default_tour = list(range(self.n))
            self.best_cost = compute_tour_cost(self.dist, default_tour)
            self.best_tour = default_tour
        else:
            self.best_cost = int(initial_ub)
            self.best_tour = list(initial_tour)

        root = self.root_vertex
        self.path = [0] * self.n
        self.path[0] = root
        self.nodes_expanded = 0
        self.nodes_pruned_fast = 0
        self.nodes_pruned_geom = 0
        self.nodes_pruned_mst = 0
        self.nodes_pruned_symmetry = 0
        self.max_depth_reached = 1
        self.aborted = False
        self.bb_mst_bound_time = 0.0

        already_tracing = tracemalloc.is_tracing()
        if already_tracing:
            tracemalloc.reset_peak()
        else:
            tracemalloc.start()

        self.start_time = time.perf_counter()
        self._dfs(
            curr=root,
            count=1,
            curr_cost=0,
            visited_mask=1 << root,
            rem_two_min=int(self.two_min_sum.sum()) - self.two_min_list[root],
            rem_mod_adj=sum(self.mod_two_adj) - self.mod_two_adj[root],
            rem_pi_sum=sum(self.pi_list) - self.pi_list[root],
        )

        # Normaliza o ciclo final para iniciar no vértice 0.
        if 0 in self.best_tour and self.best_tour[0] != 0:
            zero_pos = self.best_tour.index(0)
            self.best_tour = self.best_tour[zero_pos:] + self.best_tour[:zero_pos]

        bb_total_time = time.perf_counter() - self.start_time
        _, peak_bytes = tracemalloc.get_traced_memory()
        if not already_tracing:
            tracemalloc.stop()

        total_pruned = (
            self.nodes_pruned_fast
            + self.nodes_pruned_geom
            + self.nodes_pruned_mst
            + self.nodes_pruned_symmetry
        )

        return {
            "optimal_cost": None if self.aborted else self.best_cost,
            "best_ub_found_so_far": self.best_cost,
            "best_tour": list(self.best_tour),
            "aborted": self.aborted,
            "nodes_expanded": self.nodes_expanded,
            "nodes_pruned": total_pruned,
            "nodes_pruned_fast": self.nodes_pruned_fast,
            "nodes_pruned_geom": self.nodes_pruned_geom,
            "nodes_pruned_mst": self.nodes_pruned_mst,
            "nodes_pruned_symmetry": self.nodes_pruned_symmetry,
            "max_depth_reached": self.max_depth_reached,
            "bb_total_time": bb_total_time,
            "bb_mst_bound_time": self.bb_mst_bound_time,
            "peak_memory_kb": peak_bytes / 1024.0,
        }
