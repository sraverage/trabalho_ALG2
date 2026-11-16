from __future__ import annotations

import time
import tracemalloc
from typing import TypedDict

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


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
    min_e: list[int],
    nodes: list[int],
    dist_list: list[list[int]],
) -> int:
    """Executa os passos gulosos de Prim sobre listas de distâncias já inicializadas."""
    total = 0
    for _ in range(len(nodes)):
        best_val = 10**12
        best_idx = 0
        for idx, val in enumerate(min_e):
            if val < best_val:
                best_val = val
                best_idx = idx
        total += best_val
        u = nodes[best_idx]
        min_e.pop(best_idx)
        nodes.pop(best_idx)
        row_u = dist_list[u]
        for idx, v in enumerate(nodes):
            d = row_u[v]
            if d < min_e[idx]:
                min_e[idx] = d
    return total


def _mst_weight_numpy_prim(sub_mat: np.ndarray) -> int:
    """Prim denso in-place em NumPy para submatrizes pequenas no interior do B&B.

    Evita a conversão para matriz esparsa CSR e a alocação de objetos Python do SciPy
    em chamadas milhões de vezes para k <= 70.
    """
    k = sub_mat.shape[0]
    if k <= 1:
        return 0
    if k == 2:
        return int(sub_mat[0, 1])

    in_mst = np.zeros(k, dtype=bool)
    in_mst[0] = True
    min_e = sub_mat[0].copy()
    inf_val = int(np.max(sub_mat)) + 1
    total = 0

    for _ in range(k - 1):
        masked = np.where(in_mst, inf_val, min_e)
        u = int(np.argmin(masked))
        in_mst[u] = True
        total += int(min_e[u])
        np.minimum(min_e, sub_mat[u], out=min_e)
    return total


def _mst_weight_scipy(sub_mat: np.ndarray) -> int:
    """MST via scipy.sparse.csgraph.minimum_spanning_tree para comparação de overhead."""
    if sub_mat.shape[0] <= 1:
        return 0
    return int(round(float(minimum_spanning_tree(sub_mat).sum())))


def benchmark_mst_subroutines(
    dist_matrix: np.ndarray,
    sub_size: int = 20,
    repetitions: int = 500,
) -> dict[str, float]:
    """Compara empiricamente o tempo e a memória do Prim vetorizado vs SciPy csgraph."""
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
    """Teste de orientação CCW em O(1) para interseção própria entre (a,b) e (c,d)."""
    ccw_acd = (dy - ay) * (cx - ax) > (cy - ay) * (dx - ax)
    ccw_bcd = (dy - by) * (cx - bx) > (cy - by) * (dx - bx)
    if ccw_acd == ccw_bcd:
        return False
    ccw_abc = (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
    ccw_abd = (dy - ay) * (bx - ax) > (by - ay) * (dx - ax)
    return ccw_abc != ccw_abd


class BranchAndBoundTSP:
    """Algoritmo Exato Branch-and-Bound via DFS In-Place com Bitmask e Funil de Poda."""

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

        self.sorted_neighbors = np.argsort(self.dist, axis=1).astype(np.int64)
        self.neighbors_list: list[list[int]] = self.sorted_neighbors.tolist()
        self.dist_list: list[list[int]] = self.dist.tolist()
        sorted_dists = np.sort(self.dist, axis=1)
        self.min1 = sorted_dists[:, 1].astype(np.int64)
        self.min1_list: list[int] = self.min1.tolist()
        self.two_min_sum = (sorted_dists[:, 1] + sorted_dists[:, 2]).astype(np.int64)
        self.two_min_list: list[int] = self.two_min_sum.tolist()
        self.root_vertex: int = int(np.argmax(self.two_min_sum))

        self.best_cost: int = 0
        self.best_tour: list[int] = []
        self.path: list[int] = [0] * self.n

        self.nodes_expanded: int = 0
        self.nodes_pruned_fast: int = 0
        self.nodes_pruned_geom: int = 0
        self.nodes_pruned_mst: int = 0
        self.nodes_pruned_symmetry: int = 0
        self.max_depth_reached: int = 1
        self.aborted: bool = False

        self.start_time: float = 0.0
        self.bb_mst_bound_time: float = 0.0

    def _Violates_2opt_geometry(self, count: int, curr: int, nxt: int) -> bool:
        """Verifica se a aresta (curr, nxt) cruza alguma aresta não adjacente com ganho > 0.

        Ao exigir simultaneamente interseção própria no plano e desigualdade estrita
        d(a,b) + d(c,d) > d(a,c) + d(b,d) na matriz inteira EUC_2D, garante-se que
        qualquer completamento do caminho parcial admitiria um movimento 2-Opt válido
        estritamente melhor, preservando a exatidão sob arredondamento inteiro.
        """
        if count < 3:
            return False
        cx = float(self.coords[curr, 0])
        cy = float(self.coords[curr, 1])
        dx = float(self.coords[nxt, 0])
        dy = float(self.coords[nxt, 1])
        d_cd = int(self.dist[curr, nxt])

        for i in range(count - 2):
            u = self.path[i]
            v = self.path[i + 1]
            if int(self.dist[u, v]) + d_cd <= int(self.dist[u, curr]) + int(self.dist[v, nxt]):
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
    ) -> int:
        """Calcula o máximo entre o 1-Tree clássico em U e o 1-Tree de Held-Karp no pivô outlier r.

        Quando um vértice isolado r (como Gibraltar em ulysses16) permanece em U, uma MST
        simples sobre U conecta r como folha de grau 1, subestimando o custo real em uma
        aresta longa inteira. Ao escolher r = argmax_{u in U} two_min_sum[u] como raiz da
        1-árvore (contraindo {0, curr} como supernó da árvore geradora sobre U \\ {r} e
        somando as duas menores arestas incidentes a r em (U \\ {r}) U {0, curr}), obtém-se
        um limite inferior estritamente admissível e imune a folhas outliers.
        """
        t0 = time.perf_counter()
        root = self.root_vertex
        unvisited = [u for u in range(self.n) if not (visited_mask & (1 << u))]
        if self.mst_mode == "scipy_csgraph":
            idx_arr = np.asarray(unvisited, dtype=np.int64)
            sub_mat = self.dist[np.ix_(idx_arr, idx_arr)]
            mst_w = _mst_weight_scipy(sub_mat.astype(np.float64))
            min_0 = int(self.dist[root, idx_arr].min())
            min_curr = int(self.dist[curr, idx_arr].min())
            self.bb_mst_bound_time += time.perf_counter() - t0
            return curr_cost + mst_w + min_0 + min_curr

        dist_list = self.dist_list
        d_root = dist_list[root]
        dc = dist_list[curr]
        k = len(unvisited)

        u0 = unvisited[0]
        row0 = dist_list[u0]
        min_e = [row0[unvisited[j]] for j in range(1, k)]
        mst_w = _reduce_prim_edges(min_e, unvisited[1:], dist_list)

        min_0 = min(d_root[u] for u in unvisited)
        min_curr = min(dc[u] for u in unvisited)
        lb_classic = curr_cost + mst_w + min_0 + min_curr
        if lb_classic >= self.best_cost:
            self.bb_mst_bound_time += time.perf_counter() - t0
            return lb_classic

        two_min_l = self.two_min_list
        r = max(unvisited, key=lambda u: two_min_l[u])
        rem = [u for u in unvisited if u != r]
        min_e2 = [d_root[u] if d_root[u] < dc[u] else dc[u] for u in rem]
        mst_w2 = _reduce_prim_edges(min_e2, rem[:], dist_list)

        dr = dist_list[r]
        m1 = d_root[r] if d_root[r] < dc[r] else dc[r]
        m2 = dc[r] if d_root[r] < dc[r] else d_root[r]
        for u in rem:
            d = dr[u]
            if d < m1:
                m2 = m1
                m1 = d
            elif d < m2:
                m2 = d

        lb_held_karp = curr_cost + mst_w2 + m1 + m2
        self.bb_mst_bound_time += time.perf_counter() - t0
        return lb_classic if lb_classic >= lb_held_karp else lb_held_karp

    def _dfs(
        self,
        curr: int,
        count: int,
        curr_cost: int,
        visited_mask: int,
        rem_two_min: int,
    ) -> None:
        self.nodes_expanded += 1
        if count > self.max_depth_reached:
            self.max_depth_reached = count

        if (self.nodes_expanded & 1023) == 0:
            if time.perf_counter() - self.start_time > self.time_limit:
                self.aborted = True
                return

        root = self.root_vertex
        fast_lb = curr_cost + (
            rem_two_min + self.min1_list[root] + self.min1_list[curr] + 1
        ) // 2
        if fast_lb >= self.best_cost:
            self.nodes_pruned_fast += 1
            return

        if self.n - count > 3:
            mst_lb = self._compute_mst_lower_bound(curr, curr_cost, visited_mask)
            if mst_lb >= self.best_cost:
                self.nodes_pruned_mst += 1
                return

        n = self.n
        neighbors = self.neighbors_list[curr]
        dist_curr = self.dist_list[curr]
        dist_root = self.dist_list[root]
        two_min_l = self.two_min_list
        for nxt in neighbors:
            if nxt == curr or (visited_mask & (1 << nxt)):
                continue

            if count == n - 1 and self.path[1] > nxt:
                self.nodes_pruned_symmetry += 1
                continue

            step_cost = curr_cost + dist_curr[nxt]
            if step_cost >= self.best_cost:
                self.nodes_pruned_fast += 1
                continue

            if self.use_geom_pruning and self._Violates_2opt_geometry(count, curr, nxt):
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
                )
                if self.aborted:
                    return

    def solve(
        self,
        initial_ub: int | None = None,
        initial_tour: list[int] | None = None,
    ) -> BnBResult:
        """Executa a busca exata Branch-and-Bound inicializada com o warm-start fornecido."""
        if initial_ub is None or initial_tour is None:
            default_tour = list(range(self.n))
            idx = np.asarray(default_tour, dtype=np.int64)
            self.best_cost = int(self.dist[idx, np.roll(idx, -1)].sum())
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
        if not already_tracing:
            tracemalloc.start()
        else:
            tracemalloc.reset_peak()

        self.start_time = time.perf_counter()
        initial_rem_two_min = int(self.two_min_sum.sum()) - int(self.two_min_sum[root])

        self._dfs(
            curr=root,
            count=1,
            curr_cost=0,
            visited_mask=1 << root,
            rem_two_min=initial_rem_two_min,
        )

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
