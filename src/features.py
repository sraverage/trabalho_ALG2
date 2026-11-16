from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull, KDTree
from scipy.stats import entropy
from sklearn.cluster import DBSCAN


def _compute_hopkins_deterministic(norm_coords: np.ndarray, tree: KDTree) -> float:
    """Estatística de Hopkins determinística (semente fixa) no hiper-retângulo [0,1]^2.

    Compara a média das distâncias quadráticas de pontos de sondagem uniformes U para
    os pontos reais (u_sq) contra a média das distâncias quadráticas ao vizinho mais
    próximo entre os próprios pontos reais (w_sq). Valores próximos de 1 indicam forte
    agrupamento espacial (clusters); valores próximos de 0.5 indicam distribuição de Poisson.
    """
    rng = np.random.default_rng(42)
    probe_points = rng.uniform(0.0, 1.0, size=(max(64, len(norm_coords) * 2), 2))

    u_dists, _ = tree.query(probe_points, k=1)
    w_dists_all, _ = tree.query(norm_coords, k=2)
    w_dists = w_dists_all[:, 1]

    u_mean = float(np.mean(np.square(u_dists)))
    w_mean = float(np.mean(np.square(w_dists)))
    denom = u_mean + w_mean
    if denom <= 1e-12:
        return 0.5
    return u_mean / denom


def extract_instance_features(coords: np.ndarray, dist_matrix: np.ndarray) -> dict[str, float]:
    """Extrai métricas geométricas e topológicas de uma instância 2D do TSP."""
    arr = np.asarray(coords, dtype=np.float64)
    n = len(arr)
    if n < 4:
        raise ValueError(f"Extração de features requer ao menos 4 vértices; recebido N={n}.")

    spans = np.ptp(arr, axis=0)
    max_span = float(np.max(spans))
    min_span = float(np.min(spans))
    aspect_ratio = min_span / max_span if max_span > 1e-12 else 1.0

    hull = ConvexHull(arr)
    convex_hull_ratio = float(len(hull.vertices)) / float(n)

    mins = arr.min(axis=0)
    safe_spans = np.where(spans > 1e-12, spans, 1.0)
    norm_coords = (arr - mins) / safe_spans

    bins = max(2, int(np.ceil(np.sqrt(n / 2.0))))
    hist_2d, _, _ = np.histogram2d(
        norm_coords[:, 0],
        norm_coords[:, 1],
        bins=bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )
    flat_probs = hist_2d.ravel()
    total_mass = float(flat_probs.sum())
    if total_mass > 0.0:
        flat_probs = flat_probs / total_mass
    raw_entropy = float(entropy(flat_probs))
    max_entropy = float(np.log(bins * bins))
    spatial_entropy = raw_entropy / max_entropy if max_entropy > 0.0 else 0.0

    tree = KDTree(arr)
    nn_query_dists, _ = tree.query(arr, k=2)
    nn_dists = np.asarray(nn_query_dists[:, 1], dtype=np.float64)
    mean_nn = float(np.mean(nn_dists))
    nn_cv = float(np.std(nn_dists) / mean_nn) if mean_nn > 1e-12 else 0.0

    norm_tree = KDTree(norm_coords)
    hopkins_statistic = _compute_hopkins_deterministic(norm_coords, norm_tree)

    eps_val = max(float(np.percentile(nn_dists, 75)) * 1.35, 1e-6)
    dbscan = DBSCAN(eps=eps_val, min_samples=max(2, min(4, n // 5))).fit(arr)
    clustered_ratio = float(np.mean(dbscan.labels_ >= 0))

    mst_sparse = minimum_spanning_tree(dist_matrix.astype(np.float64))
    mst_adj = (mst_sparse.toarray() > 0).astype(np.int64)
    degrees = mst_adj.sum(axis=0) + mst_adj.sum(axis=1)
    mst_max_degree = float(np.max(degrees))
    mst_std_degree = float(np.std(degrees))

    return {
        "n_vertices": float(n),
        "aspect_ratio": aspect_ratio,
        "convex_hull_ratio": convex_hull_ratio,
        "spatial_entropy": spatial_entropy,
        "hopkins_statistic": hopkins_statistic,
        "cluster_density": clustered_ratio,
        "mst_max_degree": mst_max_degree,
        "mst_std_degree": mst_std_degree,
        "nn_cv": nn_cv,
    }
