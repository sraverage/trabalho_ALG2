from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull, KDTree
from scipy.stats import entropy
from sklearn.cluster import DBSCAN


def _compute_hopkins_statistic(norm_coords: np.ndarray) -> float:
    """Calcula a estatística de Hopkins com semente fixa para reprodutibilidade.

    Valores próximos de 1 indicam forte agrupamento espacial; valores próximos de 0.5
    sugerem distribuição aleatória uniforme; valores baixos indicam regularidade.
    """
    rng = np.random.default_rng(42)
    num_probes = max(64, len(norm_coords) * 2)
    probe_points = rng.uniform(0.0, 1.0, size=(num_probes, 2))

    tree = KDTree(norm_coords)
    probe_dists, _ = tree.query(probe_points, k=1)
    real_dists, _ = tree.query(norm_coords, k=2)

    u_mean = float(np.mean(np.square(probe_dists)))
    w_mean = float(np.mean(np.square(real_dists[:, 1])))
    denom = u_mean + w_mean
    return u_mean / denom if denom > 1e-12 else 0.5


def _compute_spatial_entropy(norm_coords: np.ndarray) -> float:
    """Entropia de Shannon normalizada em [0, 1] sobre uma grade 2D das coordenadas."""
    n = len(norm_coords)
    bins = max(2, int(np.ceil(np.sqrt(n / 2.0))))
    hist_2d, _, _ = np.histogram2d(
        norm_coords[:, 0],
        norm_coords[:, 1],
        bins=bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )

    probs = hist_2d.ravel()
    total = float(probs.sum())
    if total > 0.0:
        probs = probs / total

    max_entropy = float(np.log(bins * bins))
    return float(entropy(probs)) / max_entropy if max_entropy > 0.0 else 0.0


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

    safe_spans = np.where(spans > 1e-12, spans, 1.0)
    norm_coords = (arr - arr.min(axis=0)) / safe_spans

    spatial_entropy = _compute_spatial_entropy(norm_coords)
    hopkins_statistic = _compute_hopkins_statistic(norm_coords)

    tree = KDTree(arr)
    nn_query_dists, _ = tree.query(arr, k=2)
    nn_dists = np.asarray(nn_query_dists[:, 1], dtype=np.float64)
    mean_nn = float(np.mean(nn_dists))
    nn_cv = float(np.std(nn_dists) / mean_nn) if mean_nn > 1e-12 else 0.0

    # Usa o percentil 75 das distâncias ao vizinho mais próximo como escala natural do DBSCAN.
    eps_val = max(float(np.percentile(nn_dists, 75)) * 1.35, 1e-6)
    dbscan = DBSCAN(eps=eps_val, min_samples=max(2, min(4, n // 5))).fit(arr)
    cluster_density = float(np.mean(dbscan.labels_ >= 0))

    mst_sparse = minimum_spanning_tree(dist_matrix.astype(np.float64))
    mst_adj = (mst_sparse.toarray() > 0).astype(np.int64)
    degrees = mst_adj.sum(axis=0) + mst_adj.sum(axis=1)

    return {
        "n_vertices": float(n),
        "aspect_ratio": aspect_ratio,
        "convex_hull_ratio": convex_hull_ratio,
        "spatial_entropy": spatial_entropy,
        "hopkins_statistic": hopkins_statistic,
        "cluster_density": cluster_density,
        "mst_max_degree": float(np.max(degrees)),
        "mst_std_degree": float(np.std(degrees)),
        "nn_cv": nn_cv,
    }
