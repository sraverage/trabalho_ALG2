from __future__ import annotations

from pathlib import Path
from typing import Sequence
import urllib.error
import urllib.request

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist


TSPLIB_BASE_URL = "https://raw.githubusercontent.com/mastqe/tsplib/master"

INSTANCE_SPECS: tuple[tuple[str, str, int, str], ...] = (
    ("eil10", "eil51", 10, "truncate"),
    ("berlin12", "berlin52", 12, "truncate"),
    ("burma14", "burma14", 14, "geo_scale"),
    ("ulysses16", "ulysses16", 16, "geo_scale"),
    ("st18", "st70", 18, "truncate"),
    ("eil20", "eil51", 20, "truncate"),
    ("ulysses22", "ulysses22", 22, "geo_scale"),
    ("gr24", "gr24", 24, "explicit_lower_diag"),
    ("fri26", "fri26", 26, "explicit_lower_diag"),
    ("bayg29", "bayg29", 29, "display_coord"),
    ("bays29", "bays29", 29, "explicit_full"),
    ("swiss42", "swiss42", 42, "explicit_full"),
    ("att48", "att48", 48, "coord"),
    ("eil51", "eil51", 51, "coord"),
    ("berlin52", "berlin52", 52, "coord"),
    ("st70", "st70", 70, "coord"),
)

KNOWN_OPTIMA_EUC2D: dict[str, int] = {
    "eil10": 159,
    "berlin12": 4056,
    "burma14": 3087,
    "ulysses16": 7398,
    "st18": 339,
    "eil20": 243,
    "ulysses22": 7529,
    "gr24": 1397,
    "fri26": 936,
    "bayg29": 9073,
    "bays29": 1957,
    "swiss42": 1273,
    "att48": 33522,
    "eil51": 426,
    "berlin52": 7542,
    "st70": 675,
}


def _parse_coord_lines(text: str) -> np.ndarray:
    """Extrai pares (x, y) da seção de coordenadas ou exibição de um texto TSPLIB."""
    coords: list[tuple[float, float]] = []
    in_section = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        upper = line.upper()
        if upper.startswith("NODE_COORD_SECTION") or upper.startswith("DISPLAY_DATA_SECTION"):
            in_section = True
            continue

        if in_section:
            if upper.startswith("EOF") or upper.endswith("_SECTION"):
                break
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))

    return np.asarray(coords, dtype=np.float64)


def parse_tsplib(filepath: str | Path) -> np.ndarray:
    """Lê um arquivo .tsp e retorna um array (N, 2) de coordenadas."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo TSPLIB não encontrado: {path}")

    coords = _parse_coord_lines(path.read_text(encoding="utf-8", errors="replace"))
    if len(coords) < 3:
        raise ValueError(f"Arquivo TSPLIB inválido ou sem coordenadas 2D suficientes: {path}")

    return coords


def compute_euc2d_matrix(coords: np.ndarray) -> np.ndarray:
    """Calcula a matriz de distâncias inteiras segundo a regra EUC_2D da TSPLIB."""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        raise ValueError(f"Coordenadas devem ter shape (N, 2) com N >= 2; recebido {arr.shape}")

    raw_dist = cdist(arr, arr, metric="euclidean")
    dist_matrix = np.floor(raw_dist + 0.5).astype(np.int64)
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix


def validate_tour(tour: Sequence[int], n: int) -> None:
    """Garante que o tour é um ciclo hamiltoniano válido iniciando no vértice 0."""
    if len(tour) != n:
        raise ValueError(f"Tamanho do tour ({len(tour)}) difere de N ({n}).")
    if tour[0] != 0:
        raise ValueError(f"O tour deve iniciar no vértice 0; recebido {tour[0]}.")
    if set(tour) != set(range(n)):
        raise ValueError("O tour não é uma permutação válida de 0..N-1.")


def compute_tour_cost(dist_matrix: np.ndarray, tour: Sequence[int]) -> int:
    """Calcula o custo total fechado de um ciclo hamiltoniano."""
    n = dist_matrix.shape[0]
    validate_tour(tour, n)
    idx = np.asarray(tour, dtype=np.int64)
    next_idx = np.roll(idx, -1)
    return int(dist_matrix[idx, next_idx].sum())


def _classical_mds_2d(dist_matrix: np.ndarray) -> np.ndarray:
    """Projeta matrizes explícitas da TSPLIB em R^2 via escalonamento multidimensional clássico.

    Instâncias como gr24, fri26, bays29 e swiss42 não possuem coordenadas nativas;
    os dois maiores autovalores da matriz duplamente centrada B = -0.5 * H * D^2 * H
    reconstroem a melhor aproximação planar em norma de Frobenius.
    """
    n = dist_matrix.shape[0]
    d_sq = np.square(dist_matrix.astype(np.float64))
    centering = np.eye(n) - np.full((n, n), 1.0 / n)
    gram = -0.5 * (centering @ d_sq @ centering)

    eigvals, eigvecs = eigh(gram)
    top_idx = np.argsort(eigvals)[-2:][::-1]
    pos_eigvals = np.maximum(eigvals[top_idx], 1.0)

    coords = eigvecs[:, top_idx] * np.sqrt(pos_eigvals)
    coords -= coords.min(axis=0)
    return coords + 10.0


def _parse_explicit_matrix(text: str, n: int, fmt: str) -> np.ndarray:
    """Lê a seção EDGE_WEIGHT_SECTION em formato triangular inferior ou matriz cheia."""
    tokens: list[float] = []
    in_weights = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("EDGE_WEIGHT_SECTION"):
            in_weights = True
            continue
        if in_weights:
            if upper.startswith("EOF") or upper.endswith("_SECTION"):
                break
            tokens.extend(float(tok) for tok in line.split())

    matrix = np.zeros((n, n), dtype=np.float64)
    if fmt == "explicit_lower_diag":
        pos = 0
        for i in range(n):
            for j in range(i + 1):
                val = tokens[pos]
                matrix[i, j] = val
                matrix[j, i] = val
                pos += 1
    else:
        matrix = np.asarray(tokens[: n * n], dtype=np.float64).reshape((n, n))
        matrix = 0.5 * (matrix + matrix.T)

    np.fill_diagonal(matrix, 0.0)
    return matrix


def _write_euc2d_tsp(filepath: Path, name: str, coords: np.ndarray, comment: str) -> None:
    """Grava coordenadas padronizadas no formato TSPLIB EUC_2D."""
    header = [
        f"NAME : {name}",
        f"COMMENT : {comment}",
        "TYPE : TSP",
        f"DIMENSION : {len(coords)}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
    ]
    body = [f"{idx} {x:.4f} {y:.4f}" for idx, (x, y) in enumerate(coords, start=1)]
    filepath.write_text("\n".join(header + body + ["EOF"]) + "\n", encoding="utf-8")


def _download_raw_instance(source_name: str, cache: dict[str, str]) -> str:
    if source_name in cache:
        return cache[source_name]

    url = f"{TSPLIB_BASE_URL}/{source_name}.tsp"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            cache[source_name] = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise OSError(f"Falha ao baixar instância {source_name} de {url}: {exc}") from exc

    return cache[source_name]


def ensure_tsplib_dataset(data_dir: str | Path) -> list[Path]:
    """Garante as 16 instâncias TSPLIB padronizadas em EUC_2D e as retorna ordenadas por N."""
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_cache: dict[str, str] = {}
    instance_paths: list[Path] = []

    for inst_name, source_name, dim, mode in INSTANCE_SPECS:
        out_path = target_dir / f"{inst_name}.tsp"
        if not out_path.is_file():
            raw_text = _download_raw_instance(source_name, raw_cache)

            if mode == "geo_scale":
                # Escala 100x evita que cidades próximas em graus decimais colapsem para distância 0.
                coords = _parse_coord_lines(raw_text)[:dim] * 100.0
            elif mode in ("explicit_lower_diag", "explicit_full"):
                explicit_mat = _parse_explicit_matrix(raw_text, dim, mode)
                coords = _classical_mds_2d(explicit_mat)
            else:
                coords = _parse_coord_lines(raw_text)[:dim]

            _write_euc2d_tsp(
                out_path,
                name=inst_name,
                coords=coords,
                comment=f"TSPLIB {inst_name} standardized for EUC_2D evaluation (N={dim})",
            )

        instance_paths.append(out_path)

    return sorted(instance_paths, key=lambda p: len(parse_tsplib(p)))
