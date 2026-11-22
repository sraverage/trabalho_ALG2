from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np

from src.analysis import run_full_analysis
from src.approximations import christofides_tsp, twice_around_the_tree_tsp
from src.branch_and_bound import BnBResult, BranchAndBoundTSP, benchmark_mst_subroutines
from src.features import extract_instance_features
from src.heuristics import (
    cold_start_tsp,
    convex_hull_insertion_tsp,
    nearest_neighbor_tsp,
)
from src.io_tsplib import (
    KNOWN_OPTIMA_EUC2D,
    compute_euc2d_matrix,
    ensure_tsplib_dataset,
    parse_tsplib,
)


StrategyFn = Callable[[np.ndarray, np.ndarray], tuple[int, list[int], dict[str, float]]]

STRATEGIES: tuple[tuple[str, StrategyFn], ...] = (
    ("cold_start", cold_start_tsp),
    ("nearest_neighbor", nearest_neighbor_tsp),
    ("convex_hull", convex_hull_insertion_tsp),
    ("twice_around", twice_around_the_tree_tsp),
    ("christofides", christofides_tsp),
)

METRICS_COLUMNS: list[str] = [
    "instance",
    "n_vertices",
    "strategy",
    "reference_opt",
    "initial_ub",
    "optimal_cost",
    "best_ub_found_so_far",
    "approx_gap_pct",
    "aborted",
    "heuristic_time",
    "mst_time",
    "matching_time",
    "eulerian_time",
    "convex_hull_time",
    "bb_total_time",
    "bb_mst_bound_time",
    "total_pipeline_time",
    "net_speedup",
    "nodes_expanded",
    "nodes_pruned",
    "nodes_pruned_fast",
    "nodes_pruned_geom",
    "nodes_pruned_mst",
    "nodes_pruned_symmetry",
    "max_depth_reached",
    "peak_memory_kb",
]

FEATURES_COLUMNS: list[str] = [
    "instance",
    "n_vertices",
    "aspect_ratio",
    "convex_hull_ratio",
    "spatial_entropy",
    "hopkins_statistic",
    "cluster_density",
    "mst_max_degree",
    "mst_std_degree",
    "nn_cv",
]


def _load_completed_pairs(metrics_csv: Path) -> tuple[set[tuple[str, str]], dict[str, float]]:
    completed: set[tuple[str, str]] = set()
    cold_times: dict[str, float] = {}
    if not metrics_csv.is_file():
        return completed, cold_times

    with metrics_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            inst = row.get("instance", "")
            strat = row.get("strategy", "")
            if inst and strat:
                completed.add((inst, strat))
                if strat == "cold_start":
                    cold_times[inst] = float(row.get("bb_total_time", 0.0) or 0.0)

    return completed, cold_times


def _resolve_time_limit(mode: str, n_vertices: int, custom_limit: float | None) -> float:
    if custom_limit is not None:
        return custom_limit
    if mode == "full":
        return 1800.0
    return 4.5 if n_vertices <= 29 else 0.8


def _export_instance_features(tsp_paths: list[Path], features_csv: Path) -> None:
    rows: list[dict[str, str | float]] = []
    for path in tsp_paths:
        coords = parse_tsplib(path)
        dist = compute_euc2d_matrix(coords)
        row: dict[str, str | float] = {"instance": path.stem}
        row.update(extract_instance_features(coords, dist))
        rows.append(row)

    with features_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FEATURES_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _export_mst_benchmark(data_dir: Path, mst_bench_csv: Path) -> None:
    sample_coords = parse_tsplib(data_dir / "eil20.tsp")
    sample_dist = compute_euc2d_matrix(sample_coords)
    bench_stats = benchmark_mst_subroutines(sample_dist, sub_size=20, repetitions=400)

    with mst_bench_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(bench_stats.keys()))
        writer.writeheader()
        writer.writerow(bench_stats)


def _build_metrics_row(
    inst_name: str,
    n_vertices: int,
    strat_name: str,
    ref_opt: int,
    ub_cost: int,
    h_telemetry: dict[str, float],
    bb_res: BnBResult,
    cold_start_bb_times: dict[str, float],
) -> dict[str, str | int | float]:
    opt_cost = bb_res["optimal_cost"]
    effective_opt = int(opt_cost) if opt_cost is not None else ref_opt
    gap_pct = (
        100.0 * float(ub_cost - effective_opt) / float(effective_opt)
        if effective_opt > 0
        else 0.0
    )

    h_time = float(h_telemetry.get("heuristic_total_time", 0.0))
    bb_time = float(bb_res["bb_total_time"])
    total_time = h_time + bb_time

    if strat_name == "cold_start":
        cold_start_bb_times[inst_name] = bb_time
        net_speedup = 1.0
    else:
        base_cold = cold_start_bb_times.get(inst_name, total_time)
        net_speedup = base_cold / max(total_time, 1e-9)

    peak_mem = max(float(h_telemetry.get("peak_memory_kb", 0.0)), float(bb_res["peak_memory_kb"]))

    return {
        "instance": inst_name,
        "n_vertices": n_vertices,
        "strategy": strat_name,
        "reference_opt": effective_opt,
        "initial_ub": ub_cost,
        "optimal_cost": "NA" if bb_res["aborted"] else int(bb_res["best_ub_found_so_far"]),
        "best_ub_found_so_far": int(bb_res["best_ub_found_so_far"]),
        "approx_gap_pct": round(gap_pct, 4),
        "aborted": bool(bb_res["aborted"]),
        "heuristic_time": round(h_time, 6),
        "mst_time": round(float(h_telemetry.get("mst_time", 0.0)), 6),
        "matching_time": round(float(h_telemetry.get("matching_time", 0.0)), 6),
        "eulerian_time": round(float(h_telemetry.get("eulerian_time", 0.0)), 6),
        "convex_hull_time": round(float(h_telemetry.get("convex_hull_time", 0.0)), 6),
        "bb_total_time": round(bb_time, 6),
        "bb_mst_bound_time": round(float(bb_res["bb_mst_bound_time"]), 6),
        "total_pipeline_time": round(total_time, 6),
        "net_speedup": round(net_speedup, 4),
        "nodes_expanded": int(bb_res["nodes_expanded"]),
        "nodes_pruned": int(bb_res["nodes_pruned"]),
        "nodes_pruned_fast": int(bb_res["nodes_pruned_fast"]),
        "nodes_pruned_geom": int(bb_res["nodes_pruned_geom"]),
        "nodes_pruned_mst": int(bb_res["nodes_pruned_mst"]),
        "nodes_pruned_symmetry": int(bb_res["nodes_pruned_symmetry"]),
        "max_depth_reached": int(bb_res["max_depth_reached"]),
        "peak_memory_kb": round(peak_mem, 3),
    }


def run_pipeline(
    base_dir: str | Path,
    mode: str = "fast-test",
    custom_time_limit: float | None = None,
    reset: bool = False,
) -> dict[str, object]:
    """Executa o pipeline experimental com checkpointing incremental e gera os artefatos de análise."""
    root = Path(base_dir)
    data_dir = root / "data" / "tsplib"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = results_dir / "metrics_log.csv"
    features_csv = results_dir / "features_log.csv"
    mst_bench_csv = results_dir / "mst_benchmark.csv"

    if reset:
        for fpath in (metrics_csv, features_csv, mst_bench_csv):
            if fpath.is_file():
                fpath.unlink()

    tsp_paths = ensure_tsplib_dataset(data_dir)
    _export_instance_features(tsp_paths, features_csv)
    _export_mst_benchmark(data_dir, mst_bench_csv)

    completed_pairs, cold_start_bb_times = _load_completed_pairs(metrics_csv)
    write_header = not metrics_csv.is_file() or metrics_csv.stat().st_size == 0

    with metrics_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRICS_COLUMNS)
        if write_header:
            writer.writeheader()
            fh.flush()

        for path in tsp_paths:
            inst_name = path.stem
            coords = parse_tsplib(path)
            dist = compute_euc2d_matrix(coords)
            n = len(coords)
            ref_opt = KNOWN_OPTIMA_EUC2D.get(inst_name, 0)
            t_limit = _resolve_time_limit(mode, n, custom_time_limit)

            for strat_name, strat_fn in STRATEGIES:
                if (inst_name, strat_name) in completed_pairs:
                    continue

                ub_cost, ub_tour, h_telemetry = strat_fn(coords, dist)
                solver = BranchAndBoundTSP(coords, dist, time_limit=t_limit)
                bb_res = solver.solve(initial_ub=ub_cost, initial_tour=ub_tour)

                row_m = _build_metrics_row(
                    inst_name=inst_name,
                    n_vertices=n,
                    strat_name=strat_name,
                    ref_opt=ref_opt,
                    ub_cost=ub_cost,
                    h_telemetry=h_telemetry,
                    bb_res=bb_res,
                    cold_start_bb_times=cold_start_bb_times,
                )
                writer.writerow(row_m)
                fh.flush()
                completed_pairs.add((inst_name, strat_name))

    analysis_outputs = run_full_analysis(metrics_csv, features_csv, results_dir)
    return {
        "metrics_csv": metrics_csv,
        "features_csv": features_csv,
        "mst_bench_csv": mst_bench_csv,
        "analysis": analysis_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline experimental TP2 Algoritmos 2 - TSP Geométrico (EUC_2D)"
    )
    parser.add_argument(
        "--mode",
        choices=("fast-test", "full"),
        default="fast-test",
        help="Modo de execução: 'fast-test' (validação rápida) ou 'full' (limite de 1800s).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Sobrescreve o timeout (em segundos) de cada chamada ao Branch-and-Bound.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove logs existentes antes de iniciar a coleta.",
    )
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent
    run_pipeline(
        base_dir=base_dir,
        mode=args.mode,
        custom_time_limit=args.time_limit,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
