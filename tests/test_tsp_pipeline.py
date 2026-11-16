from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from main import run_pipeline
from src.approximations import christofides_tsp, twice_around_the_tree_tsp
from src.branch_and_bound import BranchAndBoundTSP, benchmark_mst_subroutines
from src.features import extract_instance_features
from src.heuristics import (
    cold_start_tsp,
    convex_hull_insertion_tsp,
    nearest_neighbor_tsp,
)
from src.io_tsplib import (
    compute_euc2d_matrix,
    compute_tour_cost,
    ensure_tsplib_dataset,
    parse_tsplib,
    validate_tour,
)
from src.profiler import profile_block


class TestTSPPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.square_coords = np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0],
            ],
            dtype=np.float64,
        )
        self.square_dist = compute_euc2d_matrix(self.square_coords)

    def test_euc2d_matrix_and_tour_cost(self) -> None:
        self.assertEqual(self.square_dist.shape, (4, 4))
        self.assertEqual(self.square_dist.dtype, np.int64)
        np.testing.assert_array_equal(np.diag(self.square_dist), np.zeros(4, dtype=np.int64))
        self.assertEqual(int(self.square_dist[0, 1]), 10)
        self.assertEqual(int(self.square_dist[0, 2]), 14)

        perimeter = compute_tour_cost(self.square_dist, [0, 1, 2, 3])
        crossed = compute_tour_cost(self.square_dist, [0, 2, 1, 3])
        self.assertEqual(perimeter, 40)
        self.assertEqual(crossed, 48)

    def test_validate_tour_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            validate_tour([1, 0, 2, 3], 4)
        with self.assertRaises(ValueError):
            validate_tour([0, 1, 1, 3], 4)
        with self.assertRaises(ValueError):
            validate_tour([0, 1, 2], 4)

    def test_parse_tsplib_and_profiler(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tsp_file = Path(tmpdir) / "unit.tsp"
            tsp_file.write_text(
                "NAME : unit\nTYPE : TSP\nDIMENSION : 4\nEDGE_WEIGHT_TYPE : EUC_2D\n"
                "NODE_COORD_SECTION\n1 0.0 0.0\n2 3.0 4.0\n3 6.0 0.0\n4 3.0 -4.0\nEOF\n",
                encoding="utf-8",
            )
            coords = parse_tsplib(tsp_file)
            self.assertEqual(coords.shape, (4, 2))
            dist = compute_euc2d_matrix(coords)
            self.assertEqual(int(dist[0, 1]), 5)

        telemetry: dict[str, float] = {}
        with profile_block(telemetry, "subroutine"):
            arr = np.arange(5000, dtype=np.float64)
            _ = arr.sum()
        self.assertIn("subroutine_time", telemetry)
        self.assertIn("subroutine_peak_kb", telemetry)
        self.assertIn("peak_memory_kb", telemetry)
        self.assertGreater(telemetry["subroutine_time"], 0.0)
        self.assertGreater(telemetry["peak_memory_kb"], 0.0)

    def test_ensure_tsplib_dataset_completeness(self) -> None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "tsplib"
        paths = ensure_tsplib_dataset(data_dir)
        self.assertEqual(len(paths), 16)
        sizes = [len(parse_tsplib(p)) for p in paths]
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreaterEqual(min(sizes), 10)
        self.assertLessEqual(max(sizes), 75)

    def test_extract_instance_features_invariants(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        circle_coords = np.column_stack([100.0 * np.cos(angles), 100.0 * np.sin(angles)])
        circle_dist = compute_euc2d_matrix(circle_coords)
        feats_circle = extract_instance_features(circle_coords, circle_dist)

        self.assertAlmostEqual(feats_circle["convex_hull_ratio"], 1.0, places=5)
        self.assertAlmostEqual(feats_circle["aspect_ratio"], 1.0, places=2)
        self.assertEqual(feats_circle["mst_max_degree"], 2.0)
        self.assertLess(feats_circle["nn_cv"], 0.05)

        clustered_coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [2.0, 2.0],
                [500.0, 500.0],
                [505.0, 500.0],
                [500.0, 510.0],
                [512.0, 515.0],
            ],
            dtype=np.float64,
        )
        clustered_dist = compute_euc2d_matrix(clustered_coords)
        feats_clustered = extract_instance_features(clustered_coords, clustered_dist)
        self.assertLess(feats_clustered["convex_hull_ratio"], 1.0)
        self.assertGreater(feats_clustered["nn_cv"], feats_circle["nn_cv"])
        self.assertGreater(feats_clustered["hopkins_statistic"], 0.5)

    def test_all_five_warm_start_strategies(self) -> None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "tsplib"
        ensure_tsplib_dataset(data_dir)
        coords = parse_tsplib(data_dir / "ulysses16.tsp")
        dist = compute_euc2d_matrix(coords)
        n = len(coords)

        strategies = [
            ("cold_start", cold_start_tsp),
            ("nearest_neighbor", nearest_neighbor_tsp),
            ("convex_hull", convex_hull_insertion_tsp),
            ("twice_around", twice_around_the_tree_tsp),
            ("christofides", christofides_tsp),
        ]
        costs: dict[str, int] = {}
        for name, fn in strategies:
            cost, tour, telemetry = fn(coords, dist)
            validate_tour(tour, n)
            self.assertEqual(cost, compute_tour_cost(dist, tour))
            self.assertIn("heuristic_total_time", telemetry)
            self.assertIn("peak_memory_kb", telemetry)
            costs[name] = cost

        self.assertLessEqual(costs["convex_hull"], costs["cold_start"])
        self.assertLessEqual(costs["christofides"], costs["cold_start"])

    def test_branch_and_bound_exactness_across_all_warm_starts(self) -> None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "tsplib"
        ensure_tsplib_dataset(data_dir)
        coords = parse_tsplib(data_dir / "eil10.tsp")
        dist = compute_euc2d_matrix(coords)

        strategies = [
            ("cold_start", cold_start_tsp),
            ("nearest_neighbor", nearest_neighbor_tsp),
            ("convex_hull", convex_hull_insertion_tsp),
            ("twice_around", twice_around_the_tree_tsp),
            ("christofides", christofides_tsp),
        ]
        opt_costs: list[int] = []
        expanded_counts: dict[str, int] = {}

        for name, fn in strategies:
            ub_cost, ub_tour, _ = fn(coords, dist)
            solver = BranchAndBoundTSP(coords, dist, time_limit=30.0)
            res = solver.solve(initial_ub=ub_cost, initial_tour=ub_tour)
            self.assertFalse(bool(res["aborted"]))
            opt_val = res["optimal_cost"]
            self.assertIsNotNone(opt_val)
            assert isinstance(opt_val, int)
            best_tour = res["best_tour"]
            assert isinstance(best_tour, list)
            validate_tour(best_tour, len(coords))
            self.assertEqual(opt_val, compute_tour_cost(dist, best_tour))
            self.assertLessEqual(opt_val, ub_cost)
            opt_costs.append(opt_val)
            expanded_counts[name] = res["nodes_expanded"]

        self.assertEqual(len(set(opt_costs)), 1)
        self.assertLessEqual(expanded_counts["christofides"], expanded_counts["cold_start"])

        bench = benchmark_mst_subroutines(dist, sub_size=12, repetitions=50)
        self.assertGreater(bench["speedup_ratio"], 0.1)

    def test_timeout_abort_and_telemetry_preservation(self) -> None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "tsplib"
        ensure_tsplib_dataset(data_dir)
        coords = parse_tsplib(data_dir / "berlin52.tsp")
        dist = compute_euc2d_matrix(coords)
        ub, tour, _ = cold_start_tsp(coords, dist)
        solver = BranchAndBoundTSP(coords, dist, time_limit=0.001)
        res = solver.solve(initial_ub=ub, initial_tour=tour)
        self.assertTrue(res["aborted"])
        self.assertIsNone(res["optimal_cost"])
        self.assertGreaterEqual(res["nodes_expanded"], 1024)
        self.assertGreater(res["peak_memory_kb"], 0.0)

    def test_checkpoint_skip_and_analysis_tables(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        metrics_csv = base_dir / "results" / "metrics_log.csv"
        features_csv = base_dir / "results" / "features_log.csv"
        if metrics_csv.is_file() and features_csv.is_file():
            before_size = metrics_csv.stat().st_size
            out = run_pipeline(base_dir=base_dir, mode="fast-test", custom_time_limit=0.05, reset=False)
            after_size = metrics_csv.stat().st_size
            self.assertEqual(before_size, after_size)
            self.assertIn("analysis", out)
            self.assertTrue((base_dir / "results" / "plots" / "complexity_regression.pdf").is_file())
            self.assertTrue((base_dir / "results" / "tables" / "table_regression.tex").is_file())


if __name__ == "__main__":
    unittest.main()
