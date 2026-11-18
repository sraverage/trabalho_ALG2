from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor, export_text


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

FEATURE_COLS: list[str] = [
    "aspect_ratio",
    "convex_hull_ratio",
    "spatial_entropy",
    "hopkins_statistic",
    "mst_max_degree",
    "nn_cv",
]

STRATEGY_LABELS: dict[str, str] = {
    "cold_start": "Cold Start",
    "nearest_neighbor": "Nearest Neighbor",
    "convex_hull": "Convex Hull Ins.",
    "twice_around": "Twice-Around-Tree",
    "christofides": "Christofides",
}


def _poly2(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.square(x) + b * x + c


def _poly3(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(x, 3) + b * np.square(x) + c


def _save_figure(fig: plt.Figure, plots_dir: Path, stem: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{stem}.png", dpi=300)
    fig.savefig(plots_dir / f"{stem}.pdf")
    plt.close(fig)


def _fit_polynomial_curve(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    model_fn: Callable[..., np.ndarray],
    model_name: str,
    target_label: str,
) -> dict[str, str | float]:
    params, _ = curve_fit(model_fn, x_vals, y_vals, p0=[1e-7, 1e-5, float(np.min(y_vals))], maxfev=20000)
    y_pred = model_fn(x_vals, *params)
    r2 = float(r2_score(y_vals, y_pred)) if len(y_vals) > 2 else 1.0
    rmse = float(root_mean_squared_error(y_vals, y_pred))
    return {
        "target": target_label,
        "model": model_name,
        "param_a": float(params[0]),
        "param_b": float(params[1]),
        "param_c": float(params[2]),
        "r2": r2,
        "rmse": rmse,
    }


def _fit_exponential_loglinear(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    target_label: str,
) -> tuple[dict[str, str | float], np.ndarray]:
    safe_y = np.maximum(y_vals.astype(np.float64), 1e-9)
    log_y = np.log(safe_y)
    reg = LinearRegression()
    reg.fit(x_vals.reshape(-1, 1), log_y)
    log_pred = reg.predict(x_vals.reshape(-1, 1))
    y_pred = np.exp(log_pred)

    a_val = float(np.exp(reg.intercept_))
    b_val = float(np.exp(reg.coef_[0]))
    r2_log = float(r2_score(log_y, log_pred))
    rmse_raw = float(root_mean_squared_error(safe_y, y_pred))

    row: dict[str, str | float] = {
        "target": target_label,
        "model": "a * b^N (log-linear)",
        "param_a": a_val,
        "param_b": b_val,
        "param_c": 0.0,
        "r2": r2_log,
        "rmse": rmse_raw,
    }
    return row, y_pred


def _build_complexity_regressions(
    metrics_df: pd.DataFrame,
    plots_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []

    twice_df = metrics_df[metrics_df["strategy"] == "twice_around"].sort_values("n_vertices")
    chris_df = metrics_df[metrics_df["strategy"] == "christofides"].sort_values("n_vertices")

    x_tw = twice_df["n_vertices"].to_numpy(dtype=np.float64)
    y_mst = twice_df["mst_time"].to_numpy(dtype=np.float64) * 1000.0
    y_tw = twice_df["heuristic_time"].to_numpy(dtype=np.float64) * 1000.0
    rows.append(_fit_polynomial_curve(x_tw, y_mst, _poly2, "a*N^2 + b*N + c", "MST Prim (ms)"))
    rows.append(_fit_polynomial_curve(x_tw, y_tw, _poly2, "a*N^2 + b*N + c", "Twice-Around Total (ms)"))

    x_ch = chris_df["n_vertices"].to_numpy(dtype=np.float64)
    y_match = chris_df["matching_time"].to_numpy(dtype=np.float64) * 1000.0
    y_ch = chris_df["heuristic_time"].to_numpy(dtype=np.float64) * 1000.0
    y_mem = chris_df["peak_memory_kb"].to_numpy(dtype=np.float64)
    rows.append(_fit_polynomial_curve(x_ch, y_match, _poly3, "a*N^3 + b*N^2 + c", "Matching Blossom (ms)"))
    rows.append(_fit_polynomial_curve(x_ch, y_ch, _poly3, "a*N^3 + b*N^2 + c", "Christofides Total (ms)"))
    rows.append(_fit_polynomial_curve(x_ch, y_mem, _poly2, "a*N^2 + b*N + c", "Pico Memoria (KiB)"))

    bb_solved = metrics_df[(~metrics_df["aborted"]) & (metrics_df["n_vertices"] <= 40)].copy()
    if bb_solved.empty:
        bb_main = chris_df[chris_df["n_vertices"] <= 40].sort_values("n_vertices")
    else:
        bb_main = (
            bb_solved.sort_values(["n_vertices", "bb_total_time", "nodes_expanded"])
            .drop_duplicates(subset=["instance"], keep="first")
            .sort_values("n_vertices")
        )
    x_bb = bb_main["n_vertices"].to_numpy(dtype=np.float64)
    y_bb_time = bb_main["bb_total_time"].to_numpy(dtype=np.float64)
    y_bb_nodes = bb_main["nodes_expanded"].to_numpy(dtype=np.float64)
    row_t, pred_t = _fit_exponential_loglinear(x_bb, y_bb_time, "B&B Tempo Exato (s)")
    row_n, pred_n = _fit_exponential_loglinear(x_bb, y_bb_nodes, "B&B Nos Expandidos")
    rows.append(row_t)
    rows.append(row_n)
    _ = pred_t

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    x_grid = np.linspace(float(x_ch.min()), float(x_ch.max()), 100)
    p_mst = [float(rows[0]["param_a"]), float(rows[0]["param_b"]), float(rows[0]["param_c"])]
    p_match = [float(rows[2]["param_a"]), float(rows[2]["param_b"]), float(rows[2]["param_c"])]

    axes[0].scatter(x_tw, y_mst, color="#1f77b4", label="MST Observado (ms)", zorder=3)
    axes[0].plot(x_grid, _poly2(x_grid, *p_mst), "--", color="#1f77b4", label=f"Ajuste O(N^2) R²={rows[0]['r2']:.3f}")
    axes[0].scatter(x_ch, y_match, color="#d62728", marker="s", label="Matching Observado (ms)", zorder=3)
    axes[0].plot(x_grid, _poly3(x_grid, *p_match), "-", color="#d62728", label=f"Ajuste O(N^3) R²={rows[2]['r2']:.3f}")
    axes[0].set_xlabel("Número de Vértices (N)")
    axes[0].set_ylabel("Tempo de Subrotina (ms)")
    axes[0].set_title("Complexidade de Subrotinas (MST vs Matching)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(x_bb, y_bb_nodes, color="#2ca02c", label="Nós Expandidos (Melhor Warm-Start)", zorder=3)
    axes[1].plot(x_bb, pred_n, "--", color="#111111", label="Ajuste Exponencial a·b^N")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Número de Vértices (N <= 40)")
    axes[1].set_ylabel("Nós Expandidos (escala log)")
    axes[1].set_title("Crescimento Exponencial da Árvore B&B")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    p_mem = [float(rows[4]["param_a"]), float(rows[4]["param_b"]), float(rows[4]["param_c"])]
    axes[2].scatter(x_ch, y_mem, color="#9467bd", marker="^", label="Pico RAM Observado (KiB)", zorder=3)
    axes[2].plot(x_grid, _poly2(x_grid, *p_mem), "-.", color="#9467bd", label=f"Ajuste O(N^2) R²={rows[4]['r2']:.3f}")
    axes[2].set_xlabel("Número de Vértices (N)")
    axes[2].set_ylabel("Pico de Memória Alocada (KiB)")
    axes[2].set_title("Consumo Espacial vs N")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, plots_dir, "complexity_regression")

    return pd.DataFrame(rows)


def _plot_warm_start_and_subroutines(
    metrics_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    sub_df = metrics_df[metrics_df["n_vertices"] <= 29].copy()
    instances = list(sub_df.sort_values("n_vertices")["instance"].unique())
    strategies = list(STRATEGY_LABELS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=False)
    bar_width = 0.15
    x_pos = np.arange(len(instances))

    for idx, strat in enumerate(strategies):
        strat_df = sub_df[sub_df["strategy"] == strat].set_index("instance").reindex(instances)
        exp_vals = np.maximum(strat_df["nodes_expanded"].to_numpy(dtype=np.float64), 1.0)
        pruned_vals = np.maximum(strat_df["nodes_pruned"].to_numpy(dtype=np.float64), 1.0)
        offset = (idx - 2) * bar_width
        axes[0].bar(x_pos + offset, exp_vals, width=bar_width, label=STRATEGY_LABELS[strat])
        axes[1].bar(x_pos + offset, pruned_vals, width=bar_width, label=STRATEGY_LABELS[strat])

    for ax, title_str, ylabel_str in [
        (axes[0], "Nós Expandidos por Estratégia de Warm-Start", "Nós Expandidos (log)"),
        (axes[1], "Nós Podados por Estratégia de Warm-Start", "Nós Podados (log)"),
    ]:
        ax.set_yscale("log")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(instances, rotation=35, ha="right")
        ax.set_ylabel(ylabel_str)
        ax.set_title(title_str)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", axis="y", alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, plots_dir, "warm_start_pruning")

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.4))
    solved_instances = list(
        metrics_df[~metrics_df["aborted"]]
        .sort_values("n_vertices")["instance"]
        .unique()
    )
    if not solved_instances:
        solved_instances = instances[:6]

    x_sp = np.arange(len(solved_instances))
    for idx, strat in enumerate(strategies[1:]):
        s_df = (
            metrics_df[metrics_df["strategy"] == strat]
            .set_index("instance")
            .reindex(solved_instances)
        )
        speedups = s_df["net_speedup"].to_numpy(dtype=np.float64)
        axes2[0].bar(
            x_sp + (idx - 1.5) * 0.18,
            speedups,
            width=0.18,
            label=STRATEGY_LABELS[strat],
        )
    axes2[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Break-even (1.0x)")
    axes2[0].set_xticks(x_sp)
    axes2[0].set_xticklabels(solved_instances, rotation=35, ha="right")
    axes2[0].set_ylabel("Speedup Líquido vs Cold Start")
    axes2[0].set_title("Trade-off Tempo Heurístico vs Poda (Speedup Líquido)")
    axes2[0].legend(fontsize=8)
    axes2[0].grid(True, axis="y", alpha=0.3)

    chris_all = metrics_df[metrics_df["strategy"] == "christofides"].sort_values("n_vertices")
    inst_all = chris_all["instance"].tolist()
    mst_ms = chris_all["mst_time"].to_numpy(dtype=np.float64) * 1000.0
    match_ms = chris_all["matching_time"].to_numpy(dtype=np.float64) * 1000.0
    euler_ms = chris_all["eulerian_time"].to_numpy(dtype=np.float64) * 1000.0

    x_all = np.arange(len(inst_all))
    axes2[1].bar(x_all, mst_ms, label="MST (Prim)", color="#1f77b4")
    axes2[1].bar(x_all, match_ms, bottom=mst_ms, label="Min-Weight Matching (Blossom)", color="#d62728")
    axes2[1].bar(
        x_all,
        euler_ms,
        bottom=mst_ms + match_ms,
        label="Multigrafo + Circuito Euleriano",
        color="#ff7f0e",
    )
    axes2[1].set_xticks(x_all)
    axes2[1].set_xticklabels(inst_all, rotation=45, ha="right", fontsize=8)
    axes2[1].set_ylabel("Tempo Acumulado (ms)")
    axes2[1].set_title("Decomposição de Subrotinas no Algoritmo de Christofides")
    axes2[1].legend(fontsize=8)
    axes2[1].grid(True, axis="y", alpha=0.3)

    fig2.tight_layout()
    _save_figure(fig2, plots_dir, "warm_start_speedup")


def _run_sensitivity_analysis(
    metrics_df: pd.DataFrame,
    features_df: pd.DataFrame,
    plots_dir: Path,
) -> tuple[pd.DataFrame, str]:
    pivot_gaps = metrics_df.pivot(index="instance", columns="strategy", values="approx_gap_pct")
    pivot_gaps.columns = [f"gap_{col}" for col in pivot_gaps.columns]

    chris_rows = metrics_df[metrics_df["strategy"] == "christofides"].set_index("instance")
    pruning_eff = chris_rows["nodes_pruned"] / np.maximum(
        chris_rows["nodes_expanded"] + chris_rows["nodes_pruned"], 1
    )

    merged = features_df.set_index("instance").join(pivot_gaps)
    merged["bb_pruning_ratio"] = pruning_eff

    X = merged[FEATURE_COLS].to_numpy(dtype=np.float64)
    targets = [
        ("gap_christofides", "Gap Christofides (%)"),
        ("gap_twice_around", "Gap Twice-Around (%)"),
        ("gap_convex_hull", "Gap Convex Hull (%)"),
        ("gap_nearest_neighbor", "Gap Nearest Neighbor (%)"),
        ("bb_pruning_ratio", "Taxa de Poda B&B"),
    ]

    importance_rows: list[dict[str, str | float]] = []
    tree_rules_text = ""

    for col_name, label in targets:
        y = merged[col_name].to_numpy(dtype=np.float64)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        dt = DecisionTreeRegressor(max_depth=3, random_state=42)
        dt.fit(X, y)

        if col_name == "gap_christofides":
            tree_rules_text = export_text(dt, feature_names=FEATURE_COLS)

        rf_r2 = float(r2_score(y, rf.predict(X)))
        dt_r2 = float(r2_score(y, dt.predict(X)))
        row: dict[str, str | float] = {
            "target": label,
            "rf_r2": rf_r2,
            "dt_r2": dt_r2,
        }
        for feat_name, imp_val in zip(FEATURE_COLS, rf.feature_importances_):
            row[feat_name] = float(imp_val)
        importance_rows.append(row)

    imp_df = pd.DataFrame(importance_rows)

    corr_cols = FEATURE_COLS + [t[0] for t in targets]
    spearman_corr = merged[corr_cols].corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    im = axes[0].imshow(spearman_corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0].set_xticks(np.arange(len(corr_cols)))
    axes[0].set_yticks(np.arange(len(corr_cols)))
    axes[0].set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(corr_cols, fontsize=8)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            val = float(spearman_corr.iloc[i, j])
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    axes[0].set_title("Matriz de Correlação de Spearman (Features vs Gaps)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    y_pos = np.arange(len(FEATURE_COLS))
    width = 0.25
    for idx, (col_name, label) in enumerate(targets[:3]):
        imp_vals = [float(imp_df.iloc[idx][f]) for f in FEATURE_COLS]
        axes[1].barh(y_pos + (idx - 1) * width, imp_vals, height=width, label=label)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(FEATURE_COLS)
    axes[1].set_xlabel("Importância Relativa (Random Forest Gini/MSE)")
    axes[1].set_title("Peso das Features Topológicas no Gap de Aproximação")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, plots_dir, "sensitivity_analysis")

    return imp_df, tree_rules_text


def _export_latex_tables(
    metrics_df: pd.DataFrame,
    features_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    reg_lines = [
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"\textbf{Métrica / Subrotina} & \textbf{Modelo Ajustado} & \textbf{Coef. Principal ($a$)} & \textbf{$R^2$} & \textbf{RMSE} \\",
        r"\hline",
    ]
    for _, row in reg_df.iterrows():
        target_s = str(row["target"]).replace("&", r"\&").replace("<=", r"\le ")
        model_s = str(row["model"]).replace("^2", "$^2$").replace("^3", "$^3$").replace("^N", "$^N$")
        reg_lines.append(
            f"{target_s} & {model_s} & {float(row['param_a']):.3e} & {float(row['r2']):.4f} & {float(row['rmse']):.4f} \\\\"
        )
    reg_lines.extend([r"\hline", r"\end{tabular}"])
    (tables_dir / "table_regression.tex").write_text("\n".join(reg_lines) + "\n", encoding="utf-8")

    feat_lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"\textbf{Instância} & $N$ & \textbf{Aspect} & \textbf{Hull Ratio} & \textbf{Entropia} & \textbf{Hopkins} & \textbf{MST Grau Máx} & \textbf{NN CV} \\",
        r"\hline",
    ]
    for _, row in features_df.sort_values("n_vertices").iterrows():
        feat_lines.append(
            f"{row['instance']} & {int(row['n_vertices'])} & {float(row['aspect_ratio']):.3f} & "
            f"{float(row['convex_hull_ratio']):.3f} & {float(row['spatial_entropy']):.3f} & "
            f"{float(row['hopkins_statistic']):.3f} & {int(row['mst_max_degree'])} & {float(row['nn_cv']):.3f} \\\\"
        )
    feat_lines.extend([r"\hline", r"\end{tabular}"])
    (tables_dir / "table_features.tex").write_text("\n".join(feat_lines) + "\n", encoding="utf-8")

    chris_df = metrics_df[metrics_df["strategy"] == "christofides"].sort_values("n_vertices")
    tw_df = metrics_df[metrics_df["strategy"] == "twice_around"].set_index("instance")
    hull_df = metrics_df[metrics_df["strategy"] == "convex_hull"].set_index("instance")
    nn_df = metrics_df[metrics_df["strategy"] == "nearest_neighbor"].set_index("instance")
    cold_df = metrics_df[metrics_df["strategy"] == "cold_start"].set_index("instance")

    inst_lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\hline",
        r"\textbf{Instância} & $N$ & \textbf{Ótimo} & \textbf{Gap NN\%} & \textbf{Gap Hull\%} & \textbf{Gap TAT\%} & \textbf{Gap Chris\%} & \textbf{Nós Exp (Melhor WS)} & \textbf{Tempo B\&B (s)} & \textbf{Speedup Máx} \\",
        r"\hline",
    ]
    for _, row in chris_df.iterrows():
        inst = str(row["instance"])
        inst_rows = metrics_df[metrics_df["instance"] == inst]
        solved_rows = inst_rows[~inst_rows["aborted"]]
        if not solved_rows.empty:
            best_row = solved_rows.sort_values(["bb_total_time", "nodes_expanded"]).iloc[0]
            opt_val = best_row["optimal_cost"]
            opt_str = str(int(float(opt_val))) if pd.notna(opt_val) and str(opt_val) != "NA" else "NA"
            max_speedup = float(solved_rows["net_speedup"].max())
        else:
            best_row = hull_df.loc[inst]
            opt_str = "NA"
            max_speedup = float(inst_rows["net_speedup"].max())
        inst_lines.append(
            f"{inst} & {int(row['n_vertices'])} & {opt_str} & "
            f"{float(nn_df.loc[inst, 'approx_gap_pct']):.2f} & "
            f"{float(hull_df.loc[inst, 'approx_gap_pct']):.2f} & "
            f"{float(tw_df.loc[inst, 'approx_gap_pct']):.2f} & "
            f"{float(row['approx_gap_pct']):.2f} & "
            f"{int(best_row['nodes_expanded'])} & "
            f"{float(best_row['bb_total_time']):.3f} & "
            f"{max_speedup:.2f}x \\\\"
        )
    inst_lines.extend([r"\hline", r"\end{tabular}"])
    (tables_dir / "table_instances_summary.tex").write_text("\n".join(inst_lines) + "\n", encoding="utf-8")
    _ = cold_df

    imp_lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"\textbf{Alvo Modelado} & $R^2$ (RF) & \textbf{Aspect} & \textbf{Hull} & \textbf{Entropia} & \textbf{Hopkins} & \textbf{MST Hub} & \textbf{NN CV} \\",
        r"\hline",
    ]
    for _, row in imp_df.iterrows():
        target_clean = str(row["target"]).replace("%", r"\%").replace("&", r"\&")
        imp_lines.append(
            f"{target_clean} & {float(row['rf_r2']):.3f} & "
            f"{float(row['aspect_ratio']):.3f} & {float(row['convex_hull_ratio']):.3f} & "
            f"{float(row['spatial_entropy']):.3f} & {float(row['hopkins_statistic']):.3f} & "
            f"{float(row['mst_max_degree']):.3f} & {float(row['nn_cv']):.3f} \\\\"
        )
    imp_lines.extend([r"\hline", r"\end{tabular}"])
    (tables_dir / "table_sensitivity.tex").write_text("\n".join(imp_lines) + "\n", encoding="utf-8")


def run_full_analysis(
    metrics_csv: str | Path,
    features_csv: str | Path,
    results_dir: str | Path,
) -> dict[str, pd.DataFrame | str]:
    """Executa toda a modelagem estatística, regressão assintótica e análise de sensibilidade."""
    res_path = Path(results_dir)
    plots_dir = res_path / "plots"
    tables_dir = res_path / "tables"

    metrics_df = pd.read_csv(metrics_csv, keep_default_na=False)
    numeric_cols = [
        "n_vertices",
        "reference_opt",
        "initial_ub",
        "best_ub_found_so_far",
        "approx_gap_pct",
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
    for col in numeric_cols:
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce").fillna(0.0)
    metrics_df["aborted"] = metrics_df["aborted"].astype(str).str.lower() == "true"

    features_df = pd.read_csv(features_csv)

    reg_df = _build_complexity_regressions(metrics_df, plots_dir)
    _plot_warm_start_and_subroutines(metrics_df, plots_dir)
    imp_df, tree_rules = _run_sensitivity_analysis(metrics_df, features_df, plots_dir)

    reg_df.to_csv(res_path / "regression_models.csv", index=False)
    imp_df.to_csv(res_path / "sensitivity_importance.csv", index=False)
    (res_path / "decision_tree_rules.txt").write_text(tree_rules, encoding="utf-8")

    _export_latex_tables(metrics_df, features_df, reg_df, imp_df, tables_dir)

    return {
        "regression_df": reg_df,
        "importance_df": imp_df,
        "decision_tree_rules": tree_rules,
    }
