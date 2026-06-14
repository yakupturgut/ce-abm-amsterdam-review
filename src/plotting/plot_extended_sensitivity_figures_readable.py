#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_extended_sensitivity_figures_readable.py

Redraws extended sensitivity figures from saved CSV outputs.

The purpose of this script is formatting: it uses larger fonts and layouts that
are easier to inspect in a manuscript or appendix. It assumes that the extended
sensitivity screening has already been run.
"""
from __future__ import annotations

from pathlib import Path
import argparse
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
ROOT = SRC_ROOT.parent
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"


# =============================================================================
# Labels and plotting specifications
# =============================================================================

SCENARIO_LABELS = {
    "ce_off": "Linear reference",
    "ce_light": "Low-intensity CE",
    "baseline": "Status-quo CE",
    "ce_strong": "Targeted CE expansion",
    "ce_inclusive_strong": "Inclusive CE expansion",
}

# The extended screening script usually runs baseline and targeted CE.
SCENARIO_ORDER = ["baseline", "ce_strong"]

# KPI order follows the manuscript KPI structure.
KPI_SPECS: List[Tuple[str, str]] = [
    ("MEAN_LOCAL_CE_ACTIVATION", "Mean local CE\nactivation"),
    ("CEI_SIM", "Flow-based citywide\nCE index"),
    ("TRADES_COUNT_M", "CE flows"),
    ("UNEMPLOYMENT", "Unemployment"),
    ("D_adv_exposure", "Spatial-advantage exposure\ninequality"),
    ("D_resources", "Economic-resource\ninequality"),
]

# Backward compatibility for older screening CSVs.
METRIC_ALIASES = {
    "CE_INTENSITY": "MEAN_LOCAL_CE_ACTIVATION",
}

PARAMETER_SPECS: List[Tuple[str, str]] = [
    ("beta_cei", r"$\beta_{\mathrm{CEI}}$"),
    ("gamma_need_savings", r"$\gamma_{\mathrm{ns}}$"),
    ("move_prob", r"$p_{\mathrm{move}}$"),
    ("eta_opp_scale", "Opportunity\nscale"),
    ("barrier_sensitivity_scale", "Barrier\nscale"),
    ("hotspot_burden_scale", "Waste-burden\nscale"),
    ("observability_prob", r"$p_{\mathrm{obs}}$"),
]

PARAMETER_BASELINES: Dict[str, float] = {
    "beta_cei": -0.015,
    "gamma_need_savings": 0.60,
    "move_prob": 0.10,
    "eta_opp_scale": 1.00,
    "barrier_sensitivity_scale": 1.00,
    "hotspot_burden_scale": 1.00,
    "observability_prob": 0.50,
}

REDUCED_CASE_ORDER = [
    "Full model",
    "No CE--unemployment coupling",
    "No CE-driven demand reduction",
    "No exposure update",
    "No waste-burden penalty",
    "No CE-action observability",
]

REDUCED_CASE_SHORT = {
    "Full model": "Full\nmodel",
    "No CE--unemployment coupling": "No CE--\nunemployment\ncoupling",
    "No CE-driven demand reduction": "No CE-driven\ndemand\nreduction",
    "No exposure update": "No exposure\nupdate",
    "No waste-burden penalty": "No waste-burden\npenalty",
    "No CE-action observability": "No CE-action\nobservability",
}


# =============================================================================
# Matplotlib style
# =============================================================================

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.size": 15.0,
    "axes.titlesize": 16.0,
    "axes.labelsize": 15.0,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 13.0,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.5,
})


# =============================================================================
# IO helpers
# =============================================================================

def _find_latest_run_group() -> str:
    candidates = sorted(
        TABLES_DIR.glob("scr_*_screening_parameter_outcomes.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No scr_*_screening_parameter_outcomes.csv found in {TABLES_DIR}."
        )
    return candidates[0].name.replace("_screening_parameter_outcomes.csv", "")


def _standardize_metric_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map old metric names to manuscript KPI names."""
    if df.empty or "metric" not in df.columns:
        return df

    out = df.copy()
    out["metric"] = out["metric"].replace(METRIC_ALIASES)

    # Keep a consistent human-readable label when the column exists.
    if "metric_label" in out.columns:
        label_lookup = dict(KPI_SPECS)
        out["metric_label"] = out["metric"].map(label_lookup).fillna(out["metric_label"])

    return out


def _load(run_group: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes_path = TABLES_DIR / f"{run_group}_screening_parameter_outcomes.csv"
    scores_path = TABLES_DIR / f"{run_group}_screening_sensitivity_scores.csv"
    reduced_path = TABLES_DIR / f"{run_group}_reduced_mechanism_benchmark.csv"

    for path in [outcomes_path, scores_path, reduced_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    outcomes = _standardize_metric_names(pd.read_csv(outcomes_path))
    scores = _standardize_metric_names(pd.read_csv(scores_path))
    reduced = _standardize_metric_names(pd.read_csv(reduced_path))

    # Convert numeric columns defensively. This avoids plotting failures when CSVs
    # contain empty strings or NaN strings.
    for df, cols in [
        (outcomes, ["parameter_value", "last_mean"]),
        (scores, ["relative_sensitivity_score"]),
        (reduced, ["relative_difference_from_full_model"]),
    ]:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return outcomes, scores, reduced


def _savefig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _available_in_order(values: Iterable[str], ordered: List[str]) -> List[str]:
    values_set = set(values)
    out = [v for v in ordered if v in values_set]
    out.extend(sorted(v for v in values_set if v not in set(out)))
    return out


def _finite_max(a: np.ndarray) -> float:
    finite = np.isfinite(a)
    if not finite.any():
        return float("nan")
    return float(np.nanmax(a[finite]))


# =============================================================================
# Plotting helpers
# =============================================================================

def _plot_heatmap(scores: pd.DataFrame, scenario: str, out_dir: Path) -> None:
    sub = scores[scores["scenario"] == scenario].copy()
    if sub.empty:
        return

    params = [p for p, _ in PARAMETER_SPECS if p in set(sub["parameter"])]
    metrics = [m for m, _ in KPI_SPECS if m in set(sub["metric"])]
    if not params or not metrics:
        return

    mat = np.full((len(params), len(metrics)), np.nan, dtype=float)
    p_label = dict(PARAMETER_SPECS)
    m_label = dict(KPI_SPECS)

    for i, p in enumerate(params):
        for j, m in enumerate(metrics):
            vals = sub[
                (sub["parameter"] == p)
                & (sub["metric"] == m)
            ]["relative_sensitivity_score"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                mat[i, j] = float(np.mean(vals))

    if not np.isfinite(mat).any():
        return

    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m_label[m] for m in metrics], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(params)))
    ax.set_yticklabels([p_label[p] for p in params])
    ax.set_title(
        f"Extended OAT sensitivity screening: {SCENARIO_LABELS.get(scenario, scenario)}",
        pad=16,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.025)
    cbar.set_label("Relative sensitivity score")

    max_val = _finite_max(mat)
    threshold = max_val / 2.0 if np.isfinite(max_val) else 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                val = float(mat[i, j])
                text_color = "white" if val > threshold else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=14.0,
                    color=text_color,
                )

    fig.tight_layout()
    _savefig(fig, out_dir / f"screening_heatmap_{scenario}")


def _plot_tornado(scores: pd.DataFrame, scenario: str, out_dir: Path) -> None:
    sub = scores[scores["scenario"] == scenario].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(3, 2, figsize=(11.2, 12.2))
    axes = axes.ravel()
    p_label = dict(PARAMETER_SPECS)

    for ax, (metric, metric_label) in zip(axes, KPI_SPECS):
        msub = sub[sub["metric"] == metric].copy()
        if msub.empty:
            ax.axis("off")
            continue

        # If the score table contains repeated rows, aggregate first.
        msub = (
            msub.groupby("parameter", as_index=False)["relative_sensitivity_score"]
            .mean()
            .sort_values("relative_sensitivity_score", ascending=True)
        )
        y = np.arange(len(msub))
        labels = [p_label.get(p, p) for p in msub["parameter"].tolist()]

        ax.barh(y, msub["relative_sensitivity_score"].to_numpy(dtype=float))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=13.0)
        ax.set_xlabel("Relative sensitivity")
        ax.set_title(metric_label, pad=12)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)

    fig.suptitle(
        f"Tornado ranking of extended OAT sensitivity: {SCENARIO_LABELS.get(scenario, scenario)}",
        y=0.99,
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.0, w_pad=1.0)
    _savefig(fig, out_dir / f"screening_tornado_{scenario}")


def _plot_reduced_benchmark(reduced: pd.DataFrame, scenario: str, out_dir: Path) -> None:
    sub = reduced[reduced["scenario"] == scenario].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(3, 2, figsize=(12.2, 13.2))
    axes = axes.ravel()

    for ax, (metric, metric_label) in zip(axes, KPI_SPECS):
        msub = sub[sub["metric"] == metric].copy()
        if msub.empty:
            ax.axis("off")
            continue

        vals = []
        labels = []
        for case in REDUCED_CASE_ORDER:
            arr = msub[msub["case"] == case]["relative_difference_from_full_model"].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(float(np.mean(arr)))
                labels.append(REDUCED_CASE_SHORT.get(case, case))

        if not vals:
            ax.axis("off")
            continue

        x = np.arange(len(vals))
        ax.bar(x, vals)
        ax.axhline(0.0, linewidth=1.2, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
            fontsize=11.5,
        )
        ax.set_ylabel("Relative difference\nfrom full model")
        ax.set_title(metric_label, pad=12)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        f"Reduced-mechanism benchmark cases: {SCENARIO_LABELS.get(scenario, scenario)}",
        y=0.99,
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5, w_pad=1.5)
    _savefig(fig, out_dir / f"reduced_mechanism_benchmark_{scenario}")


def _plot_response_curve(outcomes: pd.DataFrame, scenario: str, parameter: str, out_dir: Path) -> None:
    sub = outcomes[
        (outcomes["scenario"] == scenario)
        & (outcomes["parameter"] == parameter)
    ].copy()
    if sub.empty:
        return

    p_label = dict(PARAMETER_SPECS).get(parameter, parameter)
    baseline = PARAMETER_BASELINES.get(parameter, np.nan)

    fig, axes = plt.subplots(3, 2, figsize=(11.2, 12.2))
    axes = axes.ravel()

    for ax, (metric, metric_label) in zip(axes, KPI_SPECS):
        msub = sub[sub["metric"] == metric].copy()
        if msub.empty:
            ax.axis("off")
            continue

        curve = (
            msub.groupby("parameter_value", as_index=False)["last_mean"]
            .mean()
            .sort_values("parameter_value")
        )
        curve = curve[np.isfinite(curve["parameter_value"]) & np.isfinite(curve["last_mean"])]
        if curve.empty:
            ax.axis("off")
            continue

        ax.plot(curve["parameter_value"], curve["last_mean"], marker="o", markersize=7)
        if np.isfinite(baseline):
            ax.axvline(float(baseline), linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_xlabel(str(p_label))
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label, pad=12)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Response curves for {str(p_label).replace(chr(10), ' ')}: {SCENARIO_LABELS.get(scenario, scenario)}",
        y=0.99,
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.0, w_pad=1.0)
    _savefig(fig, out_dir / f"screening_response_{parameter}_{scenario}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-group",
        default="latest",
        help="Sensitivity run group, e.g. scr_0606_1100, or latest.",
    )
    parser.add_argument(
        "--out-name",
        default="screening_readable",
        help="Output subfolder under results/figures/<run_group>.",
    )
    args = parser.parse_args()

    run_group = _find_latest_run_group() if args.run_group == "latest" else args.run_group
    outcomes, scores, reduced = _load(run_group)

    out_dir = RESULTS_DIR / "figures" / run_group / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _available_in_order(outcomes.get("scenario", []), SCENARIO_ORDER)
    params = [p for p, _ in PARAMETER_SPECS if p in set(outcomes.get("parameter", []))]

    for scenario in scenarios:
        _plot_heatmap(scores, scenario, out_dir)
        _plot_tornado(scores, scenario, out_dir)
        _plot_reduced_benchmark(reduced, scenario, out_dir)
        for parameter in params:
            _plot_response_curve(outcomes, scenario, parameter, out_dir)

    print("\n[DONE] Re-drew readable sensitivity figures without rerunning simulations.")
    print(f"[DONE] Run group : {run_group}")
    print(f"[DONE] Figures   : {out_dir}")
    print("[INFO] Use the PDFs from this folder in Overleaf.")


if __name__ == "__main__":
    main()
