#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_compact_sensitivity_figures.py

Creates compact sensitivity figures from saved sensitivity-screening outputs.

This script reads the CSV files created by run_extended_sensitivity_screening.py
and combines many separate sensitivity diagnostics into a small set of readable
summary figures. It does not rerun any ABM simulations.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import shutil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
ROOT = SRC_ROOT.parent
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

SCENARIO_LABELS: Dict[str, str] = {
    "ce_off": "Linear reference",
    "ce_light": "Light CE",
    "baseline": "Baseline CE",
    "ce_strong": "Targeted CE expansion",
    "ce_inclusive_strong": "Inclusive CE expansion",
}

# Extended sensitivity screening normally focuses on baseline and targeted CE.
SCENARIO_ORDER: List[str] = ["baseline", "ce_strong"]

# Canonical KPI list used in the paper and in the ABM output.
K1_METRIC = "MEAN_LOCAL_CE_ACTIVATION"
OLD_K1_METRIC = "CE_INTENSITY"

KPI_SPECS: List[Tuple[str, str]] = [
    (K1_METRIC, "Mean local\nCE activation"),
    ("CEI_SIM", "Flow-based\nCE index"),
    ("TRADES_COUNT_M", "CE flows"),
    ("UNEMPLOYMENT", "Unemployment"),
    ("D_adv_exposure", "Spatial-advantage\nexposure inequality"),
    ("D_resources", "Economic-resource\ninequality"),
]

KPI_SHORT_LABELS: Dict[str, str] = {
    K1_METRIC: "Local CE\nactivation",
    "CEI_SIM": "Flow CEI",
    "TRADES_COUNT_M": "CE flows",
    "UNEMPLOYMENT": "Unemp.",
    "D_adv_exposure": "Spatial exposure\nineq.",
    "D_resources": "Resource\nineq.",
}

KPI_LONG_LABELS: Dict[str, str] = {
    K1_METRIC: "Mean local CE activation",
    "CEI_SIM": "Flow-based citywide CE index",
    "TRADES_COUNT_M": "CE flows",
    "UNEMPLOYMENT": "Unemployment",
    "D_adv_exposure": "Spatial-advantage exposure inequality",
    "D_resources": "Economic-resource inequality",
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

REDUCED_CASE_ORDER: List[str] = [
    "No CE--unemployment coupling",
    "No CE-driven demand reduction",
    "No exposure update",
    "No waste-burden penalty",
    "No CE-action observability",
]

REDUCED_CASE_SHORT: Dict[str, str] = {
    "No CE--unemployment coupling": "No CE--\nunemployment",
    "No CE-driven demand reduction": "No demand\nreduction",
    "No exposure update": "No exposure\nupdate",
    "No waste-burden penalty": "No waste-burden\npenalty",
    "No CE-action observability": "No action\nobservability",
}

# Selected mechanism-response plots used for the compact response-curve figure.
KEY_RESPONSE_SPECS: List[Tuple[str, str, str]] = [
    ("move_prob", K1_METRIC, "Exposure update $p_{move}$ → local CE activation"),
    ("move_prob", "CEI_SIM", "Exposure update $p_{move}$ → flow-based CE index"),
    ("move_prob", "TRADES_COUNT_M", "Exposure update $p_{move}$ → CE flows"),
    ("beta_cei", "UNEMPLOYMENT", r"CE--unemployment coupling $\beta_{CEI}$ → unemployment"),
    ("hotspot_burden_scale", "D_adv_exposure", "Waste-burden scale → spatial-exposure inequality"),
    ("gamma_need_savings", "D_resources", r"Demand reduction $\gamma_{ns}$ → resource inequality"),
    ("observability_prob", K1_METRIC, r"Observability $p_{obs}$ → local CE activation"),
    ("observability_prob", "CEI_SIM", r"Observability $p_{obs}$ → flow-based CE index"),
]

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.size": 11.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.2,
    "ytick.labelsize": 10.2,
    "legend.fontsize": 10.2,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
})


def _find_latest_run_group() -> str:
    candidates = sorted(
        TABLES_DIR.glob("scr_*_screening_parameter_outcomes.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No scr_*_screening_parameter_outcomes.csv found in results/tables. "
            "Run src/analysis/run_extended_sensitivity_screening.py first."
        )
    return candidates[0].name.replace("_screening_parameter_outcomes.csv", "")


def _canonicalize_metric_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert legacy CE_INTENSITY rows to the current K1 name.

    This keeps the compact plotting script usable for both newly rerun
    sensitivity outputs and older CSV files created before the K1 renaming.
    """
    if df.empty or "metric" not in df.columns:
        return df

    df = df.copy()
    df["metric"] = df["metric"].replace({OLD_K1_METRIC: K1_METRIC})

    if "metric_label" in df.columns:
        df.loc[df["metric"] == K1_METRIC, "metric_label"] = KPI_LONG_LABELS[K1_METRIC]
        df.loc[df["metric"] == "CEI_SIM", "metric_label"] = KPI_LONG_LABELS["CEI_SIM"]

    return df


def _load(run_group: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes_path = TABLES_DIR / f"{run_group}_screening_parameter_outcomes.csv"
    scores_path = TABLES_DIR / f"{run_group}_screening_sensitivity_scores.csv"
    reduced_path = TABLES_DIR / f"{run_group}_reduced_mechanism_benchmark.csv"

    missing = [p for p in (outcomes_path, scores_path, reduced_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing sensitivity output file(s):\n" + "\n".join(str(p) for p in missing))

    outcomes = _canonicalize_metric_column(pd.read_csv(outcomes_path))
    scores = _canonicalize_metric_column(pd.read_csv(scores_path))
    reduced = _canonicalize_metric_column(pd.read_csv(reduced_path))

    for df in (outcomes, scores, reduced):
        for col in [
            "last_mean",
            "parameter_value",
            "relative_sensitivity_score",
            "relative_difference_from_full_model",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return outcomes, scores, reduced


def _save_single(fig: plt.Figure, out_base: Path, dpi: int = 600) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _score_matrix(scores: pd.DataFrame, scenario: str) -> Tuple[np.ndarray, List[str], List[str]]:
    sub = scores[scores["scenario"] == scenario].copy()
    params = [p for p, _ in PARAMETER_SPECS if p in set(sub["parameter"])]
    metrics = [m for m, _ in KPI_SPECS if m in set(sub["metric"])]
    mat = np.full((len(params), len(metrics)), np.nan)

    for i, p in enumerate(params):
        for j, m in enumerate(metrics):
            vals = sub[(sub["parameter"] == p) & (sub["metric"] == m)]["relative_sensitivity_score"].to_numpy(float)
            if vals.size:
                mat[i, j] = vals[0]

    return mat, params, metrics


def _available_scenarios(df: pd.DataFrame) -> List[str]:
    if df.empty or "scenario" not in df.columns:
        return []
    found = set(df["scenario"].astype(str))
    ordered = [s for s in SCENARIO_ORDER if s in found]
    ordered.extend(sorted(s for s in found if s not in SCENARIO_ORDER))
    return ordered


def make_screening_summary(scores: pd.DataFrame, out_dir: Path, save: bool = True) -> plt.Figure:
    scenarios = _available_scenarios(scores)
    if not scenarios:
        raise ValueError("No scenarios found in sensitivity scores.")

    matrices = []
    max_val = 0.0
    for scenario in scenarios:
        mat, params, metrics = _score_matrix(scores, scenario)
        matrices.append((scenario, mat, params, metrics))
        if np.isfinite(mat).any():
            max_val = max(max_val, float(np.nanmax(mat)))
    vmax = max(max_val, 1e-6)

    fig_height = 3.2 * max(1, len(scenarios))
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(11.6, fig_height), squeeze=False)
    param_labels = dict(PARAMETER_SPECS)
    metric_labels = KPI_SHORT_LABELS

    image = None
    for ax, (scenario, mat, params, metrics) in zip(axes.ravel(), matrices):
        image = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=vmax)
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario), loc="left", pad=8, fontweight="bold")
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels([metric_labels.get(m, m) for m in metrics], rotation=0, ha="center")
        ax.set_yticks(np.arange(len(params)))
        ax.set_yticklabels([param_labels.get(p, p) for p in params])
        ax.grid(False)

        threshold = 0.55 * vmax
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    color = "white" if mat[i, j] > threshold else "black"
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9.5, color=color)

    fig.suptitle("Compact sensitivity screening summary", y=0.995, fontsize=14.5, fontweight="bold")
    fig.subplots_adjust(left=0.16, right=0.82, top=0.91, bottom=0.11, hspace=0.48)

    if image is not None:
        cax = fig.add_axes([0.86, 0.18, 0.020, 0.66])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label("Relative sensitivity score", labelpad=10)

    if save:
        _save_single(fig, out_dir / "compact_sensitivity_screening_summary")
    return fig


def make_reduced_benchmark_summary(reduced: pd.DataFrame, out_dir: Path, save: bool = True) -> plt.Figure:
    scenarios = _available_scenarios(reduced)
    if not scenarios:
        raise ValueError("No scenarios found in reduced benchmark file.")

    kpi_labels = KPI_SHORT_LABELS
    max_abs = 0.0
    mats = []
    for scenario in scenarios:
        sub = reduced[reduced["scenario"] == scenario].copy()
        cases = [c for c in REDUCED_CASE_ORDER if c in set(sub["case"])]
        metrics = [m for m, _ in KPI_SPECS if m in set(sub["metric"])]
        mat = np.full((len(cases), len(metrics)), np.nan)
        for i, case in enumerate(cases):
            for j, metric in enumerate(metrics):
                vals = sub[(sub["case"] == case) & (sub["metric"] == metric)]["relative_difference_from_full_model"].to_numpy(float)
                if vals.size:
                    mat[i, j] = vals[0]
        mats.append((scenario, cases, metrics, mat))
        if np.isfinite(mat).any():
            max_abs = max(max_abs, float(np.nanmax(np.abs(mat))))
    vmax = max(max_abs, 1e-6)

    fig_height = 3.3 * max(1, len(scenarios))
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12.2, fig_height), squeeze=False)
    image = None
    for ax, (scenario, cases, metrics, mat) in zip(axes.ravel(), mats):
        image = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario), loc="left", pad=8, fontweight="bold")
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels([kpi_labels.get(m, m) for m in metrics], rotation=0, ha="center")
        ax.set_yticks(np.arange(len(cases)))
        ax.set_yticklabels([REDUCED_CASE_SHORT.get(c, c) for c in cases])
        ax.grid(False)

        threshold = 0.55 * vmax
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    color = "white" if abs(mat[i, j]) > threshold else "black"
                    ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=9.2, color=color)

    fig.suptitle("Compact reduced-mechanism benchmark summary", y=0.995, fontsize=14.5, fontweight="bold")
    fig.subplots_adjust(left=0.23, right=0.82, top=0.91, bottom=0.11, hspace=0.50)

    if image is not None:
        cax = fig.add_axes([0.86, 0.18, 0.020, 0.66])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label("Relative difference from full model", labelpad=10)

    if save:
        _save_single(fig, out_dir / "compact_reduced_mechanism_benchmark")
    return fig


def _curve_data(outcomes: pd.DataFrame, scenario: str, parameter: str, metric: str) -> pd.DataFrame:
    sub = outcomes[
        (outcomes["scenario"] == scenario)
        & (outcomes["parameter"] == parameter)
        & (outcomes["metric"] == metric)
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=["parameter_value", "last_mean"])
    return (
        sub.groupby("parameter_value", as_index=False)["last_mean"]
        .mean()
        .sort_values("parameter_value")
    )


def make_key_response_summary(outcomes: pd.DataFrame, out_dir: Path, save: bool = True) -> plt.Figure:
    scenarios = _available_scenarios(outcomes)
    fig, axes = plt.subplots(4, 2, figsize=(10.8, 12.0))
    axes = axes.ravel()

    param_labels = dict(PARAMETER_SPECS)
    kpi_labels = KPI_SHORT_LABELS

    for ax, (parameter, metric, title) in zip(axes, KEY_RESPONSE_SPECS):
        plotted = False
        for scenario in scenarios:
            curve = _curve_data(outcomes, scenario, parameter, metric)
            if curve.empty:
                continue
            ax.plot(
                curve["parameter_value"],
                curve["last_mean"],
                marker="o",
                markersize=4.8,
                label=SCENARIO_LABELS.get(scenario, scenario),
            )
            plotted = True
        if not plotted:
            ax.axis("off")
            continue
        ax.set_title(title, loc="left", pad=7, fontsize=11.2)
        ax.set_xlabel(str(param_labels.get(parameter, parameter)).replace("\n", " "))
        ax.set_ylabel(kpi_labels.get(metric, metric).replace("\n", " "))
        ax.grid(alpha=0.25)

    handles, labels = [], []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle("Key mechanism-specific response curves", y=0.995, fontsize=14.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.97], h_pad=2.0, w_pad=1.3)
    if save:
        _save_single(fig, out_dir / "compact_key_response_curves")
    return fig


def _add_fig_to_pdf(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_three_page_pdf(outcomes: pd.DataFrame, scores: pd.DataFrame, reduced: pd.DataFrame, out_dir: Path) -> Path:
    pdf_path = out_dir / "compact_sensitivity_appendix_3page.pdf"
    with PdfPages(pdf_path) as pdf:
        fig1 = make_screening_summary(scores, out_dir, save=False)
        _add_fig_to_pdf(pdf, fig1)
        fig2 = make_reduced_benchmark_summary(reduced, out_dir, save=False)
        _add_fig_to_pdf(pdf, fig2)
        fig3 = make_key_response_summary(outcomes, out_dir, save=False)
        _add_fig_to_pdf(pdf, fig3)
    return pdf_path


def _copy_pdfs_to_figures_root(out_dir: Path) -> None:
    target = FIGURES_DIR / "compact_sensitivity_for_overleaf"
    target.mkdir(parents=True, exist_ok=True)
    for pdf in out_dir.glob("compact_*.pdf"):
        shutil.copy2(pdf, target / pdf.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-group", default="latest", help="Sensitivity run group, e.g. scr_0608_1430, or latest.")
    parser.add_argument("--out-name", default="screening_compact", help="Output subfolder under results/figures/<run_group>.")
    parser.add_argument(
        "--copy-to-figures-root",
        action="store_true",
        help="Copy compact PDFs to results/figures/compact_sensitivity_for_overleaf.",
    )
    args = parser.parse_args()

    run_group = _find_latest_run_group() if args.run_group == "latest" else args.run_group
    outcomes, scores, reduced = _load(run_group)

    out_dir = FIGURES_DIR / run_group / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    make_screening_summary(scores, out_dir, save=True)
    make_reduced_benchmark_summary(reduced, out_dir, save=True)
    make_key_response_summary(outcomes, out_dir, save=True)
    combined_pdf = make_three_page_pdf(outcomes, scores, reduced, out_dir)

    if args.copy_to_figures_root:
        _copy_pdfs_to_figures_root(out_dir)

    print("\n[DONE] Created compact sensitivity figures without rerunning simulations.")
    print(f"[DONE] Run group : {run_group}")
    print(f"[DONE] Figures   : {out_dir}")
    print(f"[DONE] 3-page PDF: {combined_pdf}")
    if args.copy_to_figures_root:
        print(f"[DONE] Copies    : {FIGURES_DIR / 'compact_sensitivity_for_overleaf'}")


if __name__ == "__main__":
    main()
