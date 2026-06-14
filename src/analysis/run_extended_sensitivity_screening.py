#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_extended_sensitivity_screening.py

Runs the extended sensitivity-analysis workflow for the Amsterdam CE-ABM.

The script performs one-at-a-time parameter screening and reduced-mechanism
benchmarks. It is used to check which scenario interpretations are stable and
which are sensitive to assumptions such as exposure updating, demand reduction,
waste-burden scaling, CE-action observability and the CE--unemployment coupling.

Outputs are written as CSV tables and diagnostic figures under results/ so that
plotting scripts can redraw the sensitivity figures without rerunning the ABM.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import copy
import csv
import json
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make src importable and resolve project paths when this script is run from
# src/analysis.
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import abm as model  # noqa: E402


# =============================================================================
# User settings
# =============================================================================

# Screening uses common random numbers: within each scenario, the same seed is
# used across parameter values to reduce stochastic noise in one-run screening.
N_REPS_PER_LEVEL = 1
BASE_SEED = 24680

# Main final screening. For a quick test, temporarily set ["baseline"].
SCREENING_SCENARIOS = ["baseline", "ce_strong"]

SCENARIO_LABELS = {
    "ce_off": "Linear reference",
    "ce_light": "Light CE",
    "baseline": "Baseline CE",
    "ce_strong": "Targeted CE expansion",
    "ce_inclusive_strong": "Inclusive CE expansion",
}

LAST_WINDOW = 24
SAVE_SPATIAL_FIGURES = False

RUN_GROUP = datetime.now().strftime("scr_%m%d_%H%M")

RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures" / RUN_GROUP / "screening"
EXPERIMENT_DIR = RESULTS_DIR / "experiments" / RUN_GROUP

for p in [RUNS_DIR, TABLES_DIR, FIGURES_DIR, EXPERIMENT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = TABLES_DIR / f"{RUN_GROUP}_screening_manifest.csv"
OUTCOMES_CSV = TABLES_DIR / f"{RUN_GROUP}_screening_parameter_outcomes.csv"
SCORES_CSV = TABLES_DIR / f"{RUN_GROUP}_screening_sensitivity_scores.csv"
BENCHMARK_CSV = TABLES_DIR / f"{RUN_GROUP}_reduced_mechanism_benchmark.csv"
CONFIG_JSON = TABLES_DIR / f"{RUN_GROUP}_screening_config.json"
RUNTIME_CSV = TABLES_DIR / f"{RUN_GROUP}_screening_runtime.csv"

# KPI order follows the manuscript KPI structure.
KPI_SPECS = [
    ("MEAN_LOCAL_CE_ACTIVATION", "K1 mean local CE activation"),
    ("CEI_SIM", "K2 flow-based citywide CE index"),
    ("TRADES_COUNT_M", "K3 CE flows"),
    ("UNEMPLOYMENT", "K4 unemployment"),
    ("D_adv_exposure", "K5 spatial-advantage exposure inequality"),
    ("D_resources", "K6 economic-resource inequality"),
]

# Backward-compatible reading of old outputs. Internally, all tables and figures
# use the new canonical metric names above.
METRIC_ALIASES = {
    "MEAN_LOCAL_CE_ACTIVATION": ["MEAN_LOCAL_CE_ACTIVATION", "CE_INTENSITY"],
    "CEI_SIM": ["CEI_SIM"],
    "TRADES_COUNT_M": ["TRADES_COUNT_M"],
    "UNEMPLOYMENT": ["UNEMPLOYMENT"],
    "D_adv_exposure": ["D_adv_exposure"],
    "D_resources": ["D_resources"],
}

# Parameter screening. The baseline value is used for score normalization, even
# when it is not one of the tested values. For interpretability, most baselines
# are included in values.
PARAMETER_SPECS = [
    {
        "name": "beta_cei",
        "label": r"$\beta_{\mathrm{CEI}}$",
        "description": "Reduced-form CE--unemployment coupling",
        "values": [0.005, 0.0, -0.005, -0.015, -0.030],
        "baseline": -0.015,
    },
    {
        "name": "gamma_need_savings",
        "label": r"$\gamma_{\mathrm{ns}}$",
        "description": "Strength of CE-driven new-product demand reduction",
        "values": [0.0, 0.25, 0.50, 0.60, 0.75, 1.0],
        "baseline": 0.60,
    },
    {
        "name": "move_prob",
        "label": r"$p_{\mathrm{move}}$",
        "description": "Monthly exposure-update probability",
        "values": [0.0, 0.01, 0.05, 0.10, 0.20],
        "baseline": 0.10,
    },
    {
        "name": "eta_opp_scale",
        "label": "Opportunity scale",
        "description": "Scale multiplier for scenario-specific ETA_OPP",
        "values": [0.50, 0.75, 1.00, 1.25, 1.50],
        "baseline": 1.00,
    },
    {
        "name": "barrier_sensitivity_scale",
        "label": "Barrier scale",
        "description": "Scale multiplier for cost/time/skill/stigma barrier sensitivity",
        "values": [0.50, 0.75, 1.00, 1.25, 1.50],
        "baseline": 1.00,
    },
    {
        "name": "hotspot_burden_scale",
        "label": "Waste-burden scale",
        "description": "Scale multiplier for ADV_WASTE_PENALTY",
        "values": [0.0, 0.50, 1.00, 1.50, 2.00],
        "baseline": 1.00,
    },
    {
        "name": "observability_prob",
        "label": r"$p_{\mathrm{obs}}$",
        "description": "Probability that a CE action is socially observed",
        "values": [0.0, 0.5, 1.0],
        "baseline": 0.5,
    },
]

# These are extracted from the OAT outcomes and presented as simplified
# benchmark cases against the full model.
REDUCED_MECHANISM_CASES = [
    ("Full model", "full_model", np.nan),
    ("No CE--unemployment coupling", "beta_cei", 0.0),
    ("No CE-driven demand reduction", "gamma_need_savings", 0.0),
    ("No exposure update", "move_prob", 0.0),
    ("No waste-burden penalty", "hotspot_burden_scale", 0.0),
    ("No CE-action observability", "observability_prob", 0.0),
]

PARAMETER_CODES = {
    "full_model": "F",
    "beta_cei": "B",
    "gamma_need_savings": "G",
    "move_prob": "M",
    "eta_opp_scale": "E",
    "barrier_sensitivity_scale": "R",
    "hotspot_burden_scale": "H",
    "observability_prob": "O",
}

SCENARIO_CODES = {
    "ce_off": "LO",
    "ce_light": "LC",
    "baseline": "BL",
    "ce_strong": "TG",
    "ce_inclusive_strong": "IN",
}

# Current K2 definition. These are enforced in the screening policy config so
# old scenario dictionaries cannot accidentally reintroduce the old CE-intensity
# component into CEI_SIM.
DEFAULT_CEI_W_TRADE = 0.60
DEFAULT_CEI_W_REPAIR = 0.40
DEFAULT_CEI_W_INT = 0.0


# =============================================================================
# Matplotlib style
# =============================================================================

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
})


# =============================================================================
# Helpers
# =============================================================================

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    try:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def _savefig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _set_seed(seed: int) -> None:
    seed = int(seed)
    if hasattr(model, "SIM_SEED"):
        model.SIM_SEED = seed
    if hasattr(model, "random"):
        model.random.seed(seed)
    if hasattr(model, "np"):
        model.np.random.seed(seed)
    np.random.seed(seed)


def _disable_spatial_plotting() -> None:
    setattr(model, "SAVE_SPATIAL_FIGURES", bool(SAVE_SPATIAL_FIGURES))
    setattr(model, "SPATIAL_FIGURE_INTERVAL", 999999)
    setattr(model, "SPATIAL_FIGURE_CYCLES", [])

    if SAVE_SPATIAL_FIGURES:
        return

    def _noop(*args, **kwargs):
        return None

    for name in [
        "plot_infrastructure",
        "plot_cell_mean",
        "plot_grid",
        "plot_spatial",
        "plot_spatial_map",
        "save_spatial_figures",
        "plot_cell_metric",
    ]:
        if hasattr(model, name):
            try:
                setattr(model, name, _noop)
            except Exception:
                pass


def _snapshot_model_params() -> Dict[str, Any]:
    keys = [
        "UNEMP_BETA_CEI",
        "GAMMA_NEED_SAVINGS",
        "MOVE_PROB",
        "ETA_OPP",
        "BETA_COST",
        "BETA_TIME",
        "BETA_SKILL",
        "BETA_STIG",
        "ADV_WASTE_PENALTY",
        "OBSERVABILITY_PROB",
        "CEI_W_TRADE",
        "CEI_W_REPAIR",
        "CEI_W_INT",
        "CEI_W_MATERIAL",
        "CEI_W_REPAIR_VISITS",
    ]
    snap = {}
    for k in keys:
        if hasattr(model, k):
            snap[k] = copy.deepcopy(getattr(model, k))
    return snap


def _restore_model_params(snap: Dict[str, Any]) -> None:
    for k, v in snap.items():
        try:
            setattr(model, k, v)
        except Exception:
            pass


def _base_value(name: str, fallback: float) -> float:
    if hasattr(model, name):
        try:
            return float(getattr(model, name))
        except Exception:
            return float(fallback)
    return float(fallback)


def _scenario_base_policy(scenario: str) -> Dict[str, Any]:
    if scenario not in model.POLICY_SCENARIOS:
        raise KeyError(f"Scenario '{scenario}' not found in abm.POLICY_SCENARIOS.")
    return copy.deepcopy(model.POLICY_SCENARIOS[scenario])


def _enforce_current_k2_weights(policy_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Prevent old policy dictionaries from reintroducing CE_INTENSITY into K2."""
    policy_cfg["CEI_W_TRADE"] = float(DEFAULT_CEI_W_TRADE)
    policy_cfg["CEI_W_REPAIR"] = float(DEFAULT_CEI_W_REPAIR)
    policy_cfg["CEI_W_INT"] = float(DEFAULT_CEI_W_INT)

    # Alternative names are harmless when unsupported, and useful if the model
    # file uses clearer manuscript metric names.
    policy_cfg["CEI_W_MATERIAL"] = float(DEFAULT_CEI_W_TRADE)
    policy_cfg["CEI_W_REPAIR_VISITS"] = float(DEFAULT_CEI_W_REPAIR)

    for name, value in [
        ("CEI_W_TRADE", DEFAULT_CEI_W_TRADE),
        ("CEI_W_REPAIR", DEFAULT_CEI_W_REPAIR),
        ("CEI_W_INT", DEFAULT_CEI_W_INT),
        ("CEI_W_MATERIAL", DEFAULT_CEI_W_TRADE),
        ("CEI_W_REPAIR_VISITS", DEFAULT_CEI_W_REPAIR),
    ]:
        if hasattr(model, name):
            try:
                setattr(model, name, float(value))
            except Exception:
                pass
    return policy_cfg


def _apply_parameter(policy_cfg: Dict[str, Any], parameter_name: str, value: float) -> Dict[str, Any]:
    """
    Apply a sensitivity parameter to both policy_cfg and model globals.

    This makes the script compatible with ABM versions that read from either
    policy_cfg or module-level global variables.
    """
    v = float(value)

    if parameter_name == "full_model":
        return policy_cfg

    if parameter_name == "beta_cei":
        policy_cfg["UNEMP_BETA_CEI"] = v
        setattr(model, "UNEMP_BETA_CEI", v)

    elif parameter_name == "gamma_need_savings":
        policy_cfg["GAMMA_NEED_SAVINGS"] = v
        setattr(model, "GAMMA_NEED_SAVINGS", v)

    elif parameter_name == "move_prob":
        policy_cfg["MOVE_PROB"] = v
        setattr(model, "MOVE_PROB", v)

    elif parameter_name == "eta_opp_scale":
        base = float(policy_cfg.get("ETA_OPP", _base_value("ETA_OPP", 0.05)))
        new_val = base * v
        policy_cfg["ETA_OPP"] = new_val
        setattr(model, "ETA_OPP", new_val)

    elif parameter_name == "barrier_sensitivity_scale":
        for k, default in [
            ("BETA_COST", 1.0),
            ("BETA_TIME", 1.0),
            ("BETA_SKILL", 0.7),
            ("BETA_STIG", 0.5),
        ]:
            base = _base_value(k, default)
            new_val = base * v
            policy_cfg[k] = new_val
            setattr(model, k, new_val)

    elif parameter_name == "hotspot_burden_scale":
        base = _base_value("ADV_WASTE_PENALTY", 1.2)
        new_val = base * v
        policy_cfg["ADV_WASTE_PENALTY"] = new_val
        setattr(model, "ADV_WASTE_PENALTY", new_val)

    elif parameter_name == "observability_prob":
        policy_cfg["OBSERVABILITY_PROB"] = v
        setattr(model, "OBSERVABILITY_PROB", v)

    else:
        raise ValueError(f"Unknown parameter: {parameter_name}")

    return policy_cfg


def _safe_value_string(value: float) -> str:
    """Compact value label for Windows-safe short run folder names."""
    try:
        if value is None or not np.isfinite(float(value)):
            return "base"
        v = float(value)
    except Exception:
        return "base"
    if abs(v) < 1e-12:
        return "z"
    sign = "m" if v < 0 else "p"
    scaled = int(round(abs(v) * 1000))
    return f"{sign}{scaled}"


def _run_one(
    scenario: str,
    parameter_name: str,
    parameter_value: float,
    rep: int,
    seed: int,
) -> Dict[str, Any]:
    snap = _snapshot_model_params()
    t0 = time.perf_counter()
    status = "ok"
    error = ""
    run_dir = ""
    master_csv = ""

    run_tag = f"{PARAMETER_CODES.get(parameter_name, parameter_name[:2].upper())}{_safe_value_string(parameter_value)}_{SCENARIO_CODES.get(scenario, scenario[:2].upper())}_r{rep}"
    print(f"[screen] scenario={scenario} param={parameter_name} value={parameter_value} rep={rep} seed={seed}")

    try:
        _set_seed(seed)
        if hasattr(model, "BASE_RUN_NAME"):
            model.BASE_RUN_NAME = RUN_GROUP

        policy_cfg = _scenario_base_policy(scenario)
        policy_cfg.update({
            "SEED": int(seed),
            "REPLICATE": int(rep),
            "RUN_TAG": run_tag,
            "_RUN_TAG": run_tag,
            "SAVE_SPATIAL_FIGURES": bool(SAVE_SPATIAL_FIGURES),
            "SPATIAL_FIGURE_INTERVAL": 999999,
            "SPATIAL_FIGURE_CYCLES": [],
        })
        policy_cfg = _enforce_current_k2_weights(policy_cfg)

        if parameter_name != "full_model":
            policy_cfg = _apply_parameter(policy_cfg, parameter_name, float(parameter_value))
            policy_cfg = _enforce_current_k2_weights(policy_cfg)

        model.run_sim(scenario, policy_cfg)

        run_dir_path = Path(getattr(model, "RUN_DIR", RUNS_DIR / f"{RUN_GROUP}_{run_tag}_{scenario}"))
        run_dir = str(run_dir_path)
        master_path = run_dir_path / "MASTER_metrics.csv"
        master_csv = str(master_path if master_path.exists() else "")

    except Exception as exc:
        status = "failed"
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        tb_path = EXPERIMENT_DIR / f"error_{run_tag}.txt"
        tb_path.write_text(traceback.format_exc(), encoding="utf-8")
        print("[ERROR]", error)

    finally:
        _restore_model_params(snap)

    duration = time.perf_counter() - t0
    return {
        "run_group": RUN_GROUP,
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
        "parameter": parameter_name,
        "parameter_value": parameter_value,
        "replicate": rep,
        "seed": seed,
        "run_tag": run_tag,
        "run_dir": run_dir,
        "master_csv": master_csv,
        "duration_sec": duration,
        "status": status,
        "error": error,
    }


def _resolve_metric_column(df: pd.DataFrame, canonical_metric: str) -> Optional[str]:
    for candidate in METRIC_ALIASES.get(canonical_metric, [canonical_metric]):
        if candidate in df.columns:
            return candidate
    return None


def _read_metric_summary(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    csv_path = Path(str(row["master_csv"]))
    out = []
    if row["status"] != "ok" or not csv_path.exists():
        return out

    df = pd.read_csv(csv_path)
    if "cycle" not in df.columns:
        return out

    for metric, metric_label in KPI_SPECS:
        source_col = _resolve_metric_column(df, metric)
        if source_col is None:
            continue
        values = pd.to_numeric(df[source_col], errors="coerce").to_numpy(dtype=float)
        if values.size == 0 or np.all(~np.isfinite(values)):
            continue

        w = min(LAST_WINDOW, values.size)
        initial = float(values[0]) if np.isfinite(values[0]) else np.nan
        final = float(values[-1]) if np.isfinite(values[-1]) else np.nan
        last_mean = float(np.nanmean(values[-w:]))
        delta = final - initial if np.isfinite(final) and np.isfinite(initial) else np.nan

        out.append({
            "run_group": row["run_group"],
            "scenario": row["scenario"],
            "scenario_label": row["scenario_label"],
            "parameter": row["parameter"],
            "parameter_value": row["parameter_value"],
            "replicate": row["replicate"],
            "seed": row["seed"],
            "metric": metric,
            "metric_label": metric_label,
            "source_column": source_col,
            "last_window": LAST_WINDOW,
            "last_mean": last_mean,
            "initial": initial,
            "final": final,
            "delta_final_minus_initial": delta,
        })
    return out


def _parameter_baseline_value(parameter: str) -> Optional[float]:
    for spec in PARAMETER_SPECS:
        if spec["name"] == parameter:
            return float(spec.get("baseline", np.nan))
    return None


def _baseline_outcome(outcomes: pd.DataFrame, scenario: str, metric: str) -> float:
    sub = outcomes[
        (outcomes["scenario"] == scenario)
        & (outcomes["parameter"] == "full_model")
        & (outcomes["metric"] == metric)
    ]
    if not sub.empty:
        return float(sub["last_mean"].mean())

    # fallback: use the beta_CEI baseline setting if the dedicated full-model run
    # is missing for any reason.
    beta_baseline = _parameter_baseline_value("beta_cei")
    if beta_baseline is not None and np.isfinite(beta_baseline):
        sub = outcomes[
            (outcomes["scenario"] == scenario)
            & (outcomes["parameter"] == "beta_cei")
            & (outcomes["metric"] == metric)
            & np.isclose(pd.to_numeric(outcomes["parameter_value"], errors="coerce"), beta_baseline)
        ]
        if not sub.empty:
            return float(sub["last_mean"].mean())

    return float("nan")


def _compute_scores(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if outcomes.empty or "parameter" not in outcomes.columns:
        return pd.DataFrame(rows)
    spec_lookup = {s["name"]: s for s in PARAMETER_SPECS}

    oat = outcomes[outcomes["parameter"] != "full_model"].copy()
    for (scenario, parameter, metric), grp in oat.groupby(["scenario", "parameter", "metric"]):
        if grp.empty:
            continue

        spec = spec_lookup.get(parameter, {})
        vals_by_level = grp.groupby("parameter_value", as_index=False)["last_mean"].mean()
        vals = vals_by_level["last_mean"].to_numpy(dtype=float)
        if vals.size == 0 or np.all(~np.isfinite(vals)):
            continue

        base = _baseline_outcome(outcomes, scenario, metric)
        if not np.isfinite(base):
            base = float(np.nanmean(vals))

        rng = float(np.nanmax(vals) - np.nanmin(vals))
        score = rng / (abs(base) + 1e-9)

        rows.append({
            "run_group": RUN_GROUP,
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
            "parameter": parameter,
            "parameter_label": spec.get("label", parameter),
            "parameter_description": spec.get("description", ""),
            "metric": metric,
            "metric_label": dict(KPI_SPECS).get(metric, metric),
            "baseline_last_mean_full_model": base,
            "min_last_mean": float(np.nanmin(vals)),
            "max_last_mean": float(np.nanmax(vals)),
            "range_last_mean": rng,
            "relative_sensitivity_score": score,
            "n_levels": int(vals_by_level.shape[0]),
        })
    return pd.DataFrame(rows)


def _extract_reduced_mechanism_benchmarks(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if outcomes.empty or "parameter" not in outcomes.columns:
        return pd.DataFrame(rows)

    for scenario in SCREENING_SCENARIOS:
        for case_label, parameter, value in REDUCED_MECHANISM_CASES:
            for metric, metric_label in KPI_SPECS:
                if parameter == "full_model":
                    sub = outcomes[
                        (outcomes["scenario"] == scenario)
                        & (outcomes["parameter"] == "full_model")
                        & (outcomes["metric"] == metric)
                    ]
                else:
                    sub = outcomes[
                        (outcomes["scenario"] == scenario)
                        & (outcomes["parameter"] == parameter)
                        & (outcomes["metric"] == metric)
                        & np.isclose(pd.to_numeric(outcomes["parameter_value"], errors="coerce"), float(value))
                    ]
                if sub.empty:
                    continue

                v = float(sub["last_mean"].mean())
                full = _baseline_outcome(outcomes, scenario, metric)
                diff = v - full if np.isfinite(full) else np.nan
                rel = diff / (abs(full) + 1e-9) if np.isfinite(full) else np.nan

                rows.append({
                    "run_group": RUN_GROUP,
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
                    "case": case_label,
                    "parameter": parameter,
                    "parameter_value": value,
                    "metric": metric,
                    "metric_label": metric_label,
                    "last_mean": v,
                    "full_model_last_mean": full,
                    "difference_from_full_model": diff,
                    "relative_difference_from_full_model": rel,
                })
    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def _plot_heatmap(scores: pd.DataFrame, scenario: str) -> None:
    sub = scores[scores["scenario"] == scenario].copy()
    if sub.empty:
        return

    params = [s["name"] for s in PARAMETER_SPECS if s["name"] in set(sub["parameter"])]
    metrics = [m for m, _ in KPI_SPECS if m in set(sub["metric"])]
    mat = np.full((len(params), len(metrics)), np.nan, dtype=float)

    ylabels = []
    spec_lookup = {s["name"]: s for s in PARAMETER_SPECS}
    for i, p in enumerate(params):
        ylabels.append(spec_lookup.get(p, {}).get("label", p))
        for j, m in enumerate(metrics):
            vals = sub[(sub["parameter"] == p) & (sub["metric"] == m)]["relative_sensitivity_score"].to_numpy(dtype=float)
            if vals.size:
                mat[i, j] = vals[0]

    xlabels = [dict(KPI_SPECS).get(m, m) for m in metrics]

    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(params)))
    ax.set_yticklabels(ylabels)
    ax.set_title(f"Extended OAT screening heatmap: {SCENARIO_LABELS.get(scenario, scenario)}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative sensitivity score")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5)

    fig.tight_layout()
    _savefig(fig, FIGURES_DIR / f"screening_heatmap_{scenario}")


def _plot_tornado(scores: pd.DataFrame, scenario: str) -> None:
    sub = scores[scores["scenario"] == scenario].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6))
    axes = axes.ravel()

    for ax, (metric, metric_label) in zip(axes, KPI_SPECS):
        msub = sub[sub["metric"] == metric].copy()
        if msub.empty:
            ax.axis("off")
            continue
        msub = msub.sort_values("relative_sensitivity_score", ascending=True)
        y = np.arange(len(msub))
        ax.barh(y, msub["relative_sensitivity_score"].to_numpy(dtype=float))
        ax.set_yticks(y)
        ax.set_yticklabels(msub["parameter_label"].astype(str).tolist(), fontsize=7)
        ax.set_xlabel("Relative sensitivity")
        ax.set_title(metric_label)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)

    fig.suptitle(f"Extended OAT screening sensitivity: {SCENARIO_LABELS.get(scenario, scenario)}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig(fig, FIGURES_DIR / f"screening_tornado_{scenario}")


def _plot_response_curves(outcomes: pd.DataFrame, scenario: str, parameter: str) -> None:
    sub = outcomes[(outcomes["scenario"] == scenario) & (outcomes["parameter"] == parameter)].copy()
    if sub.empty:
        return

    spec = next((s for s in PARAMETER_SPECS if s["name"] == parameter), None)
    param_label = spec["label"] if spec else parameter
    baseline = spec.get("baseline", np.nan) if spec else np.nan

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2))
    axes = axes.ravel()

    for ax, (metric, metric_label) in zip(axes, KPI_SPECS):
        msub = sub[sub["metric"] == metric].copy()
        if msub.empty:
            ax.axis("off")
            continue
        curve = msub.groupby("parameter_value", as_index=False)["last_mean"].mean().sort_values("parameter_value")
        ax.plot(curve["parameter_value"], curve["last_mean"], marker="o")
        if np.isfinite(float(baseline)):
            ax.axvline(float(baseline), linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_xlabel(str(param_label))
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(alpha=0.25)

    fig.suptitle(f"Response curves for {param_label}: {SCENARIO_LABELS.get(scenario, scenario)}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig(fig, FIGURES_DIR / f"screening_response_{parameter}_{scenario}")


def _plot_reduced_benchmark(bench: pd.DataFrame, scenario: str) -> None:
    sub = bench[bench["scenario"] == scenario].copy()
    if sub.empty:
        return

    cases = [c[0] for c in REDUCED_MECHANISM_CASES if c[0] in set(sub["case"])]
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.8))
    axes = axes.ravel()

    for ax, (metric, metric_label) in zip(axes, KPI_SPECS):
        msub = sub[sub["metric"] == metric].copy()
        if msub.empty:
            ax.axis("off")
            continue
        vals = []
        labels = []
        for c in cases:
            arr = msub[msub["case"] == c]["relative_difference_from_full_model"].to_numpy(dtype=float)
            if arr.size and np.isfinite(arr[0]):
                vals.append(arr[0])
                labels.append(c.replace("No ", "No\n").replace("Full model", "Full\nmodel"))
        ax.bar(np.arange(len(vals)), vals)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.7)
        ax.set_ylabel("Relative difference")
        ax.set_title(metric_label)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)

    fig.suptitle(f"Reduced-mechanism benchmark cases: {SCENARIO_LABELS.get(scenario, scenario)}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, FIGURES_DIR / f"reduced_mechanism_benchmark_{scenario}")


# =============================================================================
# Config and main
# =============================================================================

def _write_config() -> None:
    cfg = {
        "run_group": RUN_GROUP,
        "n_reps_per_level": N_REPS_PER_LEVEL,
        "base_seed": BASE_SEED,
        "screening_scenarios": SCREENING_SCENARIOS,
        "scenario_labels": SCENARIO_LABELS,
        "last_window": LAST_WINDOW,
        "parameter_specs": PARAMETER_SPECS,
        "reduced_mechanism_cases": [
            {"case": c, "parameter": p, "value": None if not np.isfinite(v) else v}
            for c, p, v in REDUCED_MECHANISM_CASES
        ],
        "kpi_specs": KPI_SPECS,
        "metric_aliases": METRIC_ALIASES,
        "current_k2_weights": {
            "CEI_W_TRADE": DEFAULT_CEI_W_TRADE,
            "CEI_W_REPAIR": DEFAULT_CEI_W_REPAIR,
            "CEI_W_INT": DEFAULT_CEI_W_INT,
        },
        "interpretation": (
            "Single-replication extended OAT screening with reduced-mechanism "
            "benchmark cases; not a replicated global sensitivity analysis. "
            "K1 is the mean local CE-activation field, and K2 is the flow-based "
            "citywide CE index."
        ),
    }
    CONFIG_JSON.write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=_json_safe), encoding="utf-8")


def _seed_for(scenario_idx: int, rep: int) -> int:
    # Common random numbers within a scenario for all parameter settings.
    return int(BASE_SEED + 100000 * scenario_idx + rep)


def _append_runtime(row: Dict[str, Any]) -> None:
    with open(RUNTIME_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_group", "scenario", "parameter", "parameter_value",
                "replicate", "seed", "duration_sec", "status",
            ],
        )
        writer.writerow({
            "run_group": RUN_GROUP,
            "scenario": row.get("scenario", ""),
            "parameter": row.get("parameter", ""),
            "parameter_value": row.get("parameter_value", ""),
            "replicate": row.get("replicate", ""),
            "seed": row.get("seed", ""),
            "duration_sec": row.get("duration_sec", ""),
            "status": row.get("status", ""),
        })


def main() -> None:
    t0 = time.perf_counter()
    _disable_spatial_plotting()
    _write_config()

    manifest_rows = []
    outcome_rows = []

    with open(RUNTIME_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_group", "scenario", "parameter", "parameter_value",
                "replicate", "seed", "duration_sec", "status",
            ],
        )
        writer.writeheader()

    for s_idx, scenario in enumerate(SCREENING_SCENARIOS):
        for rep in range(N_REPS_PER_LEVEL):
            seed = _seed_for(s_idx, rep)

            # Dedicated full-model run for each scenario.
            row = _run_one(scenario, "full_model", np.nan, rep, seed)
            manifest_rows.append(row)
            outcome_rows.extend(_read_metric_summary(row))
            _append_runtime(row)

            # OAT screening runs.
            for spec in PARAMETER_SPECS:
                pname = spec["name"]
                for value in spec["values"]:
                    row = _run_one(scenario, pname, float(value), rep, seed)
                    manifest_rows.append(row)
                    outcome_rows.extend(_read_metric_summary(row))
                    _append_runtime(row)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFEST_CSV, index=False)

    outcomes = pd.DataFrame(outcome_rows)
    outcomes.to_csv(OUTCOMES_CSV, index=False)

    scores = _compute_scores(outcomes)
    scores.to_csv(SCORES_CSV, index=False)

    reduced = _extract_reduced_mechanism_benchmarks(outcomes)
    reduced.to_csv(BENCHMARK_CSV, index=False)

    if not scores.empty:
        for scenario in SCREENING_SCENARIOS:
            _plot_heatmap(scores, scenario)
            _plot_tornado(scores, scenario)

    if not reduced.empty:
        for scenario in SCREENING_SCENARIOS:
            _plot_reduced_benchmark(reduced, scenario)

    if not outcomes.empty:
        for scenario in SCREENING_SCENARIOS:
            for spec in PARAMETER_SPECS:
                _plot_response_curves(outcomes, scenario, spec["name"])

    total = time.perf_counter() - t0
    print("\n[DONE] Extended sensitivity screening completed.")
    print(f"[DONE] Run group        : {RUN_GROUP}")
    print(f"[DONE] Manifest         : {MANIFEST_CSV}")
    print(f"[DONE] Outcomes         : {OUTCOMES_CSV}")
    print(f"[DONE] Scores           : {SCORES_CSV}")
    print(f"[DONE] Reduced benchmark: {BENCHMARK_CSV}")
    print(f"[DONE] Figures          : {FIGURES_DIR}")
    print(f"[DONE] Runtime          : {total/60:.2f} minutes")


if __name__ == "__main__":
    main()
