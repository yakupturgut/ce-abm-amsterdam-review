#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_verification_benchmarking.py

Builds verification and plausibility diagnostics for an existing run group.

The script reads completed replicated outputs and produces checks that help a
reader understand whether the model initialization and scenario outputs are
internally consistent. These are plausibility and transparency diagnostics, not
external empirical validation against observed Amsterdam time series.

Typical diagnostics include patch-type composition, initial household resources,
distances to CE infrastructure and waste-burden locations, spatial-advantage
layers and last-window KPI summaries.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import json
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

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
# Settings
# =============================================================================

BENCH_GROUP = datetime.now().strftime("bench_%Y%m%d_%H%M%S")

RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures" / BENCH_GROUP / "benchmarking"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SUMMARY_CSV = TABLES_DIR / f"{BENCH_GROUP}_benchmark_input_summary.csv"
OUTPUT_CHECKS_CSV = TABLES_DIR / f"{BENCH_GROUP}_benchmark_output_plausibility.csv"
REPORT_JSON = TABLES_DIR / f"{BENCH_GROUP}_benchmark_report.json"

LAST_WINDOW = 24

SCENARIO_LABELS = {
    "ce_off": "Linear reference",
    "ce_light": "Light CE",
    "baseline": "Baseline CE",
    "ce_strong": "Targeted CE expansion",
    "ce_inclusive_strong": "Inclusive CE expansion",
}

SCENARIO_LABELS_SHORT = {
    "ce_off": "Linear\nreference",
    "ce_light": "Light\nCE",
    "baseline": "Baseline\nCE",
    "ce_strong": "Targeted CE\nexpansion",
    "ce_inclusive_strong": "Inclusive CE\nexpansion",
}

SCENARIO_ORDER = ["ce_off", "ce_light", "baseline", "ce_strong", "ce_inclusive_strong"]

PATCH_LABELS = {
    0: "Empty",
    1: "Primary",
    2: "Secondary",
    3: "Waste",
    4: "Hub",
}

# Conceptual colors for patch types:
# Empty      : neutral grey
# Primary    : green, primary economic-resource patches
# Secondary  : blue, CE-enabling infrastructure/service patches
# Waste      : red-orange, waste-burden / hotspot cells
# Hub        : purple, repair cafés / circular service hubs
PATCH_COLORS = {
    0: "#d9d9d9",
    1: "#4daf4a",
    2: "#377eb8",
    3: "#e41a1c",
    4: "#984ea3",
}

# Canonical KPI checks used in the manuscript.
KPI_CHECKS = [
    "MEAN_LOCAL_CE_ACTIVATION",  # K1
    "CEI_SIM",                   # K2
    "TRADES_COUNT_M",            # K3 summary count
    "UNEMPLOYMENT",              # K4
    "D_adv_exposure",            # K5
    "D_resources",               # K6
]

# Backward compatibility with earlier result files.
METRIC_ALIASES = {
    "MEAN_LOCAL_CE_ACTIVATION": ["MEAN_LOCAL_CE_ACTIVATION", "CE_INTENSITY"],
    "CEI_SIM": ["CEI_SIM"],
    "TRADES_COUNT_M": ["TRADES_COUNT_M"],
    "UNEMPLOYMENT": ["UNEMPLOYMENT"],
    "D_adv_exposure": ["D_adv_exposure"],
    "D_resources": ["D_resources"],
}

METRIC_LABELS = {
    "MEAN_LOCAL_CE_ACTIVATION": "Mean local CE\nactivation (K1)",
    "CEI_SIM": "Flow-based CE\nindex (K2)",
    "TRADES_COUNT_M": "CE flows\n(K3)",
    "UNEMPLOYMENT": "Unemployment\n(K4)",
    "D_adv_exposure": "Spatial-advantage\nGini (K5)",
    "D_resources": "Resource\nGini (K6)",
}


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
})


# =============================================================================
# Utility functions
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


def _unique_cells(cells) -> List[Tuple[int, int]]:
    """Return sorted unique (x, y) grid cells from possibly repeated records."""
    out = []
    for c in list(cells):
        try:
            x, y = int(c[0]), int(c[1])
            out.append((x, y))
        except Exception:
            continue
    return sorted(set(out))


def safe_gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if np.min(x) < 0:
        x = x - np.min(x)
    mean = np.mean(x)
    if mean <= 1e-12:
        return 0.0
    diff = np.abs(x[:, None] - x[None, :]).sum()
    return float(diff / (2.0 * x.size * x.size * mean))


def _nearest_distance_to_cells(points: np.ndarray, cells: List[Tuple[int, int]]) -> np.ndarray:
    """Euclidean nearest-cell distance on the grid. Returns NaN if cells empty."""
    if points.size == 0:
        return np.array([], dtype=float)
    if not cells:
        return np.full(points.shape[0], np.nan, dtype=float)
    cell_arr = np.asarray(cells, dtype=float)
    out = np.empty(points.shape[0], dtype=float)
    for i, p in enumerate(points.astype(float)):
        d = np.sqrt(((cell_arr - p) ** 2).sum(axis=1))
        out[i] = float(np.min(d))
    return out


def _distance_field(cells: List[Tuple[int, int]], grid_size: int) -> np.ndarray:
    if not cells:
        return np.full((grid_size, grid_size), np.nan, dtype=float)
    coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)], dtype=float)
    d = _nearest_distance_to_cells(coords, cells)
    return d.reshape((grid_size, grid_size))


def _normalize01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    out = np.zeros_like(a, dtype=float)
    if not finite.any():
        return out
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    if abs(mx - mn) <= 1e-12:
        out[finite] = 0.5
        return out
    out[finite] = (a[finite] - mn) / (mx - mn)
    return out


def _metric_source_column(df: pd.DataFrame, canonical_metric: str) -> Optional[str]:
    """Return the first available source column for a canonical metric."""
    for candidate in METRIC_ALIASES.get(canonical_metric, [canonical_metric]):
        if candidate in df.columns:
            return candidate
    return None


def _read_metric_values(df: pd.DataFrame, canonical_metric: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Read a canonical metric from a result dataframe, using aliases if needed."""
    col = _metric_source_column(df, canonical_metric)
    if col is None:
        return None, None
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if vals.size == 0 or np.all(~np.isfinite(vals)):
        return None, col
    return vals, col


def _boxplot_with_labels(ax: plt.Axes, vals: List[np.ndarray], labels: List[str]) -> None:
    """Matplotlib-compatible boxplot label handling across versions."""
    try:
        ax.boxplot(vals, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(vals, labels=labels, showmeans=True)


def _ordered_scenarios(keys) -> List[str]:
    keys = list(keys)
    ordered = [s for s in SCENARIO_ORDER if s in keys]
    ordered.extend(sorted([s for s in keys if s not in SCENARIO_ORDER]))
    return ordered


def _load_agents() -> pd.DataFrame:
    ag_path = Path(getattr(model, "AGNT_FILE"))
    with open(ag_path, "rb") as f:
        raw = pickle.load(f)

    rows = []
    for rec in list(raw):
        try:
            x, y, r0, vision, recognition, generation = rec[:6]
        except Exception:
            continue
        rows.append({
            "x": int(x),
            "y": int(y),
            "initial_resource": float(r0),
            "vision": float(vision),
            "recognition": float(recognition),
            "generation": generation,
        })
    return pd.DataFrame(rows)


def _compute_advantage_map() -> Tuple[np.ndarray, np.ndarray]:
    ptype = np.asarray(model.ptype_base)
    grid = int(ptype.shape[0])

    hubs = _unique_cells(getattr(model, "hub_cells_base", []))
    containers = _unique_cells(getattr(model, "container_cells_base", []))
    recycling = _unique_cells(getattr(model, "recycling_cells_base", []))
    waste = _unique_cells(getattr(model, "waste_cells_base", []))

    dh = _distance_field(hubs, grid)
    dc = _distance_field(containers, grid)
    dr = _distance_field(recycling, grid)
    dw = _distance_field(waste, grid)

    hub_w = float(getattr(model, "ADV_HUB_WEIGHT", 1.0))
    cont_w = float(getattr(model, "ADV_CONT_WEIGHT", 0.6))
    rec_w = float(getattr(model, "ADV_RECYCL_WEIGHT", 0.5))
    sec_bonus = float(getattr(model, "ADV_SEC_BONUS", 0.5))
    waste_penalty = float(getattr(model, "ADV_WASTE_PENALTY", 1.2))

    def invdist(d):
        d = np.asarray(d, dtype=float)
        out = np.zeros_like(d, dtype=float)
        finite = np.isfinite(d)
        out[finite] = 1.0 / (1.0 + d[finite])
        return out

    raw = (
        hub_w * invdist(dh)
        + cont_w * invdist(dc)
        + rec_w * invdist(dr)
        + sec_bonus * (ptype == getattr(model, "SECONDARY", 2)).astype(float)
        - waste_penalty * invdist(dw)
    )
    return raw, _normalize01(raw)


# =============================================================================
# Input benchmarking
# =============================================================================

def _input_benchmarking() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ptype = np.asarray(model.ptype_base)
    agents = _load_agents()

    hubs = _unique_cells(getattr(model, "hub_cells_base", []))
    containers = _unique_cells(getattr(model, "container_cells_base", []))
    recycling = _unique_cells(getattr(model, "recycling_cells_base", []))
    waste = _unique_cells(getattr(model, "waste_cells_base", []))

    points = agents[["x", "y"]].to_numpy(dtype=float) if not agents.empty else np.empty((0, 2))
    d_hub = _nearest_distance_to_cells(points, hubs)
    d_cont = _nearest_distance_to_cells(points, containers)
    d_recy = _nearest_distance_to_cells(points, recycling)
    d_waste = _nearest_distance_to_cells(points, waste)

    raw_adv, adv_plus = _compute_advantage_map()
    if not agents.empty:
        hx = agents["x"].to_numpy(dtype=int)
        hy = agents["y"].to_numpy(dtype=int)
        h_adv = adv_plus[hx, hy]
    else:
        h_adv = np.array([], dtype=float)

    rows = []

    def add(check, value, diagnostic_type, interpretation=""):
        rows.append({
            "check": check,
            "value": value,
            "diagnostic_type": diagnostic_type,
            "interpretation": interpretation,
        })

    add("grid_shape", f"{ptype.shape[0]} x {ptype.shape[1]}", "input_structure", "Regular grid used as computational support.")
    add("grid_is_60x60", bool(ptype.shape == (60, 60)), "verification", "Expected manuscript grid size.")
    add("n_households_initialized", int(len(agents)), "input_structure", "Number of representative household agents in initial population.")
    add("initial_resource_mean", float(agents["initial_resource"].mean()) if not agents.empty else np.nan, "input_distribution", "Mean initialized accumulated resources.")
    add("initial_resource_gini", safe_gini(agents["initial_resource"].to_numpy()) if not agents.empty else np.nan, "input_distribution", "Initial resource inequality among synthetic households.")
    add("recognition_mean", float(agents["recognition"].mean()) if "recognition" in agents else np.nan, "input_distribution", "Mean initial recognition.")

    add("n_unique_hub_cells", len(hubs), "infrastructure", "Unique grid cells containing repair cafés / circular service hubs.")
    add("n_unique_container_cells", len(containers), "infrastructure", "Unique grid cells containing sorting containers.")
    add("n_unique_recycling_cells", len(recycling), "infrastructure", "Unique grid cells containing recycling centres.")
    add("n_unique_waste_cells", len(waste), "infrastructure", "Unique grid cells containing waste hotspots.")

    # Raw records are reported separately because many empirical infrastructure
    # records can fall into the same 60 x 60 grid cell.
    add("n_raw_hub_records", len(list(getattr(model, "hub_cells_base", []))), "infrastructure_metadata", "Raw mapped hub records before grid-cell deduplication.")
    add("n_raw_container_records", len(list(getattr(model, "container_cells_base", []))), "infrastructure_metadata", "Raw mapped container records before grid-cell deduplication.")
    add("n_raw_recycling_records", len(list(getattr(model, "recycling_cells_base", []))), "infrastructure_metadata", "Raw mapped recycling records before grid-cell deduplication.")
    add("n_raw_waste_records", len(list(getattr(model, "waste_cells_base", []))), "infrastructure_metadata", "Raw mapped waste-hotspot records before grid-cell deduplication.")

    add("mean_household_distance_to_hub", float(np.nanmean(d_hub)) if d_hub.size else np.nan, "spatial_access", "Average grid distance from households to nearest hub.")
    add("mean_household_distance_to_container", float(np.nanmean(d_cont)) if d_cont.size else np.nan, "spatial_access", "Average grid distance from households to nearest sorting container.")
    add("mean_household_distance_to_recycling", float(np.nanmean(d_recy)) if d_recy.size else np.nan, "spatial_access", "Average grid distance from households to nearest recycling centre.")
    add("mean_household_distance_to_waste", float(np.nanmean(d_waste)) if d_waste.size else np.nan, "spatial_burden", "Average grid distance from households to nearest waste hotspot.")

    add("advantage_plus_min", float(np.nanmin(adv_plus)), "verification", "Normalized advantage should be within [0,1].")
    add("advantage_plus_max", float(np.nanmax(adv_plus)), "verification", "Normalized advantage should be within [0,1].")
    add("household_advantage_mean", float(np.nanmean(h_adv)) if h_adv.size else np.nan, "spatial_access", "Mean normalized spatial-advantage exposure among households.")
    add("household_advantage_gini", safe_gini(h_adv) if h_adv.size else np.nan, "spatial_access", "Initial inequality in normalized spatial-advantage exposure.")

    # Optional empirical-grid checks if the preprocessed NPZ contains extra empirical layers.
    try:
        data = np.load(Path(getattr(model, "GRID_FILE")), allow_pickle=True)
        available = data.files
        add("npz_available_layers", "; ".join(available), "input_metadata", "Layers available in the preprocessed grid file.")
        for key in available:
            key_l = key.lower()
            if "pop" in key_l or "income" in key_l or "bbga" in key_l:
                arr = np.asarray(data[key], dtype=float)
                if arr.shape == ptype.shape:
                    add(f"{key}_mean", float(np.nanmean(arr)), "optional_empirical_layer", f"Mean of optional layer {key}.")
                    add(f"{key}_nonmissing_share", float(np.isfinite(arr).mean()), "optional_empirical_layer", f"Share non-missing for optional layer {key}.")
    except Exception as exc:
        add("npz_extra_layer_scan_error", str(exc), "metadata_warning", "Could not scan optional empirical layers.")

    diag = {
        "agents": agents,
        "distances": {
            "Hub": d_hub,
            "Container": d_cont,
            "Recycling centre": d_recy,
            "Waste hotspot": d_waste,
        },
        "advantage_plus": adv_plus,
        "raw_advantage": raw_adv,
        "h_advantage": h_adv,
        "ptype": ptype,
    }
    return pd.DataFrame(rows), diag


def _plot_input_benchmarking(diag: Dict[str, Any]) -> None:
    ptype = diag["ptype"]
    agents = diag["agents"]
    adv_plus = diag["advantage_plus"]
    distances = diag["distances"]

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.5))

    # Patch type shares
    ax = axes[0, 0]

    # Keep all patch categories in a fixed order, including categories with
    # very small or zero shares. This avoids hiding sparse but important cell
    # types such as waste hotspots.
    vals = np.array(sorted(PATCH_LABELS.keys()), dtype=int)
    counts = np.array([(ptype == v).sum() for v in vals], dtype=int)
    labels = [PATCH_LABELS.get(int(v), str(v)) for v in vals]
    shares = counts / counts.sum()
    colors = [PATCH_COLORS.get(int(v), "#999999") for v in vals]

    bars = ax.bar(labels, shares, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Share of grid cells")
    ax.set_title("(a) Patch-type composition")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, max(shares) * 1.18 if shares.size else 1.0)

    # Labels show both count and percentage. For tiny bars, place the label
    # higher and connect it with a thin leader line so that categories such as
    # Waste do not look absent.
    max_share = float(np.max(shares)) if shares.size else 1.0
    y_offset = max_share * 0.025
    tiny_threshold = max_share * 0.035

    for bar, count, share in zip(bars, counts, shares):
        x = bar.get_x() + bar.get_width() / 2
        label = f"{int(count)}\n({share * 100:.2f}%)"

        if share <= tiny_threshold:
            text_y = max_share * 0.075
            ax.annotate(
                label,
                xy=(x, max(share, 0.0005)),
                xytext=(x, text_y),
                ha="center",
                va="bottom",
                fontsize=7.2,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.35"),
            )
        else:
            ax.text(
                x,
                share + y_offset,
                label,
                ha="center",
                va="bottom",
                fontsize=7.2,
            )

    # Initial resources
    ax = axes[0, 1]
    if not agents.empty:
        ax.hist(agents["initial_resource"], bins=30)
    ax.set_xlabel("Initial accumulated resources")
    ax.set_ylabel("Households")
    ax.set_title("(b) Initial resource distribution")

    # Distances
    ax = axes[1, 0]
    data = []
    labels = []
    for lab, arr in distances.items():
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            data.append(arr)
            labels.append(lab)
    if data:
        _boxplot_with_labels(ax, data, labels)
        ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("Nearest grid distance")
    ax.set_title("(c) Distribution of nearest household distances")

    # Advantage map
    ax = axes[1, 1]
    im = ax.imshow(adv_plus.T, origin="lower", aspect="equal")
    ax.set_title("(d) Normalized spatial-advantage layer")
    ax.set_xlabel("Grid x")
    ax.set_ylabel("Grid y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$A_i^+$")

    fig.suptitle("Verification and empirical plausibility checks of model initialization", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig(fig, FIGURES_DIR / "initialization_plausibility_checks")


# =============================================================================
# Output plausibility from an existing replicated run group
# =============================================================================

def _find_latest_run_group() -> Optional[str]:
    manifests = sorted(TABLES_DIR.glob("r*_run_manifest.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifests:
        return None
    return manifests[0].name.replace("_run_manifest.csv", "")


def _manifest_path(run_group: str) -> Path:
    return TABLES_DIR / f"{run_group}_run_manifest.csv"


def _load_main_outputs(run_group: str) -> pd.DataFrame:
    manifest_path = _manifest_path(run_group)
    if not manifest_path.exists():
        return pd.DataFrame()

    manifest = pd.read_csv(manifest_path)
    if "experiment" in manifest.columns:
        sub = manifest[(manifest["experiment"] == "main") & (manifest["status"] == "ok")].copy()
    else:
        sub = manifest[manifest["status"] == "ok"].copy()

    rows = []
    for _, row in sub.iterrows():
        csv_path = Path(str(row.get("master_csv", "")))
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for metric in KPI_CHECKS:
            vals, source_col = _read_metric_values(df, metric)
            if vals is None:
                continue
            w = min(LAST_WINDOW, vals.size)
            scenario = str(row["scenario"])
            rows.append({
                "run_group_source": run_group,
                "scenario": scenario,
                "scenario_label": row.get("scenario_label", SCENARIO_LABELS.get(scenario, scenario)),
                "replicate": row.get("replicate", np.nan),
                "metric": metric,
                "source_column": source_col,
                "last_mean": float(np.nanmean(vals[-w:])),
                "initial": float(vals[0]) if np.isfinite(vals[0]) else np.nan,
                "final": float(vals[-1]) if np.isfinite(vals[-1]) else np.nan,
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
            })
    return pd.DataFrame(rows)


def _output_plausibility_checks(run_group: Optional[str]) -> pd.DataFrame:
    if not run_group or run_group.lower() == "none":
        return pd.DataFrame([{
            "check": "output_plausibility_not_run",
            "value": "",
            "status": "skipped",
            "interpretation": "No run group supplied; only input benchmarking was performed.",
        }])

    outputs = _load_main_outputs(run_group)
    if outputs.empty:
        return pd.DataFrame([{
            "check": "output_plausibility_no_data",
            "value": "",
            "status": "warning",
            "interpretation": f"No readable main-output metrics found for run group {run_group}.",
        }])

    rows = []

    def add(check, value, status, interpretation):
        rows.append({
            "check": check,
            "value": value,
            "status": status,
            "interpretation": interpretation,
        })

    # K1 bounds: grid-cell activation field is normalized and averaged.
    vals = outputs[outputs["metric"] == "MEAN_LOCAL_CE_ACTIVATION"]["last_mean"].to_numpy(dtype=float)
    if vals.size:
        ok = bool(np.nanmin(vals) >= -1e-9 and np.nanmax(vals) <= 1.0 + 1e-6)
        add(
            "k1_mean_local_ce_activation_within_bounds",
            f"{np.nanmin(vals):.6f}--{np.nanmax(vals):.6f}",
            "ok" if ok else "check",
            "K1 is the spatial average of a normalized local CE-activation field and should remain within [0,1].",
        )

    # K2 bounds: flow-based CEI is non-negative and clipped by the implementation.
    vals = outputs[outputs["metric"] == "CEI_SIM"]["last_mean"].to_numpy(dtype=float)
    if vals.size:
        ok = bool(np.nanmin(vals) >= -1e-9 and np.nanmax(vals) <= 10.0 + 1e-6)
        add(
            "k2_flow_based_cei_within_model_bounds",
            f"{np.nanmin(vals):.6f}--{np.nanmax(vals):.6f}",
            "ok" if ok else "check",
            "K2 is the flow-based citywide CE index and should remain non-negative and within the implementation bounds.",
        )

    # K3 should be non-negative.
    vals = outputs[outputs["metric"] == "TRADES_COUNT_M"]["last_mean"].to_numpy(dtype=float)
    if vals.size:
        ok = bool(np.nanmin(vals) >= -1e-9)
        add(
            "k3_ce_flow_counts_non_negative",
            f"{np.nanmin(vals):.6f}--{np.nanmax(vals):.6f}",
            "ok" if ok else "check",
            "Monthly CE-flow counts should not be negative.",
        )

    # General bounds for Gini-type indicators.
    for metric in ["D_adv_exposure", "D_resources"]:
        vals = outputs[outputs["metric"] == metric]["last_mean"].to_numpy(dtype=float)
        if vals.size:
            ok = bool(np.nanmin(vals) >= -1e-9 and np.nanmax(vals) <= 1.0 + 1e-6)
            add(
                f"{metric}_within_gini_bounds",
                f"{np.nanmin(vals):.4f}--{np.nanmax(vals):.4f}",
                "ok" if ok else "check",
                "Gini-type indicators should remain on a non-negative bounded support.",
            )

    vals = outputs[outputs["metric"] == "UNEMPLOYMENT"]["last_mean"].to_numpy(dtype=float)
    if vals.size:
        unemp_max = float(getattr(model, "UNEMP_MAX", 0.35))
        ok = bool(np.nanmin(vals) >= -1e-9 and np.nanmax(vals) <= unemp_max + 1e-6)
        add(
            "unemployment_within_model_bounds",
            f"{np.nanmin(vals):.4f}--{np.nanmax(vals):.4f}",
            "ok" if ok else "check",
            "Unemployment should remain within the bounds imposed by the model.",
        )

    # CE-off logic checks for K1 and K2.
    ceoff_k1 = outputs[(outputs["scenario"] == "ce_off") & (outputs["metric"] == "MEAN_LOCAL_CE_ACTIVATION")]
    if not ceoff_k1.empty:
        v = float(ceoff_k1["last_mean"].mean())
        ok = abs(v) < 1e-3
        add(
            "linear_reference_k1_near_zero",
            f"{v:.6f}",
            "ok" if ok else "check",
            "Mean local CE activation should be near zero when CE mechanisms are disabled.",
        )

    ceoff_k2 = outputs[(outputs["scenario"] == "ce_off") & (outputs["metric"] == "CEI_SIM")]
    if not ceoff_k2.empty:
        v = float(ceoff_k2["last_mean"].mean())
        ok = abs(v) < 1e-3
        add(
            "linear_reference_k2_near_zero",
            f"{v:.6f}",
            "ok" if ok else "check",
            "The flow-based CE index should be near zero when CE mechanisms are disabled.",
        )

    # Scenario-ordering diagnostics, not hard pass/fail.
    mean = outputs.groupby(["scenario", "metric"], as_index=False)["last_mean"].mean()

    def m(scenario, metric):
        arr = mean[(mean["scenario"] == scenario) & (mean["metric"] == metric)]["last_mean"].to_numpy(dtype=float)
        return float(arr[0]) if arr.size else np.nan

    for metric, label in [
        ("MEAN_LOCAL_CE_ACTIVATION", "K1 mean local CE activation"),
        ("CEI_SIM", "K2 flow-based CE index"),
    ]:
        v_light = m("ce_light", metric)
        v_base = m("baseline", metric)
        v_strong = m("ce_strong", metric)
        if np.all(np.isfinite([v_light, v_base, v_strong])):
            add(
                f"{metric}_scenario_ordering_diagnostic",
                f"Light={v_light:.4f}; Baseline={v_base:.4f}; Targeted={v_strong:.4f}",
                "diagnostic",
                f"{label} is expected to increase under stronger CE support, subject to scenario design and stochastic variation.",
            )

    u_off = m("ce_off", "UNEMPLOYMENT")
    u_strong = m("ce_strong", "UNEMPLOYMENT")
    if np.all(np.isfinite([u_off, u_strong])):
        add(
            "targeted_unemployment_vs_linear_reference",
            f"Linear={u_off:.4f}; Targeted={u_strong:.4f}",
            "diagnostic",
            "Under negative beta_CEI, stronger flow-based CE activity should tend to reduce unemployment relative to the linear reference.",
        )

    # Record source-column use for transparency, especially if K1 was read from the old CE_INTENSITY column.
    if "source_column" in outputs.columns:
        source_map = (
            outputs[["metric", "source_column"]]
            .drop_duplicates()
            .sort_values(["metric", "source_column"])
        )
        for _, r in source_map.iterrows():
            add(
                f"source_column_for_{r['metric']}",
                str(r["source_column"]),
                "metadata",
                "Column used when reading canonical KPI metrics from MASTER_metrics.csv files.",
            )

    return pd.DataFrame(rows)


def _plot_output_plausibility(run_group: Optional[str]) -> None:
    if not run_group or run_group.lower() == "none":
        return
    outputs = _load_main_outputs(run_group)
    if outputs.empty:
        return

    available_scenarios = _ordered_scenarios(set(outputs["scenario"]))
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.8))
    axes = axes.ravel()

    for ax, metric in zip(axes, KPI_CHECKS):
        vals = []
        ticklabs = []
        for s in available_scenarios:
            arr = outputs[(outputs["scenario"] == s) & (outputs["metric"] == metric)]["last_mean"].to_numpy(dtype=float)
            if arr.size:
                vals.append(arr)
                ticklabs.append(SCENARIO_LABELS_SHORT.get(s, s))
        if vals:
            _boxplot_with_labels(ax, vals, ticklabs)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_ylabel("Last-window mean")
        ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Output plausibility checks from replicated KPI results", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig(fig, FIGURES_DIR / "output_plausibility_checks")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-group",
        default="latest",
        help="Existing replicated run group to check. Use 'latest', a run group id, or 'none'.",
    )
    args = parser.parse_args()

    if args.run_group == "latest":
        source_run_group = _find_latest_run_group()
        if source_run_group is None:
            source_run_group = "none"
    else:
        source_run_group = args.run_group

    input_summary, diag = _input_benchmarking()
    input_summary.to_csv(INPUT_SUMMARY_CSV, index=False)
    _plot_input_benchmarking(diag)

    output_checks = _output_plausibility_checks(source_run_group)
    output_checks.to_csv(OUTPUT_CHECKS_CSV, index=False)
    _plot_output_plausibility(source_run_group)

    report = {
        "benchmark_run_group": BENCH_GROUP,
        "source_replicated_run_group": source_run_group,
        "input_summary_csv": str(INPUT_SUMMARY_CSV),
        "output_checks_csv": str(OUTPUT_CHECKS_CSV),
        "figures_dir": str(FIGURES_DIR),
        "kpi_alignment": {
            "K1": "MEAN_LOCAL_CE_ACTIVATION; backward-compatible alias CE_INTENSITY",
            "K2": "CEI_SIM; flow-based citywide CE throughput index",
            "K3": "TRADES_COUNT_M; CE-flow count diagnostic",
            "K4": "UNEMPLOYMENT",
            "K5": "D_adv_exposure",
            "K6": "D_resources",
        },
        "interpretation": (
            "These checks provide verification, initialization benchmarking, "
            "and output-plausibility diagnostics. They are not predictive validation."
        ),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_safe), encoding="utf-8")

    print("\n[DONE] Verification and plausibility benchmarking complete.")
    print(f"[DONE] Input summary : {INPUT_SUMMARY_CSV}")
    print(f"[DONE] Output checks : {OUTPUT_CHECKS_CSV}")
    print(f"[DONE] Figures       : {FIGURES_DIR}")
    print(f"[DONE] Report        : {REPORT_JSON}")
    print(f"[INFO] Source run group for output plausibility: {source_run_group}")


if __name__ == "__main__":
    main()
