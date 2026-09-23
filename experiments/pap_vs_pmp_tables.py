"""Turn the cached PAP-vs-PMP runs into the comparison tables.

Shared by `experiments/run_pap_vs_pmp.py` (the tmux driver) and
`experiments/pap_vs_pmp_performance_computational_cost.ipynb`, so the script and the notebook
cannot drift apart. Everything here reads JSON off disk: no GPU, no model, and a partial sweep
is fine.
"""

from __future__ import annotations

import json
import os

import pandas as pd

# The grid defines the experiment, so the tables report exactly the cells it defines -- no more.
# Sweeps run under an older grid leave their records on disk; those cells are filtered out here
# rather than deleted, so widening `pap_vs_pmp_grid` again brings them straight back. `max_time`
# only lands in a cell's params, never in its run_id, so any value gives the same id set.
from pap_vs_pmp_grid import build_grid

GRID_RUN_IDS = {c["run_id"] for c in build_grid(0.0)}
GRID_SIZE = len(GRID_RUN_IDS)

COLUMNS = ["model", "modality", "positional", "strategy", "hyperparam",
           "faithfulness_all", "faithfulness_attention",
           "pct_nodes_retained", "pct_heads_retained", "pct_mlps_retained",
           "n_components", "n_heads_found", "n_mlps_found", "n_paths",
           "seconds", "timed_out", "peak_gpu_mb", "search_peak_mb"]

RENAME = {"faithfulness_all": "faith_all", "faithfulness_attention": "faith_attn",
          "pct_nodes_retained": "%nodes", "pct_heads_retained": "%heads",
          "pct_mlps_retained": "%mlps", "n_components": "comps", "n_heads_found": "heads",
          "n_mlps_found": "mlps", "n_paths": "paths", "seconds": "time_s",
          "timed_out": "TO", "peak_gpu_mb": "peak_mb", "search_peak_mb": "search_mb"}

TABLE_COLUMNS = ["model", "modality", "pos", "strategy", "hyperparam",
                 "faith_all", "faith_attn", "%nodes", "%heads", "%mlps",
                 "comps", "heads", "mlps", "paths", "time_s", "TO", "peak_mb", "search_mb"]

MODALITY_ORDER = {"pap": 0, "pmp": 1, "pmp_batched": 2}

# Size bands for the matched-size comparison: PAP and PMP score on different scales, so a
# shared threshold means nothing and a shared circuit size means everything.
BANDS = [(0, 5), (5, 15), (15, 35), (35, 70), (70, 101)]

ROUNDING = {"faith_all": 3, "faith_attn": 3, "%nodes": 2, "%heads": 2, "%mlps": 2,
            "time_s": 1, "peak_mb": 0, "search_mb": 0}


def load_scores(out_dir: str) -> pd.DataFrame:
    """Every scored run on disk that the current grid still defines, one row each.

    Unreadable files are skipped, not fatal. Off-grid cells (a hyperparameter since dropped from
    `pap_vs_pmp_grid`) are excluded so that every model is reported over the same cell list,
    whatever it was run under.
    """
    root = os.path.join(out_dir, "scores")
    if not os.path.isdir(root):
        return pd.DataFrame()
    rows = []
    for model_slug in sorted(os.listdir(root)):
        d = os.path.join(root, model_slug)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".json") or fname.startswith("_"):
                continue
            try:
                with open(os.path.join(d, fname)) as f:
                    rows.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["run_id"].isin(GRID_RUN_IDS)]
    if df.empty:
        return pd.DataFrame()
    df["hyperparam"] = df["params"].apply(
        lambda p: next((f"{k}={v:g}" for k, v in (p or {}).items()
                        if k in ("min_contribution", "top_n", "max_width")), ""))
    df["peak_gpu_mb"] = df["gpu_after"].apply(lambda g: (g or {}).get("peak_allocated_mb"))
    df["model_weights_mb"] = df["meta"].apply(lambda m: (m or {}).get("model_weights_mb"))
    df["dtype"] = df["meta"].apply(lambda m: (m or {}).get("dtype"))
    df["minutes"] = df["seconds"] / 60.0
    df["positional"] = df["positional"].astype(bool)
    return df


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """The flat deliverable: one row per (model, modality, positional, strategy, hyperparam)."""
    if df.empty:
        return df
    t = df[(df["status"] == "ok") & (df["eval_status"] == "ok")][COLUMNS].copy()
    t = t.rename(columns=RENAME)
    t["pos"] = t.pop("positional").map({True: "pos", False: "nopos"})
    t = t.sort_values(["model", "pos", "modality", "strategy", "%nodes"],
                      key=lambda c: c.map(MODALITY_ORDER) if c.name == "modality" else c)
    return t[TABLE_COLUMNS].reset_index(drop=True)


def modality_summary(t: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, positional, modality): what it achieves and what it costs.

    Each row aggregates the block's 7 hyperparameter cells (`build_grid`: 2 max_widths,
    2 top_ns, 3 thresholds). Read the columns as coming from *different* cells: `best_*` is a
    max, `median_pct_nodes` a median, `mean_time_s` a mean -- no single search has all three.

    `mean_time_s` rather than a median because the cells span three orders of magnitude and
    the median lands on a different cell for each modality, which makes cross-modality
    comparison meaningless. The mean is the sweep's average cell cost; it equals
    `total_time_s / cells` and so only adds information where blocks have unequal cell counts
    (a model with failed cells). Where `timed_out > 0` it is a floor: a truncated search
    records exactly its budget, not what it would have taken.
    """
    if t.empty:
        return t
    g = t.groupby(["model", "pos", "modality"], dropna=False)
    out = g.agg(cells=("strategy", "size"),
                timed_out=("TO", "sum"),
                best_faith_all=("faith_all", "max"),
                best_faith_attn=("faith_attn", "max"),
                median_pct_nodes=("%nodes", "median"),
                mean_time_s=("time_s", "mean"),
                total_time_s=("time_s", "sum"),
                median_peak_mb=("peak_mb", "median")).reset_index()
    return out.sort_values(["model", "pos", "modality"])


def matched_size(t: pd.DataFrame) -> pd.DataFrame:
    """Best faithfulness each modality reaches per circuit-size band, and what it paid.

    Rows with no faithfulness are dropped first. A search that completed but admitted no path
    at all -- a threshold set too high for that model -- yields an empty circuit that is scored
    as null rather than as a number, and a size band containing only such rows would otherwise
    make `idxmax` raise on an all-NA group.
    """
    if t.empty:
        return t
    t = t[t["faith_all"].notna()].copy()
    if t.empty:
        return pd.DataFrame()
    t["size_band"] = pd.cut(t["%nodes"],
                            bins=[b[0] for b in BANDS] + [BANDS[-1][1]], right=False,
                            labels=[f"{lo}-{hi}%" for lo, hi in BANDS])
    idx = t.groupby(["model", "size_band", "modality"], observed=True)["faith_all"].idxmax()
    best = t.loc[idx.dropna()]
    return best.pivot_table(index=["model", "size_band"], columns="modality",
                            values=["faith_all", "time_s", "%nodes"],
                            observed=True, aggfunc="first")


def load_runs(out_dir: str) -> pd.DataFrame:
    """Every *search* on disk, scored or not, one row each.

    Deliberately separate from `load_scores`: a cell that searched fine but was never scored
    (or whose search failed) still cost wall-clock time, and the grid total has to count it.
    """
    root = os.path.join(out_dir, "runs")
    if not os.path.isdir(root):
        return pd.DataFrame()
    rows = []
    for model_slug in sorted(os.listdir(root)):
        d = os.path.join(root, model_slug)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            # `_inflight.json` is the crash marker, not a result.
            if not fname.endswith(".json") or fname.startswith("_"):
                continue
            try:
                with open(os.path.join(d, fname)) as f:
                    rows.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["run_id"].isin(GRID_RUN_IDS)]
    if df.empty:
        return pd.DataFrame()
    df["positional"] = df["positional"].astype(bool)
    df["pos"] = df["positional"].map({True: "pos", False: "nopos"})
    # A crashed cell records seconds=None; it burned time we cannot attribute, so count it as 0
    # and let the `error` column say how much of the grid the total is blind to.
    df["seconds"] = pd.to_numeric(df.get("seconds"), errors="coerce").fillna(0.0)
    df["timed_out"] = df.get("timed_out").fillna(False).astype(bool)
    return df


def grid_totals(out_dir: str, grid_size: int = GRID_SIZE) -> pd.DataFrame:
    """Wall-clock cost of the sweep itself, per model, with an `all models` total row.

    Counts searches only -- the eval phase is scored separately and is not part of discovery
    cost. `budget_h` is the share of `hours` spent in searches that hit their time budget, i.e.
    the part of the total that is a floor rather than a measurement.
    """
    df = load_runs(out_dir)
    if df.empty:
        return pd.DataFrame()

    def block(d: pd.DataFrame, label: str) -> dict:
        return {"model": label,
                "cells_run": len(d), "of_grid": grid_size,
                "ok": int((d["status"] == "ok").sum()),
                "failed": int((d["status"] != "ok").sum()),
                "timed_out": int(d["timed_out"].sum()),
                "hours": d["seconds"].sum() / 3600.0,
                "budget_h": d.loc[d["timed_out"], "seconds"].sum() / 3600.0}

    rows = [block(d, m) for m, d in df.groupby("model", sort=True)]
    total = block(df, "ALL MODELS")
    total["of_grid"] = grid_size * df["model"].nunique()
    rows.append(total)
    return pd.DataFrame(rows)


def all_tables(out_dir: str) -> dict[str, pd.DataFrame]:
    """Every table, from whatever is cached."""
    scores = load_scores(out_dir)
    table = build_table(scores)
    return {"table_full": table,
            "summary_modality": modality_summary(table),
            "summary_matched_size": matched_size(table),
            "grid_totals": grid_totals(out_dir)}


def save_tables(tables: dict[str, pd.DataFrame], out_dir: str) -> list[str]:
    """Write each table as CSV, and as Markdown when `tabulate` is installed."""
    written = []
    for name, frame in tables.items():
        if frame is None or frame.empty:
            continue
        csv_path = os.path.join(out_dir, f"{name}.csv")
        frame.to_csv(csv_path, index=not isinstance(frame.index, pd.RangeIndex))
        written.append(csv_path)
        try:
            rendered = frame.round(3).to_markdown()   # render before opening the file
        except ImportError:
            continue                                  # no tabulate; the CSV is written anyway
        md_path = os.path.join(out_dir, f"{name}.md")
        with open(md_path, "w") as f:
            f.write(rendered)
        written.append(md_path)
    return written
