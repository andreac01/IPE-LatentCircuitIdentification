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
    """Every scored run on disk, one row each. Unreadable files are skipped, not fatal."""
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
    """One row per (model, positional, modality): what it achieves and what it costs."""
    if t.empty:
        return t
    g = t.groupby(["model", "pos", "modality"], dropna=False)
    out = g.agg(cells=("strategy", "size"),
                timed_out=("TO", "sum"),
                best_faith_all=("faith_all", "max"),
                best_faith_attn=("faith_attn", "max"),
                median_pct_nodes=("%nodes", "median"),
                median_time_s=("time_s", "median"),
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


def all_tables(out_dir: str) -> dict[str, pd.DataFrame]:
    """Every table, from whatever is cached."""
    scores = load_scores(out_dir)
    table = build_table(scores)
    return {"table_full": table,
            "summary_modality": modality_summary(table),
            "summary_matched_size": matched_size(table)}


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
