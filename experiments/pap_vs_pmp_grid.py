"""The PAP-vs-PMP sweep definition: which models, which cells, which hyperparameters.

Deliberately free of torch and of `ipe`, so the driver's `status` / `tables` commands and the
notebook's analysis section can import it on a machine with no GPU and no model. The worker
re-exports everything here, so `from pap_vs_pmp_worker import build_grid` keeps working.
"""

from __future__ import annotations

# Chosen to span coarse -> fine at each strategy while keeping the cell count workable.
# `min_contribution` is on the scale of the `logit_difference` metric under counterfactual
# denoising, so these values are NOT comparable across models: read the final table at matched
# circuit size, not at matched hyperparameter.
THRESHOLDS = [0.5, 0.1, 0.02]
# `top_n=1000` was removed after it OOM'd the host (not the GPU) on positional searches.
# BestFirstSearch keeps an *unbounded* heap frontier -- unlike LimitedLevelWidth, which caps it
# with `heapq.nlargest(max_width, ...)` -- so positionally, where each expansion enumerates
# n_components x seq_len candidates, the frontier of live Node objects outgrows system RAM
# before 1000 complete paths are found. `top_n=100` terminates long before that.
# Cached `*_n1000` records from earlier sweeps stay on disk but are excluded from the tables,
# which report whatever `build_grid` currently defines. Restore the value here to bring them back.
TOP_NS = [100]
MAX_WIDTHS = [100, 1000]

# PMP can batch the expansion (evaluate an attention block whole, then only split the winners
# into heads / positions); PAP has no equivalent, so it appears once. "pmp" is the strict
# algorithmic-parity comparison against PAP, "pmp_batched" is PMP as you would actually run it.
MODALITIES = ("pap", "pmp", "pmp_batched")

# The sweep. Cheapest model first, so a complete GPT-2 table exists before Qwen starts and
# Llama's long tail is last -- an interrupted sweep is then still useful.
MODELS = [
    # (name, worker flags)
    ("gpt2-small",                 ["--dtype", "float32",  "--eval-minibatch", "16"]),
    ("Qwen/Qwen2.5-0.5B",          ["--dtype", "float32",  "--eval-minibatch", "16"]),
    # Added after Meta-Llama-3-8B could not fit a positional search on a 24GB card: at 5.7GB of
    # float32 weights this one can, and float32 keeps it comparable with the two models above.
    ("meta-llama/Llama-3.2-1B",    ["--dtype", "float32",  "--eval-minibatch", "16"]),
    # 8B x 4 bytes does not fit a 24GB card, so bf16; smaller eval minibatch because the
    # knockout needs use_attn_result, i.e. [batch, pos, n_heads, d_model] per layer.
    ("meta-llama/Meta-Llama-3-8B", ["--dtype", "bfloat16", "--eval-minibatch", "4"]),
]


def batching_params(modality: str, positional: bool) -> dict:
    """The batching flags for one modality. Only PMP accepts them."""
    if modality != "pmp_batched":
        return {}
    params = {"batch_heads": True}
    if positional:
        # batch_positions needs a positioned root; it is meaningless non-positionally.
        params["batch_positions"] = True
    return params


def build_grid(max_time: float) -> list[dict]:
    """Every cell of the grid, ordered cheapest-first so partial sweeps are still useful."""
    grid = []
    for positional in (False, True):
        for modality in MODALITIES:
            algorithm = "PathAttributionPatching" if modality == "pap" else "PathMessagePatching"
            batching = batching_params(modality, positional)
            cells = []
            for width in MAX_WIDTHS:
                cells.append(("LimitedLevelWidth", {"max_width": width}, f"w{width}"))
            for top_n in TOP_NS:
                cells.append(("BestFirstSearch", {"top_n": top_n}, f"n{top_n}"))
            for threshold in THRESHOLDS:
                cells.append(("Threshold", {"min_contribution": threshold}, f"t{threshold:g}"))
            for strategy, params, tag in cells:
                grid.append({
                    "modality": modality,
                    "algorithm": algorithm,
                    "positional": positional,
                    "strategy": strategy,
                    "params": {**params, **batching, "max_time": max_time,
                               "include_negative": True},
                    "run_id": f"{modality}_{'pos' if positional else 'nopos'}_{strategy}_{tag}",
                })
    return grid
