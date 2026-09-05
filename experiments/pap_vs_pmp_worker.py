"""Run the PAP-vs-PMP grid for **one** model, and score the circuits it finds.

Driven by `experiments/pap_vs_pmp_performance_computational_cost.ipynb`, which calls this
once per model as a subprocess. It is a script rather than notebook cells for three reasons:

* an 8B model is loaded **once** for the whole grid instead of once per run;
* a run that OOMs or is killed takes the subprocess down, not the notebook kernel;
* every run is cached to its own JSON file, so re-invoking resumes where it stopped.

Two phases, both resumable and both cached separately:

``search``  every (modality x positional x strategy x hyperparameter) cell of the grid, each
            under a wall-clock budget (``--max-time``), writing the circuit it found plus its
            time and GPU-memory cost.
``eval``    the knockout faithfulness of each cached circuit, on a disjoint slice of IOI.

Usage::

    PYTHONPATH=src python experiments/pap_vs_pmp_worker.py --model gpt2-small
    PYTHONPATH=src python experiments/pap_vs_pmp_worker.py --model Qwen/Qwen2.5-0.5B
    PYTHONPATH=src python experiments/pap_vs_pmp_worker.py --model meta-llama/Meta-Llama-3-8B \
        --dtype bfloat16 --eval-minibatch 4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (os.path.join(REPO_ROOT, "src"), REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # `ipe` is used from src/, not pip-installed

from knockout import (  # noqa: E402
    EvalSet, Faithfulness, Knockout, ModelGraph, circuit_from_paths,
)

DEFAULT_OUT = os.path.join(HERE, "pap_vs_pmp_performance_computational_cost")


# ----------------------------------------------------------------------------- the grid

# The sweep definition lives in a torch-free module so the driver's `status`/`tables` commands
# and the notebook's analysis can import it without a GPU. Re-exported here for back-compat.
from pap_vs_pmp_grid import (  # noqa: E402,F401
    MAX_WIDTHS, MODALITIES, MODELS, THRESHOLDS, TOP_NS, batching_params, build_grid,
)


# ----------------------------------------------------------------------------- config


@dataclass
class WorkerConfig:
    model_name: str = "gpt2-small"
    dtype: str = "float32"
    out_dir: str = DEFAULT_OUT
    max_time: float = 3600.0        # wall-clock budget per search, seconds
    search_batch: int = 5           # prompts the search itself scores on
    target_length: int | None = None  # None -> modal tokenised length for this tokenizer
    max_scan: int = 4000
    search_metric: str = "logit_difference"
    eval_size: int = 64             # IOI prompts the faithfulness knockout averages over
    eval_minibatch: int = 16
    seed: int = 0
    phases: tuple = ("search", "eval")
    only: str | None = None         # substring filter on run_id, for smoke tests
    force: bool = False
    retry_errors: bool = False      # redo cells whose cached record is an error

    @property
    def model_slug(self) -> str:
        return self.model_name.replace("/", "_")

    @property
    def run_dir(self) -> str:
        return os.path.join(self.out_dir, "runs", self.model_slug)

    @property
    def eval_dir(self) -> str:
        return os.path.join(self.out_dir, "scores", self.model_slug)


# ----------------------------------------------------------------------------- model / data


def load_model(cfg: WorkerConfig):
    from transformer_lens import HookedTransformer

    dtype = getattr(torch, cfg.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {cfg.model_name} ({cfg.dtype}) on {device} ...", flush=True)
    model = HookedTransformer.from_pretrained(
        cfg.model_name, device=device, torch_dtype=dtype, center_unembed=True,
    )
    model.eval()
    # Deliberately NOT calling set_ungroup_grouped_query_attention: `ipe.nodes` does its own
    # GQA handling off `cfg.n_key_value_heads` (it de-repeats W_K/W_V itself), and the Qwen runs
    # in experiments/MIB/run_search.py -- the ones that produced results/PMP/*/qwen-ioi -- load
    # the model exactly like this. Ungrouping here would double-apply the correction.
    # `use_attn_result` is switched on later, for the eval phase only: it makes run_with_cache
    # retain [batch, pos, n_heads, d_model] per layer, which the search does not need and an 8B
    # model cannot spare.
    kv = getattr(model.cfg, "n_key_value_heads", None)
    print(f"  {model.cfg.n_layers} layers x {model.cfg.n_heads} heads"
          + (f" (GQA, {kv} kv heads)" if kv else ""), flush=True)
    return model, device


def _single_token(model, text: str) -> bool:
    return len(model.to_str_tokens(text, prepend_bos=False)) == 1


def load_ioi(model, cfg: WorkerConfig):
    """A search batch and a disjoint evaluation set, both bucketed to one tokenised length.

    The length is discovered per tokenizer rather than hard-coded: 15 is right for GPT-2 and
    Qwen but not for Llama-3. Samples must have single-token IO/S (the logit difference indexes
    them directly) and an ABC counterfactual of the same length (it is mean-ablated positionwise).
    """
    from datasets import load_dataset

    ds = load_dataset("mib-bench/ioi", split="train")
    buckets: dict[int, list] = {}
    for i, sample in enumerate(ds):
        if i >= cfg.max_scan:
            break
        n_tok = model.to_tokens(sample["prompt"], prepend_bos=True).shape[1]
        if cfg.target_length is not None and n_tok != cfg.target_length:
            continue
        io, subj = sample["metadata"]["indirect_object"], sample["metadata"]["subject"]
        if not (_single_token(model, f" {io}") and _single_token(model, f" {subj}")):
            continue
        if model.to_tokens(sample["abc_counterfactual"]["prompt"],
                           prepend_bos=True).shape[1] != n_tok:
            continue
        buckets.setdefault(n_tok, []).append(sample)

    need = cfg.search_batch + cfg.eval_size
    usable = {k: v for k, v in buckets.items() if len(v) >= need}
    if not usable:
        best = max(buckets, key=lambda k: len(buckets[k])) if buckets else None
        raise RuntimeError(
            f"no tokenised length has {need} usable IOI samples within --max-scan={cfg.max_scan}"
            + (f" (best: length {best} with {len(buckets[best])})" if best else ""))
    chosen = max(usable, key=lambda k: len(usable[k]))
    samples = usable[chosen]
    search_samples = samples[:cfg.search_batch]
    eval_samples = samples[cfg.search_batch:cfg.search_batch + cfg.eval_size]
    print(f"  IOI at length {chosen}: {len(search_samples)} search + {len(eval_samples)} eval")

    cf_key = "s2_io_flip_counterfactual"
    search = {
        "prompts": [s["prompt"] for s in search_samples],
        "answers": [" " + s["metadata"]["indirect_object"] for s in search_samples],
        "cf_prompts": [s[cf_key]["prompt"] for s in search_samples],
        "cf_answers": [" " + s[cf_key]["choices"][s[cf_key]["answerKey"]] for s in search_samples],
    }
    device = next(model.parameters()).device
    evalset = EvalSet(
        tokens=model.to_tokens([s["prompt"] for s in eval_samples], prepend_bos=True),
        abc_tokens=model.to_tokens([s["abc_counterfactual"]["prompt"] for s in eval_samples],
                                   prepend_bos=True),
        io=torch.tensor([model.to_single_token(" " + s["metadata"]["indirect_object"])
                         for s in eval_samples], device=device),
        s=torch.tensor([model.to_single_token(" " + s["metadata"]["subject"])
                        for s in eval_samples], device=device),
    )
    return search, evalset, chosen


# ----------------------------------------------------------------------------- serialisation


def node_to_dict(n) -> dict:
    """Model-free description of one search node, keeping its branch contribution."""
    from ipe.nodes import ATTN_Node, EMBED_Node, FINAL_Node, MLP_Node

    d = {"contribution": getattr(n, "contribution", None)}
    if isinstance(n, FINAL_Node):
        d["type"] = "final"
    elif isinstance(n, EMBED_Node):
        d["type"] = "embed"
    elif isinstance(n, MLP_Node):
        d.update(type="mlp", layer=n.layer)
    elif isinstance(n, ATTN_Node):
        d.update(type="attn", layer=n.layer, head=n.head,
                 q=bool(n.patch_query), k=bool(n.patch_key), v=bool(n.patch_value))
    else:
        raise TypeError(f"unknown node type: {type(n)}")
    if d["contribution"] is not None:
        d["contribution"] = float(d["contribution"])
    return d


def serialise_paths(paths) -> list[dict]:
    """`[(score, [EMBED, ..., FINAL]), ...]` -> JSON records `circuit_from_paths` can read."""
    return [{"contribution": float(score), "nodes": [node_to_dict(n) for n in path]}
            for score, path in paths]


# ----------------------------------------------------------------------------- search phase


def write_json(path: str, payload: dict) -> None:
    """Write atomically, so a kill mid-write cannot leave a half-file that resume trusts.

    The sweep is long enough to be interrupted routinely; a truncated JSON would otherwise be
    indistinguishable from a completed cell and would poison the table much later.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cached_result(path: str) -> dict | None:
    """The cached record at `path`, or None if it is absent or unreadable.

    An unparseable file counts as absent, so a cell interrupted mid-write is simply redone.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"    (discarding unreadable cache {os.path.basename(path)})", flush=True)
        return None


def is_done(path: str, retry_errors: bool, key: str = "status") -> bool:
    """Whether a cached cell counts as complete for resume purposes."""
    rec = cached_result(path)
    if rec is None:
        return False
    return not (retry_errors and rec.get(key) != "ok")


def inflight_path(run_dir: str) -> str:
    return os.path.join(run_dir, "_inflight.json")


def claim_crashed_cell(run_dir: str) -> str | None:
    """Turn a leftover in-flight marker into a recorded crash.

    A segfault or an OOM-killer SIGKILL takes the process down before `except` can run, so the
    cell leaves no record at all and silently vanishes from both the done and the failed lists.
    The marker is written before each search and removed after it, so one still present at
    startup names the cell that killed the previous process.

    It is recorded as a failure rather than left absent, so that a cell which reliably crashes
    does not restart the same crash on every resume; `--retry-errors` redoes it deliberately.
    """
    marker = inflight_path(run_dir)
    rec = cached_result(marker)
    if rec is None:
        if os.path.exists(marker):
            os.remove(marker)
        return None
    run_id = rec.get("run_id")
    out_file = os.path.join(run_dir, f"{run_id}.json")
    if not os.path.exists(out_file):
        write_json(out_file, {
            **{k: rec.get(k) for k in ("run_id", "modality", "algorithm", "positional",
                                       "strategy", "params", "model", "meta")},
            "status": "crashed",
            "error": "process died without an exception (segfault, OOM-killer, or host reset) "
                     f"after {time.time() - rec.get('started', time.time()):.0f}s",
            "seconds": None, "timed_out": None, "n_paths": None, "paths": [],
        })
        print(f"  recorded crash of {run_id} from the previous run "
              f"(no exception was raised; re-run with --retry-errors to retry it)", flush=True)
    os.remove(marker)
    return run_id


def gpu_stats(device) -> dict:
    if device.type != "cuda":
        return {}
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2 ** 20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2 ** 20,
        "allocated_mb": torch.cuda.memory_allocated(device) / 2 ** 20,
    }


def run_search_phase(cfg: WorkerConfig, model, device, search, grid, meta) -> None:
    from ipe.experiment import ExperimentManager
    from tree_vs_path import clone_root

    os.makedirs(cfg.run_dir, exist_ok=True)
    claim_crashed_cell(cfg.run_dir)
    todo = [c for c in grid
            if cfg.force or not is_done(os.path.join(cfg.run_dir, c["run_id"] + ".json"),
                                        cfg.retry_errors)]
    if cfg.only:
        todo = [c for c in todo if cfg.only in c["run_id"]]
    print(f"\nsearch phase: {len(todo)} of {len(grid)} cells to run "
          f"(budget {cfg.max_time:g}s each, worst case {len(todo) * cfg.max_time / 3600:.1f}h)\n",
          flush=True)

    # One ExperimentManager at a time, reused across every cell that shares its positional
    # setting (the grid is ordered so that happens exactly once). It owns the clean and
    # counterfactual activation caches, which for an 8B model are large enough that keeping
    # the positional and non-positional ones alive together is worth avoiding.
    managers: dict[bool, object] = {}

    def manager_for(positional: bool):
        """The manager for this positional setting, evicting the other one first.

        Building it costs two forward passes plus the metric baseline.
        """
        for stale in [k for k in managers if k != positional]:
            del managers[stale]
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if positional not in managers:
            managers[positional] = ExperimentManager(
                model=model,
                prompts=search["cf_prompts"], targets=search["cf_answers"],
                cf_prompts=search["prompts"], cf_targets=search["answers"],
                algorithm="PathMessagePatching", search_strategy="Threshold",
                algorithm_params={"min_contribution": 1.0},
                metric=cfg.search_metric,
                positional_search=positional,
                patch_type="counterfactual",
                patch_clean_into_cf=True,
            )
        return managers[positional]

    for i, cell in enumerate(todo, 1):
        out_file = os.path.join(cfg.run_dir, cell["run_id"] + ".json")
        print(f"[{i}/{len(todo)}] {cell['run_id']}", flush=True)
        record = {**cell, "model": cfg.model_name, "meta": meta}
        write_json(inflight_path(cfg.run_dir), {**record, "started": time.time()})
        manager = None  # drop the previous cell's reference so eviction can actually free it
        try:
            manager = manager_for(cell["positional"])
            manager.load_algorithm(cell["algorithm"], cell["strategy"], dict(cell["params"]))
            root = clone_root(manager.root)
            manager.algorithm.keywords["root"] = root

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            before = gpu_stats(device)

            t0 = time.time()
            paths = manager.algorithm()
            seconds = time.time() - t0
            after = gpu_stats(device)

            record.update({
                "status": "ok",
                "seconds": seconds,
                "timed_out": bool(getattr(root, "timed_out", False)),
                "n_paths": len(paths),
                "gpu_before": before,
                "gpu_after": after,
                # The search's own footprint, over the model weights and activation caches that
                # were already resident when it started.
                "search_peak_mb": (after.get("peak_allocated_mb", 0)
                                   - before.get("allocated_mb", 0)) or None,
                "paths": serialise_paths(paths),
            })
            print(f"    {len(paths)} paths in {seconds:.0f}s"
                  f"{'  [TIMED OUT]' if record['timed_out'] else ''}"
                  f"  peak {after.get('peak_allocated_mb', 0):.0f} MB", flush=True)
            del paths, root
        except Exception as exc:  # a single cell must not abort the sweep
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc(), "paths": []})
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
        write_json(out_file, record)
        if os.path.exists(inflight_path(cfg.run_dir)):
            os.remove(inflight_path(cfg.run_dir))
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    managers.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- eval phase


def run_eval_phase(cfg: WorkerConfig, model, device, evalset) -> None:
    """Knockout faithfulness of every cached circuit, plus the fraction of nodes retained."""
    os.makedirs(cfg.eval_dir, exist_ok=True)
    runs = sorted(f for f in os.listdir(cfg.run_dir)
                  if f.endswith(".json") and not f.startswith("_"))
    todo = [f for f in runs
            if cfg.force or not is_done(os.path.join(cfg.eval_dir, f), cfg.retry_errors,
                                        key="eval_status")]
    if cfg.only:
        todo = [f for f in todo if cfg.only in f]
    print(f"\neval phase: {len(todo)} of {len(runs)} circuits to score", flush=True)
    if not todo:
        return

    # Per-head outputs, so the knockout can replace one head at a time. Only needed here.
    model.cfg.use_attn_result = True
    graph = ModelGraph.of(model)
    knockout = Knockout(model, evalset, graph, ablation="mean", minibatch=cfg.eval_minibatch)
    scorer = Faithfulness(knockout, graph)
    sanity = scorer.sanity()
    print("  " + "  ".join(f"{k}={v:+.4f}" for k, v in sanity.items()), flush=True)
    if sanity["full_vs_clean_gap"] > 2e-2:
        print(f"  WARNING: the full circuit does not reproduce the unhooked model "
              f"(gap {sanity['full_vs_clean_gap']:.4f}); knockout numbers are suspect.")
    write_json(os.path.join(cfg.eval_dir, "_baseline.json"),
               {**sanity, "n_components": graph.n_components,
                "n_layers": graph.n_layers, "n_heads": graph.n_heads})

    for i, fname in enumerate(todo, 1):
        with open(os.path.join(cfg.run_dir, fname)) as f:
            run = json.load(f)
        row = {k: run[k] for k in
               ("run_id", "modality", "algorithm", "positional", "strategy", "params",
                "model", "status")}
        row.update({k: run.get(k) for k in
                    ("seconds", "timed_out", "n_paths", "search_peak_mb", "gpu_after", "meta")})
        try:
            circuit = circuit_from_paths(run.get("paths", []), name=run["run_id"])
            row.update({
                "n_components": len(circuit.components),
                "n_heads_found": len(circuit.heads),
                "n_mlps_found": len(circuit.mlps),
                "n_edges": len(circuit.edges),
                "pct_nodes_retained": 100.0 * len(circuit.components) / graph.n_components,
                "pct_heads_retained": 100.0 * len(circuit.heads)
                                      / (graph.n_layers * graph.n_heads),
                "pct_mlps_retained": 100.0 * len(circuit.mlps) / graph.n_layers,
            })
            row.update(scorer.score(circuit) if circuit.components else
                       {"F_all": None, "faithfulness_all": None,
                        "F_attention": None, "faithfulness_attention": None})
            row["eval_status"] = "ok"
            print(f"[{i}/{len(todo)}] {run['run_id']}: "
                  f"{row['n_components']} comps ({row['pct_nodes_retained']:.1f}%)  "
                  f"faith_all={row.get('faithfulness_all')}", flush=True)
        except Exception as exc:
            row.update({"eval_status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i}/{len(todo)}] {run['run_id']}: EVAL FAILED: {exc}", flush=True)
        write_json(os.path.join(cfg.eval_dir, fname), row)


# ----------------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2-small")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--max-time", type=float, default=3600.0,
                   help="wall-clock budget per search, seconds (default: 1h)")
    p.add_argument("--search-batch", type=int, default=5)
    p.add_argument("--target-length", type=int, default=None,
                   help="tokenised prompt length; default picks the modal usable length")
    p.add_argument("--eval-size", type=int, default=64)
    p.add_argument("--eval-minibatch", type=int, default=16)
    p.add_argument("--phase", default="both", choices=["search", "eval", "both"])
    p.add_argument("--only", default=None, help="substring filter on run_id (smoke tests)")
    p.add_argument("--force", action="store_true", help="recompute cached runs")
    p.add_argument("--retry-errors", action="store_true",
                   help="redo cells whose cached record is an error (e.g. a transient OOM)")
    a = p.parse_args()

    cfg = WorkerConfig(
        model_name=a.model, dtype=a.dtype, out_dir=a.out_dir, max_time=a.max_time,
        search_batch=a.search_batch, target_length=a.target_length,
        eval_size=a.eval_size, eval_minibatch=a.eval_minibatch,
        phases=("search", "eval") if a.phase == "both" else (a.phase,),
        only=a.only, force=a.force, retry_errors=a.retry_errors,
    )
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.run_dir, exist_ok=True)

    model, device = load_model(cfg)
    weights_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20
    search, evalset, length = load_ioi(model, cfg)
    meta = {"dtype": cfg.dtype, "n_layers": model.cfg.n_layers, "n_heads": model.cfg.n_heads,
            "n_components": model.cfg.n_layers * (model.cfg.n_heads + 1),
            "target_length": length, "search_batch": cfg.search_batch,
            "eval_size": len(evalset), "metric": cfg.search_metric,
            "max_time": cfg.max_time, "model_weights_mb": weights_mb}
    print(f"  weights: {weights_mb:.0f} MB", flush=True)

    grid = build_grid(cfg.max_time)
    if "search" in cfg.phases:
        run_search_phase(cfg, model, device, search, grid, meta)
    if "eval" in cfg.phases:
        run_eval_phase(cfg, model, device, evalset)
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
