"""Compare the path-based search (PathMessagePatching) against the tree-based
search (TreeMessagePatching) on the IOI task.

Both methods now score each candidate by its own isolated contribution along the
branch it would create (TreeMessagePatching no longer uses a joint-ablation
baseline), so the tree is the same set of paths materialised as a trie sharing
common suffixes toward the root. This script runs both with identical settings
and reports, for each, the components discovered, the runtime, and the overlap.

The IOI setup mirrors experiments/MIB/run_search.py: prompts come from the
`mib-bench/ioi` dataset, with the `s2_io_flip_counterfactual` as the
counterfactual, counterfactual (denoising) patching, and the experiment metric
built by ExperimentManager. Because clean and counterfactual prompts solve the
same task with different names, we follow run_search and feed the counterfactual
prompts as the clean run (and vice-versa).

It also checks the core invariant that evaluate_tree generalizes evaluate_path:
on a single-child chain the two must return the same contribution.

Usage:
    python experiments/tree_vs_path.py --model gpt2-small --metric logit_difference --min-contribution 0.05
"""

import argparse
import time
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

from ipe.nodes import EMBED_Node, MLP_Node, ATTN_Node, FINAL_Node, Node
from ipe.paths import evaluate_path, evaluate_tree
from ipe.experiment import ExperimentManager
from ipe.graph_search import (
    PathMessagePatching,
    PathMessagePatching_LimitedLevelWidth,
    TreeMessagePatching,
    TreeMessagePatching_LimitedLevelWidth,
    setup_tree_debug_log,
)


# ----------------------------------------------------------------------------- helpers

def tree_nodes(node: Node) -> list[Node]:
    """All nodes in the tree rooted at `node` (depth-first, root included)."""
    nodes = [node]
    for child in node.children:
        nodes.extend(tree_nodes(child))
    return nodes


def tree_paths(node: Node) -> list[list[Node]]:
    """All leaf->root paths in the tree, each ordered [leaf, ..., root] to match
    the [EMBED, ..., FINAL] convention of PathMessagePatching."""
    if not node.children:
        return [[node]]
    paths = []
    for child in node.children:
        for sub in tree_paths(child):
            paths.append(sub + [node])
    return paths


def chain_to_tree(path: list[Node]) -> None:
    """Wire a [leaf, ..., root] path as a single-child tree (in place)."""
    path[0].children = set()
    for child, parent in zip(path[:-1], path[1:]):
        parent.children = {child}


def clone_root(root: FINAL_Node) -> FINAL_Node:
    """A fresh FINAL_Node with the same caches/metric/patch config as `root`.

    Each search mutates its root's `children` in place, so path and tree search
    each need their own root built from the identical configuration."""
    return FINAL_Node(
        model=root.model,
        layer=root.layer,
        position=root.position,
        msg_cache=root.msg_cache,
        cf_cache=root.cf_cache,
        metric=root.metric,
        patch_type=root.patch_type,
    )


def load_ioi_batch(model: HookedTransformer, batch_size: int, target_length: int | None, max_scan: int):
    """Pull `batch_size` IOI samples (and their s2-io-flip counterfactuals) that
    share a single tokenised length, mirroring experiments/MIB/run_search.py.

    If `target_length` is None the modal length among the first `max_scan`
    samples is used, so the batch is always equal-length (required for batched
    caching) without hard-coding a model-specific length.
    """
    cf_key = "s2_io_flip_counterfactual"
    dataset = load_dataset("mib-bench/ioi", split="train")

    buckets: dict[int, list[dict]] = {}
    chosen_len = None
    for i, sample in enumerate(dataset):
        if i >= max_scan:
            break
        n_tok = model.to_tokens(sample["prompt"], prepend_bos=True).shape[1]
        if target_length is not None and n_tok != target_length:
            continue
        buckets.setdefault(n_tok, []).append(sample)
        if len(buckets[n_tok]) >= batch_size:
            chosen_len = n_tok
            break

    if chosen_len is None:
        # Scan budget exhausted before any length bucket filled.
        if not buckets:
            raise RuntimeError("No IOI samples found; try increasing --max-scan or check the dataset.")
        chosen_len = max(buckets, key=lambda k: len(buckets[k]))
        if len(buckets[chosen_len]) < batch_size:
            raise RuntimeError(
                f"Only {len(buckets[chosen_len])} samples of length {chosen_len} found within "
                f"--max-scan={max_scan}; lower --batch-size or raise --max-scan."
            )

    samples = buckets[chosen_len][:batch_size]
    prompts, answers, cf_prompts, cf_answers = [], [], [], []
    for s in samples:
        prompts.append(s["prompt"])
        cf_prompts.append(s[cf_key]["prompt"])
        answers.append(" " + s["metadata"]["indirect_object"])
        cf_answers.append(" " + s[cf_key]["choices"][s[cf_key]["answerKey"]])
    return prompts, answers, cf_prompts, cf_answers, chosen_len


def check_invariant(model: HookedTransformer, root: FINAL_Node) -> None:
    """A single-child tree must score identically to the corresponding path.

    Nodes are built from the search root's caches / patch_type / metric so the
    check exercises the same configuration (counterfactual patching included) as
    the run, at a concrete position regardless of positional_search."""
    metric = root.metric
    L = model.cfg.n_layers - 1
    position = root.msg_cache["blocks.0.hook_resid_post"].shape[1] - 1
    kw = dict(msg_cache=root.msg_cache, cf_cache=root.cf_cache, patch_type=root.patch_type)
    print("\n=== Invariant: single-child tree == path ===")
    for label, mid in (
        ("EMBED -> MLP -> FINAL", MLP_Node(model, layer=L, position=position, **kw)),
        ("EMBED -> ATTN -> FINAL", ATTN_Node(model, layer=L, head=0, position=position, keyvalue_position=position, **kw)),
    ):
        final = FINAL_Node(model, layer=L, position=position, metric=metric, **kw)
        emb = EMBED_Node(model, position=position, **kw)
        path = [emb, mid, final]
        chain_to_tree(path)
        with torch.no_grad():
            # Prime each node's own output cache. Under counterfactual patching a
            # mid-chain node's forward(message) reads msg_cache[output_name], which
            # is only populated by forward(None) (the search always scores a node
            # as the deepest node first, so this happens implicitly there).
            for n in path:
                n.forward(message=None)
            p = float(evaluate_path(path, metric))
            t = float(evaluate_tree(final, metric))
        ok = abs(p - t) < 1e-4
        print(f"  {label:<24} path={p:+.5f}  tree={t:+.5f}  {'OK' if ok else 'MISMATCH'}")
        assert ok, f"Invariant violated for {label}: {p} vs {t}"


# ----------------------------------------------------------------------------- IOI circuit

# Canonical GPT-2 small IOI circuit, (layer, head) grouped by functional role
# (Wang et al., 2022, "Interpretability in the Wild"). Used only to report how
# much of the known circuit each search recovers; it does not steer the search.
IOI_CIRCUIT = {
    "Name Mover": [(9, 9), (10, 0), (9, 6)],
    "Negative Name Mover": [(10, 7), (11, 10)],
    "Backup Name Mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "S-Inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "Induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "Duplicate Token": [(0, 1), (0, 10), (3, 0)],
    "Previous Token": [(2, 2), (4, 11)],
}


def attn_heads(nodes) -> set[tuple[int, int]]:
    """The set of (layer, head) attention heads present in `nodes`.

    Heads appear under several patch variants (query / key-value) and, in
    positional mode, several positions; membership in the IOI circuit only cares
    about (layer, head), so we collapse to that."""
    return {(n.layer, n.head) for n in nodes if isinstance(n, ATTN_Node) and n.head is not None}


def mlp_layers(nodes) -> set[int]:
    """The set of MLP layers present in `nodes`. IOI has no canonical MLP set, so
    these are reported descriptively rather than scored against a ground truth."""
    return {n.layer for n in nodes if isinstance(n, MLP_Node)}


def write_ioi_circuit_report(filepath: str, pmp_nodes: set, tmp_nodes: set, meta: dict) -> dict:
    """Write a per-group breakdown of how much of the known IOI circuit each
    search recovered to `filepath`, and return the headline coverage counts."""
    pmp_heads = attn_heads(pmp_nodes)
    tmp_heads = attn_heads(tmp_nodes)
    known = {lh for heads in IOI_CIRCUIT.values() for lh in heads}

    lines = ["IOI known-circuit recovery report", "=" * 60]
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    lines += ["", f"Legend: P = found by PathMessagePatching, T = found by TreeMessagePatching", ""]

    tot = dict(known=0, path=0, tree=0, both=0)
    for group, heads in IOI_CIRCUIT.items():
        g_path = sum((lh in pmp_heads) for lh in heads)
        g_tree = sum((lh in tmp_heads) for lh in heads)
        lines.append(f"--- {group}  (path {g_path}/{len(heads)}, tree {g_tree}/{len(heads)}) ---")
        for (l, h) in heads:
            p = "P" if (l, h) in pmp_heads else "-"
            t = "T" if (l, h) in tmp_heads else "-"
            lines.append(f"    L{l:>2}H{h:<2}   [{p}{t}]")
            tot["known"] += 1
            tot["path"] += (l, h) in pmp_heads
            tot["tree"] += (l, h) in tmp_heads
            tot["both"] += (l, h) in pmp_heads and (l, h) in tmp_heads
        lines.append("")

    lines.append("=" * 60)
    lines.append(f"Known-circuit heads recovered: path {tot['path']}/{tot['known']}, "
                 f"tree {tot['tree']}/{tot['known']}, both {tot['both']}/{tot['known']}")

    # Heads discovered that are NOT part of the documented circuit.
    extra_path = sorted(pmp_heads - known)
    extra_tree = sorted(tmp_heads - known)
    lines += ["", f"Off-circuit heads found by path ({len(extra_path)}): "
              + ", ".join(f"L{l}H{h}" for l, h in extra_path)]
    lines.append(f"Off-circuit heads found by tree ({len(extra_tree)}): "
                 + ", ".join(f"L{l}H{h}" for l, h in extra_tree))

    # MLP components (no canonical IOI MLP set; report what each search found).
    path_mlps = mlp_layers(pmp_nodes)
    tree_mlps = mlp_layers(tmp_nodes)
    tot["path_mlp"] = len(path_mlps)
    tot["tree_mlp"] = len(tree_mlps)
    tot["both_mlp"] = len(path_mlps & tree_mlps)
    lines += ["", "=" * 60,
              f"MLP layers found: path {tot['path_mlp']}, tree {tot['tree_mlp']}, "
              f"both {tot['both_mlp']}"]
    for l in sorted(path_mlps | tree_mlps):
        p = "P" if l in path_mlps else "-"
        t = "T" if l in tree_mlps else "-"
        lines.append(f"    MLP{l:<2}   [{p}{t}]")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    return tot


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--metric", default="logit_difference",
                        choices=["logit_difference", "indirect_effect", "target_logit_percentage",
                                 "target_probability_percentage", "kl_divergence"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-length", type=int, default=None,
                        help="Force a tokenised prompt length; default picks the modal length automatically.")
    parser.add_argument("--max-scan", type=int, default=2000,
                        help="How many dataset samples to scan when forming the batch.")
    parser.add_argument("--strategy", default="threshold", choices=["threshold", "topk"],
                        help="threshold: admit any candidate above --min-contribution; "
                             "topk: keep the top --max-width candidates per depth (beam).")
    parser.add_argument("--min-contribution", type=float, default=0.05,
                        help="Admission threshold (threshold strategy); scale depends on the chosen metric.")
    parser.add_argument("--max-width", type=int, default=20000,
                        help="Candidates retained per depth (topk strategy).")
    parser.add_argument("--positional", action="store_true", default=False,
                        help="Position-specific search (default: non-positional, as in run_search for IOI).")
    parser.add_argument("--include-negative", action="store_true", default=True)
    parser.add_argument("--circuit-report", default="ioi_circuit_report.txt",
                        help="File to write the known-IOI-circuit recovery breakdown to.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(
        args.model, device=device, torch_dtype=torch.float32, center_unembed=True,
    )
    model.eval()

    prompts, answers, cf_prompts, cf_answers, seq_len = load_ioi_batch(
        model, args.batch_size, args.target_length, args.max_scan,
    )
    budget = (f"min_contribution={args.min_contribution}" if args.strategy == "threshold"
              else f"max_width={args.max_width}")
    print(f"model={args.model}  task=ioi  metric={args.metric}  batch={len(prompts)}  "
          f"seq_len={seq_len}  positional={args.positional}  strategy={args.strategy}  {budget}")
    print(f"  clean   e.g. {prompts[0]!r} -> {answers[0]!r}")
    print(f"  cfactual e.g. {cf_prompts[0]!r} -> {cf_answers[0]!r}")

    # IOI with counterfactual denoising: feed the counterfactual prompts as the
    # clean run (and vice-versa), exactly as experiments/MIB/run_search.py does.
    experiment = ExperimentManager(
        model=model,
        prompts=cf_prompts,
        targets=cf_answers,
        cf_prompts=prompts,
        cf_targets=answers,
        algorithm="PathMessagePatching",
        search_strategy="Threshold",
        algorithm_params={"min_contribution": args.min_contribution, "include_negative": args.include_negative},
        metric=args.metric,
        positional_search=args.positional,
        patch_type="counterfactual",
        patch_clean_into_cf=True,
    )
    metric = experiment.metric

    check_invariant(model, experiment.root)

    # --- path search ---------------------------------------------------------
    pmp_root = clone_root(experiment.root)
    t0 = time.time()
    if args.strategy == "topk":
        pmp_paths = PathMessagePatching_LimitedLevelWidth(
            model, metric, pmp_root,
            max_width=args.max_width,
            include_negative=args.include_negative,
        )
    else:
        pmp_paths = PathMessagePatching(
            model, metric, pmp_root,
            min_contribution=args.min_contribution,
            include_negative=args.include_negative,
        )
    pmp_time = time.time() - t0
    pmp_nodes = {n for _, path in pmp_paths for n in path}

    # --- tree search ---------------------------------------------------------
    setup_tree_debug_log("tree_search_debug.log")
    tmp_root = clone_root(experiment.root)
    t0 = time.time()
    if args.strategy == "topk":
        tmp_root = TreeMessagePatching_LimitedLevelWidth(
            model, metric, tmp_root,
            max_width=args.max_width,
            include_negative=args.include_negative,
        )
    else:
        tmp_root = TreeMessagePatching(
            model, metric, tmp_root,
            min_contribution=args.min_contribution,
            include_negative=args.include_negative,
        )
    tmp_time = time.time() - t0
    tmp_all = tree_nodes(tmp_root)
    tmp_nodes = set(tmp_all)
    tmp_branches = tree_paths(tmp_root)

    # --- report --------------------------------------------------------------
    print("\n=== PathMessagePatching ===")
    print(f"  runtime: {pmp_time:.2f}s   complete paths: {len(pmp_paths)}   unique nodes: {len(pmp_nodes)}")
    for score, path in pmp_paths[:10]:
        chain = " <- ".join(repr(n) for n in reversed(path))
        print(f"    {float(score):+.4f}  {chain}")

    print("\n=== TreeMessagePatching ===")
    print(f"  runtime: {tmp_time:.2f}s   tree nodes: {len(tmp_all)}   root->leaf branches: {len(tmp_branches)}")
    print(f"  joint ablation of full tree: {float(evaluate_tree(tmp_root, metric)):+.4f}")
    for branch in tmp_branches[:10]:
        chain = " <- ".join(repr(n) for n in reversed(branch))
        print(f"    {chain}")

    print("\n=== Overlap (by component identity) ===")
    inter = pmp_nodes & tmp_nodes
    print(f"  shared: {len(inter)}   path-only: {len(pmp_nodes - tmp_nodes)}   tree-only: {len(tmp_nodes - pmp_nodes)}")

    # --- known IOI circuit recovery ------------------------------------------
    meta = {
        "model": args.model, "metric": args.metric, "batch": len(prompts),
        "seq_len": seq_len, "positional": args.positional,
        "strategy": args.strategy,
        "min_contribution": args.min_contribution if args.strategy == "threshold" else "n/a",
        "max_width": args.max_width if args.strategy == "topk" else "n/a",
        "path_complete_paths": len(pmp_paths), "tree_branches": len(tmp_branches),
        "path_unique_nodes": len(pmp_nodes), "tree_unique_nodes": len(tmp_nodes),
    }
    tot = write_ioi_circuit_report(args.circuit_report, pmp_nodes, tmp_nodes, meta)
    print("\n=== Known IOI circuit recovery ===")
    print(f"  heads recovered: path {tot['path']}/{tot['known']}, tree {tot['tree']}/{tot['known']}, "
          f"both {tot['both']}/{tot['known']}")
    print(f"  MLP layers found: path {tot['path_mlp']}, tree {tot['tree_mlp']}, both {tot['both_mlp']}")
    print(f"  path complete paths: {len(pmp_paths)}   tree branches: {len(tmp_branches)}")
    print(f"  full breakdown written to: {args.circuit_report}")


if __name__ == "__main__":
    main()
