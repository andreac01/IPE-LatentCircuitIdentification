"""Compare the path-based search (PathMessagePatching) against the tree-based
search (TreeMessagePatching) on the same model, prompt and metric.

The two methods score differently on purpose:
- PathMessagePatching scores each linear path in isolation.
- TreeMessagePatching scores each candidate by its *marginal* effect on the
  joint ablation of the tree discovered so far.

This script runs both with identical settings and reports, for each, the
components discovered, the runtime, and the overlap between the two.

It also checks the core invariant that evaluate_tree generalizes evaluate_path:
on a single-child chain the two must return the same contribution.

Usage:
    python experiments/tree_vs_path.py --prompt "The capital of France is" --target " Paris" --min-contribution 1.0
"""

import argparse
import time
import torch
from functools import partial
from transformer_lens import HookedTransformer

from ipe.nodes import EMBED_Node, MLP_Node, ATTN_Node, FINAL_Node, Node
from ipe.metrics import target_logit_percentage
from ipe.paths import evaluate_path, evaluate_tree
from ipe.graph_search import PathMessagePatching, TreeMessagePatching


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


def check_invariant(model: HookedTransformer, metric, msg_cache: dict, position: int) -> None:
    """A single-child tree must score identically to the corresponding path."""
    L = model.cfg.n_layers - 1
    print("\n=== Invariant: single-child tree == path ===")
    for label, mid in (
        ("EMBED -> MLP -> FINAL", MLP_Node(model, layer=L, position=position, msg_cache=msg_cache)),
        ("EMBED -> ATTN -> FINAL", ATTN_Node(model, layer=L, head=0, position=position, keyvalue_position=position, msg_cache=msg_cache)),
    ):
        final = FINAL_Node(model, layer=L, position=position, msg_cache=msg_cache, metric=metric)
        emb = EMBED_Node(model, position=position, msg_cache=msg_cache)
        path = [emb, mid, final]
        chain_to_tree(path)
        with torch.no_grad():
            p = evaluate_path(path, metric)
            t = evaluate_tree(final, metric)
        p, t = float(p), float(t)
        ok = abs(p - t) < 1e-4
        print(f"  {label:<24} path={p:+.5f}  tree={t:+.5f}  {'OK' if ok else 'MISMATCH'}")
        assert ok, f"Invariant violated for {label}: {p} vs {t}"


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--target", default=" Paris", help="single-token continuation to track")
    parser.add_argument("--min-contribution", type=float, default=1.0)
    parser.add_argument("--include-negative", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(args.model, device=device, torch_dtype=torch.float32)
    model.eval()

    _, cache = model.run_with_cache(args.prompt, prepend_bos=True)
    msg_cache = dict(cache)
    L = model.cfg.n_layers - 1
    position = msg_cache["blocks.0.hook_resid_post"].shape[1] - 1
    target_tokens = [model.to_single_token(args.target)]
    clean_final_resid = msg_cache[f"blocks.{L}.hook_resid_post"]

    # Bind everything except corrupted_resid, mirroring ExperimentManager.load_metric.
    metric = partial(
        target_logit_percentage,
        clean_final_resid=clean_final_resid,
        model=model,
        target_tokens=target_tokens,
    )

    print(f"model={args.model}  prompt={args.prompt!r}  target={args.target!r}  "
          f"position={position}  min_contribution={args.min_contribution}")

    check_invariant(model, metric, msg_cache, position)

    # --- path search ---------------------------------------------------------
    pmp_root = FINAL_Node(model, layer=L, position=position, msg_cache=msg_cache, metric=metric)
    t0 = time.time()
    pmp_paths = PathMessagePatching(
        model, metric, pmp_root,
        min_contribution=args.min_contribution,
        include_negative=args.include_negative,
    )
    pmp_time = time.time() - t0
    pmp_nodes = {n for _, path in pmp_paths for n in path}

    # --- tree search ---------------------------------------------------------
    tmp_root = FINAL_Node(model, layer=L, position=position, msg_cache=msg_cache, metric=metric)
    t0 = time.time()
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


if __name__ == "__main__":
    main()
