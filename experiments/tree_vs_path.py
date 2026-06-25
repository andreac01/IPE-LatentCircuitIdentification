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

For the greater-than task (Hanna et al., 2023) the ground truth is a full circuit
graph, not a head set, so we parse it into the same Node tree the search produces
and the report compares tree-against-tree (common vs. divergent branches, plus an
ASCII drawing of each).

Usage:
    python experiments/tree_vs_path.py --task ioi --min-contribution 0.05
    python experiments/tree_vs_path.py --task greater-than --strategy topk --max-width 20
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


GREATER_THAN_NOUNS = [
    "war", "reign", "famine", "drought", "feud", "siege", "voyage",
    "journey", "plague", "rivalry", "conflict", "dynasty", "revolt", "campaign",
]


def load_greater_than_batch(model: HookedTransformer, batch_size: int, max_scan: int = 2000):
    """Synthetic greater-than prompts (Hanna et al., 2023). The clean prompt opens
    a two-digit year ``17YY`` (YY in 02..98) and the model should complete a later
    year; the counterfactual sets ``YY=01`` (the standard corruption). As with
    `load_ioi_batch` the batch is bucketed to a single tokenised length.

    Targets are a simple two-digit proxy (a later vs. an earlier year) so the
    existing logit_difference metric runs unchanged; swap in the exact Hanna et
    al. probability-difference metric if you need it.
    """
    candidates = []
    for noun in GREATER_THAN_NOUNS:
        for yy in range(2, 99):
            candidates.append((
                f"The {noun} lasted from the year 17{yy:02d} to the year 17",
                f"{min(yy + 1, 99):02d}",
                f"The {noun} lasted from the year 1701 to the year 17",
                f"{max(yy - 1, 1):02d}",
            ))

    buckets: dict[int, list] = {}
    chosen_len = None
    for prompt, answer, cf_prompt, cf_answer in candidates[:max_scan]:
        n_tok = model.to_tokens(prompt, prepend_bos=True).shape[1]
        buckets.setdefault(n_tok, []).append((prompt, answer, cf_prompt, cf_answer))
        if len(buckets[n_tok]) >= batch_size:
            chosen_len = n_tok
            break
    if chosen_len is None:
        chosen_len = max(buckets, key=lambda k: len(buckets[k]))

    sel = buckets[chosen_len][:batch_size]
    prompts = [s[0] for s in sel]
    answers = [s[1] for s in sel]
    cf_prompts = [s[2] for s in sel]
    cf_answers = [s[3] for s in sel]
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


# ----------------------------------------------------------------------------- greater-than circuit

# Canonical GPT-2 small greater-than circuit (Hanna et al., 2023) as data-flow
# edges (source -> target). Unlike IOI this ground truth is a full graph, so we
# parse it into the same Node tree our tree search produces and compare the two
# tree-against-tree (common vs. divergent branches).
GREATER_THAN_EDGES = [
    # embeddings -> read-in-YY components
    ("embeddings", "a0.h5"), ("embeddings", "a0.h3"), ("embeddings", "a0.h1"), ("embeddings", "m0"),
    # read-in-YY internal connections
    ("m0", "m1"), ("a0.h1", "m1"),
    ("m0", "m2"), ("a0.h1", "m2"),
    ("m0", "m3"), ("m1", "m3"), ("m2", "m3"), ("a0.h1", "m3"),
    # read-in-YY -> create-logit-spike-at-YY (teal heads)
    ("m0", "a5.h1"), ("m1", "a5.h1"), ("m2", "a5.h1"), ("m3", "a5.h1"),
    ("a0.h1", "a5.h1"), ("a0.h3", "a5.h1"), ("a0.h5", "a5.h1"),
    ("m0", "a5.h5"), ("m1", "a5.h5"), ("m2", "a5.h5"), ("m3", "a5.h5"),
    ("a0.h1", "a5.h5"), ("a0.h3", "a5.h5"), ("a0.h5", "a5.h5"),
    ("m0", "a6.h9"), ("m1", "a6.h9"), ("m2", "a6.h9"), ("m3", "a6.h9"),
    ("a0.h1", "a6.h9"), ("a0.h3", "a6.h9"), ("a0.h5", "a6.h9"),
    ("m0", "a7.h10"), ("m1", "a7.h10"), ("m2", "a7.h10"), ("m3", "a7.h10"),
    ("a0.h1", "a7.h10"), ("a0.h3", "a7.h10"), ("a0.h5", "a7.h10"),
    ("m0", "a8.h8"), ("m1", "a8.h8"), ("m2", "a8.h8"), ("m3", "a8.h8"),
    ("a0.h1", "a8.h8"), ("a0.h3", "a8.h8"), ("a0.h5", "a8.h8"),
    ("m0", "a8.h11"), ("m1", "a8.h11"), ("m2", "a8.h11"), ("m3", "a8.h11"),
    ("a0.h1", "a8.h11"), ("a0.h3", "a8.h11"), ("a0.h5", "a8.h11"),
    ("m0", "a9.h1"), ("m1", "a9.h1"), ("m2", "a9.h1"), ("m3", "a9.h1"),
    ("a0.h1", "a9.h1"), ("a0.h3", "a9.h1"), ("a0.h5", "a9.h1"),
    # teal heads -> boost-logits MLPs (orange box)
    ("a5.h1", "m8"), ("a5.h5", "m8"), ("a6.h9", "m8"),
    ("a7.h10", "m8"), ("a8.h8", "m8"), ("a8.h11", "m8"),
    ("a9.h1", "m9"),
    # boost-logits internal connections
    ("m8", "m9"),
    ("m8", "m10"), ("m9", "m10"),
    ("m8", "m11"), ("m9", "m11"), ("m10", "m11"),
    # direct contributions to logits
    ("m8", "logits"), ("m9", "logits"), ("m10", "logits"), ("m11", "logits"),
    ("a5.h1", "logits"), ("a5.h5", "logits"), ("a6.h9", "logits"),
    ("a7.h10", "logits"), ("a8.h8", "logits"), ("a8.h11", "logits"), ("a9.h1", "logits"),
]


def component_name(n: Node) -> str:
    """Canonical name in the GREATER_THAN_EDGES vocabulary, used to compare
    branches across trees (token position / key-value are ignored)."""
    if isinstance(n, FINAL_Node):
        return "logits"
    if isinstance(n, EMBED_Node):
        return "embeddings"
    if isinstance(n, MLP_Node):
        return f"m{n.layer}"
    if isinstance(n, ATTN_Node):
        return f"a{n.layer}.h{n.head}"
    return repr(n)


def _gt_node(name: str, model: HookedTransformer) -> Node:
    """Build one Node for a greater-than component name (embeddings/logits/mL/aL.hH)."""
    if name == "embeddings":
        return EMBED_Node(model, position=None)
    if name == "logits":
        return FINAL_Node(model, layer=model.cfg.n_layers - 1, position=None)
    if name[0] == "m":
        return MLP_Node(model, layer=int(name[1:]), position=None)
    if name[0] == "a":
        layer, head = name[1:].split(".h")
        return ATTN_Node(model, layer=int(layer), head=int(head), position=None)
    raise ValueError(f"Unknown greater-than component: {name}")


def build_ground_truth_tree(edges: list[tuple[str, str]], model: HookedTransformer) -> Node:
    """Parse data-flow edges (source -> target) into our Node tree rooted at logits.
    Our tree points from a successor to its predecessors, so a data-flow edge
    (u -> v) makes `u` a child of `v`. One Node per component, so the result is a
    DAG (shared predecessors) rooted at the FINAL/logits node."""
    nodes: dict[str, Node] = {}

    def get(name: str) -> Node:
        if name not in nodes:
            nodes[name] = _gt_node(name, model)
        return nodes[name]

    for src, dst in edges:
        get(dst).children.add(get(src))
    return get("logits")


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


def node_label(n: Node) -> str:
    """A compact one-token label for a node, used in the ASCII tree drawing."""
    pos = f"@{n.position}" if getattr(n, "position", None) is not None else ""
    if isinstance(n, FINAL_Node):
        return f"FINAL{pos}"
    if isinstance(n, EMBED_Node):
        return f"EMB{pos}"
    if isinstance(n, MLP_Node):
        return f"MLP{n.layer}{pos}"
    if isinstance(n, ATTN_Node):
        head = f"H{n.head}" if n.head is not None else "H*"
        kv = f"kv{n.keyvalue_position}" if n.keyvalue_position is not None else ""
        return f"A{n.layer}{head}{pos}{kv}"
    return repr(n)


def _layout_subtree(node: Node, gap: int = 3) -> tuple[list[str], int, int]:
    """Lay out the tree rooted at `node` top-down as a block of text rows.

    Returns (rows, width, anchor): `rows` are equal-length lines with the node's
    label on row 0 and its whole subtree below; `width` is their common length;
    `anchor` is the column of this node's label centre. Each tree depth lands on
    its own row (root at top, leaves at bottom) and a dashed connector row joins
    every parent to the columns of its children, so the layers line up globally.
    """
    label = node_label(node)
    children = sorted(node.children)
    if not children:
        return [label], len(label), len(label) // 2

    blocks = [_layout_subtree(c, gap) for c in children]
    height = max(len(rows) for rows, _, _ in blocks)
    blocks = [(rows + [" " * w] * (height - len(rows)), w, a) for rows, w, a in blocks]

    # Place child blocks side by side, separated by `gap` spaces.
    merged = [""] * height
    child_anchors = []
    cursor = 0
    for i, (rows, w, a) in enumerate(blocks):
        if i > 0:
            merged = [m + " " * gap for m in merged]
            cursor += gap
        child_anchors.append(cursor + a)
        merged = [m + rows[r] for r, m in enumerate(merged)]
        cursor += w
    width = cursor

    # This node sits centred above the span of its children.
    anchor = (child_anchors[0] + child_anchors[-1]) // 2
    sep = [" "] * width
    for c in range(child_anchors[0], child_anchors[-1] + 1):
        sep[c] = "-"
    for ca in child_anchors:
        sep[ca] = "+"
    sep[anchor] = "+"
    sep_row = "".join(sep)

    # Grow the block if the label would overflow past its children's span.
    label_start = anchor - len(label) // 2
    if label_start < 0:
        pad = -label_start
        merged = [" " * pad + m for m in merged]
        sep_row = " " * pad + sep_row
        anchor += pad
        label_start = 0
        width += pad
    if label_start + len(label) > width:
        pad = label_start + len(label) - width
        merged = [m + " " * pad for m in merged]
        sep_row = sep_row + " " * pad
        width += pad
    label_row = " " * label_start + label + " " * (width - label_start - len(label))

    return [label_row, sep_row, *merged], width, anchor


def render_ascii_tree(root: Node) -> str:
    """Render the discovered tree as an ASCII art drawing, root at the top and
    one model component per node down to the embedding leaves."""
    rows, _, _ = _layout_subtree(root)
    legend = ("Legend: FINAL=output  EMB=embedding  MLP<layer>  A<layer>H<head> "
              "(H*=all heads, @<pos>=position, kv<pos>=key/value position)")
    return "\n".join([legend, ""] + rows)


# ----------------------------------------------------------------------------- position grid

def grid_node_label(n: Node) -> str:
    """Short label for the position grid. Position is encoded by the column and
    key/value reads by the edges, so neither is repeated in the label."""
    if isinstance(n, FINAL_Node):
        return "FINAL"
    if isinstance(n, EMBED_Node):
        return "EMB"
    if isinstance(n, MLP_Node):
        return f"MLP{n.layer}"
    if isinstance(n, ATTN_Node):
        return f"A{n.layer}H{n.head}" if n.head is not None else f"A{n.layer}H*"
    return repr(n)


def _grid_band(n: Node) -> tuple[int, str]:
    """(rank, row-label) placing each node on the layer (y) axis. Higher rank is
    nearer the top: FINAL above everything, then for each layer MLP above ATTN
    (their compute order), embeddings at the bottom."""
    if isinstance(n, FINAL_Node):
        return 1 << 30, "FINAL"
    if isinstance(n, EMBED_Node):
        return -1, "EMB"
    if isinstance(n, MLP_Node):
        return 2 * n.layer + 1, f"L{n.layer} mlp"
    if isinstance(n, ATTN_Node):
        return 2 * n.layer, f"L{n.layer} attn"
    return 0, "?"


def _grid_key(n: Node) -> tuple:
    """Identity of a node *as placed on the grid*: the component and its token
    position. Key/value-position and patch flags are deliberately excluded, so a
    head that the tree repeats across branches (reading different positions, etc.)
    collapses to a single grid node, with its reads expressed by the edges."""
    if isinstance(n, FINAL_Node):
        return ("FINAL", n.position)
    if isinstance(n, EMBED_Node):
        return ("EMB", n.position)
    if isinstance(n, MLP_Node):
        return ("MLP", n.layer, n.position)
    if isinstance(n, ATTN_Node):
        return ("ATTN", n.layer, n.head, n.position)
    return ("?", id(n))


def render_position_grid(root: Node, tokens: list[str] | None = None) -> str:
    """Render the search result as a DAG on a grid with layer on the y-axis and
    token position on the x-axis, routing orthogonal ASCII edges between cells.

    The tree is first collapsed into a DAG: every tree-node that shares a grid
    identity (component + position; see `_grid_key`) becomes one node, and the
    parent->child edges are deduplicated, counting how many tree edges merged
    into each one. This removes the per-branch duplication that otherwise stacks
    the same component into one cell over and over.

    Columns are token positions (with the prompt token as a header); rows are
    layers (FINAL at the top, embeddings at the bottom). Distinct components that
    still share a cell (e.g. two heads at the same layer/position) are spread onto
    extra sub-rows. Edges leave a parent, run along the routing line just beneath
    it, then drop down the child's column - so attention key/value reads show up
    as a horizontal jog to an earlier position. Edges that merged several tree
    edges are listed below the grid with their count."""
    from collections import defaultdict, Counter

    # --- collapse the tree to a DAG: one representative node per grid identity,
    #     edges deduplicated with a multiplicity (how many tree edges merged).
    reps: dict[tuple, Node] = {}
    edges: Counter = Counter()

    def walk(node: Node) -> None:
        k = _grid_key(node)
        reps.setdefault(k, node)
        for child in node.children:
            ck = _grid_key(child)
            if ck != k:
                edges[(k, ck)] += 1
            walk(child)

    walk(root)
    keys = list(reps)

    # --- columns: one per token position, plus a trailing '*' if any node is
    #     position-agnostic (position is None).
    positions = sorted({reps[k].position for k in keys if reps[k].position is not None})
    has_none = any(reps[k].position is None for k in keys)
    cols = positions + ([None] if has_none else [])
    col_idx = {p: i for i, p in enumerate(cols)}

    def key_col(k: tuple) -> int:
        p = reps[k].position
        return col_idx[p] if p is not None else col_idx[None]

    # --- rows: group by layer band, then spread cell-sharing nodes onto sub-rows.
    bands: dict[int, list[tuple]] = defaultdict(list)
    labels: dict[int, str] = {}
    for k in keys:
        rank, lab = _grid_band(reps[k])
        bands[rank].append(k)
        labels[rank] = lab

    rows: list[tuple[str, dict[int, tuple]]] = []
    for rank in sorted(bands, reverse=True):
        subrows: list[dict[int, tuple]] = []
        for k in sorted(bands[rank], key=key_col):
            ci = key_col(k)
            for sr in subrows:
                if ci not in sr:
                    sr[ci] = k
                    break
            else:
                subrows.append({ci: k})
        for i, sr in enumerate(subrows):
            rows.append((labels[rank] if i == 0 else "", sr))

    node_rc: dict[tuple, tuple[int, int]] = {}
    for ri, (_, sr) in enumerate(rows):
        for ci, k in sr.items():
            node_rc[k] = (ri, ci)

    # --- geometry.
    col_w = max([len(grid_node_label(reps[k])) for k in keys] + [3])
    col_gap = 2
    gutter = max([len(lab) for lab, _ in rows] + [len("pos")])
    left = gutter + 2
    header_h = 3  # position index, token, rule
    row_h = 2     # label row + one routing row beneath it

    def col_x(ci: int) -> int:
        return left + ci * (col_w + col_gap) + col_w // 2

    def row_y(ri: int) -> int:
        return header_h + ri * row_h

    width = left + len(cols) * (col_w + col_gap)
    height = header_h + len(rows) * row_h
    grid = [[" "] * width for _ in range(height)]
    is_label = [[False] * width for _ in range(height)]

    def put_label(y: int, x: int, text: str) -> None:
        for i, c in enumerate(text):
            if 0 <= y < height and 0 <= x + i < width:
                grid[y][x + i] = c
                is_label[y][x + i] = True

    def line(y: int, x: int, ch: str) -> None:
        if not (0 <= y < height and 0 <= x < width) or is_label[y][x]:
            return
        cur = grid[y][x]
        if ch == "|":
            grid[y][x] = "+" if cur in "-+" else "|"
        elif ch == "-":
            grid[y][x] = "+" if cur in "|+" else "-"
        else:
            grid[y][x] = ch

    def draw_v(x: int, y0: int, y1: int) -> None:
        for y in range(min(y0, y1), max(y0, y1) + 1):
            line(y, x, "|")

    def draw_h(y: int, x0: int, x1: int) -> None:
        for x in range(min(x0, x1), max(x0, x1) + 1):
            line(y, x, "-")

    # --- header: position numbers, tokens, and a rule.
    put_label(0, 0, "pos")
    put_label(1, 0, "tok")
    for ci, p in enumerate(cols):
        cx = col_x(ci)
        head = "*" if p is None else str(p)
        tok = "any" if p is None else (tokens[p].strip() if tokens and p < len(tokens) else "")
        tok = tok.replace("\n", " ")[:col_w]
        put_label(0, cx - len(head) // 2, head)
        put_label(1, cx - len(tok) // 2, tok)
    for x in range(left, width):
        grid[2][x] = "-"

    # --- nodes.
    for _, sr in rows:
        for ci, k in sr.items():
            text = grid_node_label(reps[k])
            put_label(row_y(node_rc[k][0]), col_x(ci) - len(text) // 2, text)
    for ri, (lab, _) in enumerate(rows):
        if lab:
            put_label(row_y(ri), 0, lab)

    # --- edges: parent (upper) -> child (lower), routed orthogonally.
    for (pk, ck) in edges:
        if pk not in node_rc or ck not in node_rc:
            continue
        pr, pc = node_rc[pk]
        cr, cc = node_rc[ck]
        py, cy = row_y(pr), row_y(cr)
        px, cx = col_x(pc), col_x(cc)
        jy = py + 1  # routing line just beneath the parent
        draw_h(jy, px, cx)
        draw_v(cx, jy, cy - 1)

    body = "\n".join("".join(row).rstrip() for row in grid)
    legend = ("Legend: y=layer (FINAL top, EMB bottom), x=token position.  DAG: each "
              "component appears once (tree branches merged).  FINAL=output  EMB=embedding  "
              "MLP<layer>  A<layer>H<head> (H*=all heads).  A horizontal jog in an edge is an "
              "attention key/value read at an earlier position.")
    out = [legend, "", body]

    # --- multiplicity table for edges that merged more than one tree edge.
    def edge_name(k: tuple) -> str:
        p = reps[k].position
        return grid_node_label(reps[k]) + (f"@{p}" if p is not None else "")

    merged = sorted(((c, pk, ck) for (pk, ck), c in edges.items() if c > 1), reverse=True)
    if merged:
        out += ["", f"Edges merged from multiple branches ({len(merged)}); 'xN' = tree edges merged:"]
        for c, pk, ck in merged[:40]:
            out.append(f"    {edge_name(pk)} -> {edge_name(ck)}  x{c}")
        if len(merged) > 40:
            out.append(f"    ... ({len(merged) - 40} more)")
    return "\n".join(out)


def render_circuit(root: Node, tokens: list[str] | None = None) -> str:
    """Position grid when the search is position-specific, else the depth tree."""
    if any(n.position is not None for n in tree_nodes(root)):
        return render_position_grid(root, tokens)
    return ("(non-positional search: no token positions to place on the x-axis; "
            "run with --positional for the position grid)\n\n" + render_ascii_tree(root))


def write_ioi_circuit_report(filepath: str, pmp_nodes: set, tmp_nodes: set, meta: dict,
                             tmp_root: Node = None, tokens: list[str] | None = None) -> dict:
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

    # ASCII art of the tree found by TreeMessagePatching (root -> leaves).
    if tmp_root is not None:
        lines += ["", "=" * 60, "TreeMessagePatching circuit (ASCII art)", "=" * 60, ""]
        lines.append(render_circuit(tmp_root, tokens))

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    return tot


def write_greater_than_report(filepath: str, gt_root: Node, tmp_root: Node, meta: dict,
                              tokens: list[str] | None = None, max_list: int = 80) -> dict:
    """Compare the ground-truth greater-than tree against the TreeMessagePatching
    result branch-by-branch (a branch = a root->leaf path, by component identity),
    list the common and divergent branches, and draw both as ASCII art."""
    gt = {tuple(component_name(n) for n in path) for path in tree_paths(gt_root)}

    # Split tree-search branches: complete ones reach an embedding leaf, incomplete
    # ones were pruned before getting there. Only complete branches are compared
    # against the ground truth (which always reaches embeddings).
    complete, incomplete = set(), set()
    for path in tree_paths(tmp_root):
        key = tuple(component_name(n) for n in path)
        (complete if isinstance(path[0], EMBED_Node) else incomplete).add(key)
    common, gt_only, tr_only = gt & complete, gt - complete, complete - gt

    def block(title: str, bset: set) -> list[str]:
        out = [f"--- {title} ({len(bset)}) ---"]
        out += [f"    {' -> '.join(b)}" for b in sorted(bset)[:max_list]]
        if len(bset) > max_list:
            out.append(f"    ... ({len(bset) - max_list} more)")
        return out

    lines = ["Greater-than tree comparison report", "=" * 60]
    lines += [f"{k}: {v}" for k, v in meta.items()]
    lines += ["",
              "Branches (root->leaf, embeddings -> ... -> logits, by component identity):",
              f"  ground-truth {len(gt)}   tree-search complete {len(complete)} "
              f"(incomplete {len(incomplete)})   common {len(common)}   "
              f"ground-truth-only {len(gt_only)}   tree-only {len(tr_only)}",
              "(only complete tree branches that reach embeddings are compared)", ""]
    lines += block("Common branches", common) + [""]
    lines += block("Ground-truth-only branches (missed by tree search)", gt_only) + [""]
    lines += block("Tree-only branches (found, not in ground truth)", tr_only) + [""]
    lines += block("Incomplete tree branches (pruned before reaching embeddings)", incomplete)
    lines += ["", "=" * 60, "Ground-truth circuit (ASCII art)", "=" * 60, "",
              render_circuit(gt_root, tokens)]
    lines += ["", "=" * 60, "TreeMessagePatching circuit (ASCII art)", "=" * 60, "",
              render_circuit(tmp_root, tokens)]

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    return dict(gt=len(gt), tree=len(complete), incomplete=len(incomplete),
                common=len(common), gt_only=len(gt_only), tree_only=len(tr_only))


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="ioi", choices=["ioi", "greater-than"],
                        help="ioi: recover the known IOI heads (set comparison). "
                             "greater-than: compare our tree against the Hanna et al. ground-truth tree.")
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--metric", default="logit_difference",
                        choices=["logit_difference", "indirect_effect", "target_logit_percentage",
                                 "target_probability_percentage", "kl_divergence"])
    parser.add_argument("--batch-size", type=int, default=20)
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
    parser.add_argument("--circuit-report", default=None,
                        help="File to write the report to (default: <task>_report.txt).")
    args = parser.parse_args()
    report_path = args.circuit_report or f"{args.task.replace('-', '_')}_report.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(
        args.model, device=device, torch_dtype=torch.float32, center_unembed=True,
    )
    model.eval()

    if args.task == "ioi":
        prompts, answers, cf_prompts, cf_answers, seq_len = load_ioi_batch(
            model, args.batch_size, args.target_length, args.max_scan,
        )
    else:
        prompts, answers, cf_prompts, cf_answers, seq_len = load_greater_than_batch(
            model, args.batch_size, args.max_scan,
        )
    budget = (f"min_contribution={args.min_contribution}" if args.strategy == "threshold"
              else f"max_width={args.max_width}")
    print(f"model={args.model}  task={args.task}  metric={args.metric}  batch={len(prompts)}  "
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

    # --- ground-truth comparison ---------------------------------------------
    meta = {
        "task": args.task, "model": args.model, "metric": args.metric, "batch": len(prompts),
        "seq_len": seq_len, "positional": args.positional,
        "strategy": args.strategy,
        "min_contribution": args.min_contribution if args.strategy == "threshold" else "n/a",
        "max_width": args.max_width if args.strategy == "topk" else "n/a",
        "path_complete_paths": len(pmp_paths), "tree_branches": len(tmp_branches),
        "path_unique_nodes": len(pmp_nodes), "tree_unique_nodes": len(tmp_nodes),
    }
    grid_tokens = model.to_str_tokens(prompts[0])
    if args.task == "ioi":
        tot = write_ioi_circuit_report(report_path, pmp_nodes, tmp_nodes, meta,
                                       tmp_root=tmp_root, tokens=grid_tokens)
        print("\n=== Known IOI circuit recovery ===")
        print(f"  heads recovered: path {tot['path']}/{tot['known']}, tree {tot['tree']}/{tot['known']}, "
              f"both {tot['both']}/{tot['known']}")
        print(f"  MLP layers found: path {tot['path_mlp']}, tree {tot['tree_mlp']}, both {tot['both_mlp']}")
        print(f"  path complete paths: {len(pmp_paths)}   tree branches: {len(tmp_branches)}")
    else:
        gt_root = build_ground_truth_tree(GREATER_THAN_EDGES, model)
        cmp = write_greater_than_report(report_path, gt_root, tmp_root, meta, tokens=grid_tokens)
        print("\n=== Greater-than tree comparison ===")
        print(f"  branches: ground-truth {cmp['gt']}, tree-search complete {cmp['tree']} "
              f"(incomplete {cmp['incomplete']}), common {cmp['common']}")
        print(f"  ground-truth-only {cmp['gt_only']}, tree-only {cmp['tree_only']}")
    print(f"  full report written to: {report_path}")


if __name__ == "__main__":
    main()
