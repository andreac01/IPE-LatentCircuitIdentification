# Tree search: design history, iterations, and current state

This document records how the tree-based circuit search (`TreeMessagePatching`)
was designed, the alternatives that were tried and abandoned, and where the
tree-vs-path comparison stands. It complements the code in
`src/ipe/graph_search.py`, `src/ipe/paths.py` and the driver experiment
`experiments/tree_vs_path.py`.

## 1. Motivation

`PathMessagePatching` discovers a circuit as a *set of independent paths*
`[EMBED, ..., FINAL]`, each scored by its isolated effect (the message the path
carries, removed from the clean residual, passed through the metric). Many of
those paths share long suffixes toward the root (e.g. hundreds of paths ending
in `... -> A9H9 -> FINAL`), so the natural question was whether the search
could instead grow a **single tree rooted at FINAL**, sharing those suffixes,
and whether scoring candidates *in the context of the tree found so far*
(joint ablation) would find a different — possibly better — circuit than
scoring each branch in isolation.

Two ingredients were added to support this:

- `evaluate_tree(root, metric)` (`src/ipe/paths.py:45`): generalizes
  `evaluate_path` to a whole tree. `get_tree_msg` recursively computes, for
  each node, the message it sends to its parent given the **joint** ablation of
  all its children's subtrees (children's messages are summed and removed from
  the node's clean input before its component is re-evaluated). A leaf emits
  `forward(None)`, matching path semantics. A single-child chain reduces
  exactly to `evaluate_path` — this invariant is asserted at the start of every
  `tree_vs_path.py` run (`check_invariant`).
- `TreeMessagePatching`: a backwards BFS from the root that grows the tree by
  attaching admitted candidates as children of the current frontier leaves.

## 2. Iterations (what was tried, in order)

### 2.1 v1 — Marginal contribution against a joint-ablation baseline (commit `8c556f7`)

The first implementation scored each candidate `C` at leaf `L` by its
**marginal** effect on the joint ablation of the tree `T` discovered so far:

```
score(C) = evaluate_tree(T + C) - evaluate_tree(T)
```

The marginal (rather than the raw tree score) was needed because the metric
grows monotonically with the ablated mass: the total tree score is dominated by
already-discovered branches and can no longer discriminate an individual
candidate. To keep scoring order-independent within a depth, the baseline was
frozen per BFS level: all leaves' candidates were scored against the same
`evaluate_tree(T)`, survivors were attached only after the whole level was
scored, then the baseline was recomputed.

**Bug found and fixed** (commit `0e8f934`): `evaluate_tree` on a childless root
called `get_tree_msg(root)` = `root.forward(None)`, i.e. the baseline for the
empty tree ablated the *entire* final residual instead of nothing. Fixed to
`message = get_tree_msg(root) if root.children else 0`.

### 2.2 v1.5 — Sibling-message caching optimization (commit `16af7c6`)

Scoring one candidate under v1 re-evaluated the whole tree
(`evaluate_tree(T + C)`), which is O(|tree|) forwards per candidate and became
the bottleneck as the tree grew. Two helpers made it O(depth):

- `refresh_tree_messages(root)`: a bottom-up pass caching on each node its
  outgoing message `_tree_msg` and the summed incoming children messages
  `_tree_incoming`.
- `candidate_message(root, leaf, candidate)`: recomputes only the
  `leaf -> root` chain with the candidate hypothetically attached, reusing the
  cached sibling messages at every level
  (`incoming = parent._tree_incoming - old + new`). The cache was refreshed
  once per BFS level, after attachment.

### 2.3 v2 (current) — Isolated branch contribution, no baseline (commit `6a3e1d6`)

Running v1 revealed that the marginal-contribution scoring made the comparison
against the path search ill-posed: a candidate's admission depended on which
sibling subtrees happened to be discovered already (interaction/saturation
effects under joint ablation), whereas the path search scores every path in
isolation. The two searches were exploring under different objectives, so
differences in the recovered circuits could not be attributed to the tree
structure itself.

The fix was to drop the joint-ablation baseline entirely: each candidate is now
scored by **its own isolated branch contribution**

```
score(C) = evaluate_path([C, leaf, ..., root])
```

— exactly the per-path score `PathMessagePatching` uses. Consequences:

- The tree is now, by construction, *the same set of paths* the path search
  admits, materialised as a suffix-sharing trie toward the root. Any path
  pruning strategy transfers directly to the tree.
- Sibling caching (v1.5) became unnecessary for scoring and the
  `refresh_tree_messages`/`candidate_message` machinery was removed from the
  scoring loop; `evaluate_tree` is retained for *reporting* the joint ablation
  of the final tree.
- Scoring is embarrassingly order-independent, so survivors are attached
  immediately.

The same commit added verbose debug logging (`setup_tree_debug_log`, writing
per-candidate ADMIT/reject records with contributions to
`tree_search_debug.log`) and the top-k search variant (below).

### 2.4 Search strategies: threshold vs. top-k beam

Both admission strategies exist for both searches, so the comparison can be
run apples-to-apples under either budget:

| strategy | path search | tree search | admission rule |
|---|---|---|---|
| `threshold` | `PathMessagePatching` | `TreeMessagePatching` | admit any candidate with contribution ≥ `min_contribution` (or `|contribution|` ≥ threshold with `include_negative`) |
| `topk` | `PathMessagePatching_LimitedLevelWidth` | `TreeMessagePatching_LimitedLevelWidth` | at each BFS depth, score all candidate extensions of all leaves and keep the global top `max_width` by `|contribution|` |

`TreeMessagePatching_LimitedLevelWidth` mirrors the path beam exactly; `topk`
compares the two searches at a matched width budget rather than a matched
threshold, which avoids the threshold meaning slightly different things when
path counts differ.

### 2.5 Granularity

Both searches expand attention at the per-head level
(`get_expansion_candidates(..., include_head=True)`); MLPs are per-block. The
path search's `batch_heads` two-stage block-then-head shortcut is **not**
implemented for the tree.

## 3. The comparison experiment (`experiments/tree_vs_path.py`)

One driver runs both searches with identical settings (same model, batch,
metric, caches — each search gets its own `clone_root` of the same configured
`FINAL_Node`) and reports components discovered, runtime, and overlap.

- **Invariant check** first: a single-child chain must score identically under
  `evaluate_path` and `evaluate_tree` (EMBED→MLP→FINAL and EMBED→ATTN→FINAL),
  exercising the run's actual caches and counterfactual patching.
- **Flags**: `--task {ioi,greater-than}`, `--strategy {threshold,topk}`,
  `--min-contribution`, `--max-width`, `--positional`, `--batch-size`,
  `--metric`.

### 3.1 IOI task

Setup mirrors `experiments/MIB/run_search.py`: `mib-bench/ioi` prompts with the
`s2_io_flip_counterfactual`, counterfactual (denoising) patching, and the
run_search convention of feeding counterfactual prompts as the clean run (and
vice-versa). Batches are bucketed to a single tokenised length.

Ground truth is the Wang et al. (2022) head set, grouped by functional role.
The report (`ioi_circuit_report.txt`) marks each known head `[PT]`/`[P-]`/etc.,
lists off-circuit heads and MLP layers found (IOI has no canonical MLP ground
truth — descriptive only), and draws the tree.

**Latest run** (topk, max_width=20, batch=4, positional, logit_difference):
path and tree recover the **same 17/26 known heads** (all Name Movers, all
Negative Name Movers, 4/8 Backup Name Movers, 3/4 S-Inhibition; misses are
mostly backup/induction heads). Each finds 5 off-circuit heads, all in layer 0.
Path produced 161 complete paths / 70 unique nodes; tree 90 branches / 65
unique nodes. This is the expected outcome under matched scoring: the tree is
the trie of (approximately) the same admitted paths.

### 3.2 Greater-than task

Synthetic Hanna et al. (2023) prompts:
`"The {noun} lasted from the year 17YY to the year 17"` with the standard
`YY→01` corruption as counterfactual. Ground truth here is a full circuit
*graph*, not a head set, so `GREATER_THAN_EDGES` encodes the published
data-flow edges and `build_ground_truth_tree` parses them into the same `Node`
structure the search produces; the report compares **tree-against-tree**:
branches (root→leaf component-identity paths) split into common /
ground-truth-only / tree-only / incomplete (pruned before reaching
embeddings), plus ASCII drawings of both trees and, for positional runs, a
layer × token-position DAG grid with orthogonal edge routing.

**Latest run** (topk, max_width=20, batch=4, non-positional,
logit_difference): ground truth expands to 826 branches; the tree search found
27 complete branches (23 incomplete), 7 in common — e.g.
`embeddings -> m0 -> a9.h1 -> logits`, `embeddings -> a0.h1 -> a9.h1 -> m9 ->
logits`. The overlap includes the key components (a0.h1, m0, a8.h11, a9.h1,
m8–m11) but branch-level recall is low at this width budget.

**Known caveats of the current greater-than setup** (vs. the official
`gpt2-greater-than` repo):

1. **Metric mismatch (most important).** Official metric is the probability
   difference `Σ P(YY' > YY) − Σ P(YY' ≤ YY)` over all valid two-digit year
   tokens; the experiment uses a single-token `logit_difference` proxy
   (`yy+1` good vs. `yy-1` bad). The published circuit was discovered under
   prob-diff, so the proxy weakens the comparison.
2. **Effectively unbalanced batch.** Candidates are generated noun-outer /
   YY-inner and bucketed by token length, so the default batch is one noun
   ("war") with the smallest YYs (2..21). The official dataset balances YY
   uniformly over 2..98 and samples 120 nouns.
3. **Century fixed at 17**; the official pool spans centuries 10–18 filtered
   through `get_valid_years` (two-token `[ XX, YY]` split, century edge years
   dropped).
4. A hand-picked 14-noun list instead of the official
   `cache/potential_nouns.txt`.

Planned/possible fixes, in order of value: balance the batch across YY (and
nouns) before length-bucketing; implement the prob-diff metric over the
valid-year token indices; optionally vary the century and adopt the official
noun list.

## 4. Current file map

| file | role |
|---|---|
| `src/ipe/paths.py` | `evaluate_path`, `get_tree_msg`/`evaluate_tree` (joint ablation, used for reporting and the invariant) |
| `src/ipe/graph_search.py` | `TreeMessagePatching` (threshold), `TreeMessagePatching_LimitedLevelWidth` (top-k beam), path-search counterparts, `setup_tree_debug_log` |
| `experiments/tree_vs_path.py` | driver: batch loading (IOI / greater-than), invariant check, both searches, overlap stats, ground-truth reports, ASCII tree + position-grid rendering |
| `ioi_circuit_report.txt`, `greater_than_report.txt` | latest generated reports |
| `tree_search_debug.log` | per-candidate ADMIT/reject trace of the last tree run |

## 5. Open questions / next steps

- Implement the greater-than prob-diff metric and a balanced batch (caveats
  above) so the ground-truth comparison is faithful to Hanna et al.
- Port the `batch_heads` block-then-head expansion shortcut to the tree search
  for parity in runtime comparisons.
- Now that scoring is matched, the interesting deltas between path and tree
  are (a) runtime/memory (suffix sharing avoids re-scoring shared suffixes)
  and (b) the reporting value of the joint-ablation score of the whole tree
  (`evaluate_tree` at the end of each run). If a *behaviorally* different tree
  search is wanted again, the v1 marginal-contribution scoring (with the v1.5
  sibling caching) is documented above and recoverable from commits
  `8c556f7`/`16af7c6`.
