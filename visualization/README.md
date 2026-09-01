# IPE circuit discovery — interactive visualization

A local web UI for running the IPE searches (path-based `PathMessagePatching`
and tree-based `TreeMessagePatching`, threshold or top-k strategy) — plus the
**ACDC** baseline vendored under `benchmark/` — and viewing the discovered
circuit on a stylized model grid:

- **x axis** = token positions (with the tokenized clean prompt as the header),
- **y axis** = layers (FINAL at the top, per layer MLP above ATTN, EMB at the bottom),
- the whole model is drawn as faint placeholder cells; discovered components
  appear as head-level chips colored by signed contribution (slate blue
  positive, brick red negative). Branches that reach an embedding node are
  drawn as thicker **coral-orange** edges; pruned branches stay as thin dashed
  gray lines and dashed chips. A horizontal jog in an edge is a key/value read
  at an earlier position. The styling follows the Anthropic publication look
  (warm cream/ivory surfaces, coral `#D97757` accent, serif headings).

Search progress is **streamed live** (Server-Sent Events): admitted nodes are
revealed one by one with a pulse, edges pump coral dashes from the root toward
the leaves while the search runs, and a run can be cancelled mid-way.

## ACDC baseline

The **ACDC** button in the Algorithm segment runs
`benchmark/Automatic-Circuit-Discovery` on the same prompts and draws its result
on the same grid, for a like-for-like comparison. The checkout is used read-only:
`src/ipe/webutils/acdc_bridge.py` only calls `TLACDCExperiment.step()` and
translates the resulting `TLACDCCorrespondence` into our graph JSON. ACDC's own
graphviz rendering is switched off (its `show` is rebound to a no-op, and the
`pygraphviz` import it needs is stubbed, so no system graphviz install is
required).

Three things differ from the IPE view and are worth knowing before reading the
picture:

- **The x axis is the attention head, not the token position.** ACDC is
  position-agnostic (`TLACDCExperiment` raises if you pass `positions`), so all
  144 gpt2-small heads would otherwise pile into one column. Layers still run up
  the y axis; MLP/EMB/FINAL sit in the trailing `mlp / emb` column.
- **One grid node per component.** ACDC works on the hook graph, where a head is
  up to seven nodes (`hook_{q,k,v}`, `hook_{q,k,v}_input`, `hook_result`). Those
  collapse into one chip; which of Q/K/V survived shows up in the tooltip as
  *patched streams*. The summary reports both counts (grid edges vs. the
  hook-level edges ACDC itself counts).
- **Contribution means "effect of cutting"**: `evaluated_metric - old_metric` for
  the strongest edge the component sends. ACDC minimizes its metric, so positive
  (blue) = removing it hurts = important, the same reading as the IPE colors.
  Dashed chips/edges are components ACDC left dangling off the input, which its
  own README notes it does.

Parameters: threshold τ (default 0.0575, the paper's KL value), metric
(`kl_div` needs no target; `logit_diff` uses the target as the correct answer and
the *counterfactual* target as the wrong one; `nll` uses the target), and the
counterfactual prompts — empty means zero ablation. The IPE-only knobs (strategy,
patching direction, positional, include-negative) are hidden because ACDC has no
equivalent.

**It is slow**, and how slow depends strongly on τ: ACDC evaluates one forward
pass per candidate edge (32.9k of them in the full gpt2-small graph), but only
visits nodes that are still connected, so a high τ prunes the work away early.
Measured on CPU with one 15-token IOI prompt: τ=0.15 converges in ~6 min (35
nodes visited, 15 hook-level edges left → 9 grid nodes), while a low τ keeps most
of the graph alive and runs far longer — against seconds for an IPE search.
Progress reports how far through the node order it is and how many edges are
left, and streams a snapshot of the graph once it has shrunk enough to draw.
Cancel works between steps.

## Run

From the repo root (needs `fastapi` and `uvicorn` in addition to the core deps):

```bash
PYTHONPATH=src python visualization/server.py
# or
PYTHONPATH=src uvicorn visualization.server:app --port 8321
```

then open http://127.0.0.1:8321. The first tokenize/search on a model loads it
once and caches it (fp16 on GPU, fp32 on CPU, centered unembedding — the same
configuration as `experiments/tree_vs_path.py`). One search runs at a time;
extra requests queue.

## Usage notes

- **Prompts**: one per line (a batch); with *positional search* enabled every
  prompt and counterfactual must tokenize to the same length. Targets are one
  single token per line — most need a leading space (` Mary`).
- **Counterfactuals**: when provided, the search uses counterfactual (denoising)
  patching, feeding the counterfactual prompts as the clean run and vice-versa
  exactly as `experiments/tree_vs_path.py` does. Leave empty for zero patching.
- **Tree vs Path**: the tree search materializes the same set of paths as a
  suffix-sharing trie; both are shown on the same grid after collapsing to a
  DAG (one grid node per component + position; hover a chip to see how many
  tree nodes merged into it, which streams were patched, and its K/V reads).
- After a run, the **filter slider** hides nodes below a |contribution|
  threshold without re-running, and the graph can be downloaded as JSON.

## Layout of the code

- `server.py` — FastAPI app: model cache, job queue (background thread + lock),
  `POST /api/search` (`method` selects `tree` / `path` / `acdc`), SSE stream at
  `GET /api/search/{id}/events`, `POST /api/tokenize` for the live token header.
- `static/` — dependency-free frontend (vanilla JS + SVG).
- Graph serialization (tree/paths → grid DAG JSON) lives in
  `src/ipe/webutils/serialization.py`; live progress events come from the
  optional `on_event` callback added to the search functions in
  `src/ipe/graph_search.py`.
- `src/ipe/webutils/acdc_bridge.py` — everything ACDC: importing the vendored
  checkout, building its metric from the prompt boxes, driving `exp.step()`, and
  translating `TLACDCCorrespondence` into the same grid DAG JSON. The ACDC
  checkout itself is never modified.
