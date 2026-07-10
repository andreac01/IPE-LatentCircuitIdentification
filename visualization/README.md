# IPE circuit discovery — interactive visualization

A local web UI for running the IPE searches (path-based `PathMessagePatching`
and tree-based `TreeMessagePatching`, threshold or top-k strategy) and viewing
the discovered circuit on a stylized model grid:

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
  `POST /api/search`, SSE stream at `GET /api/search/{id}/events`,
  `POST /api/tokenize` for the live token header.
- `static/` — dependency-free frontend (vanilla JS + SVG).
- Graph serialization (tree/paths → grid DAG JSON) lives in
  `src/ipe/webutils/serialization.py`; live progress events come from the
  optional `on_event` callback added to the search functions in
  `src/ipe/graph_search.py`.
