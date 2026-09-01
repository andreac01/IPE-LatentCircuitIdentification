"""Run ACDC (benchmark/Automatic-Circuit-Discovery) from the web UI.

ACDC is vendored read-only under `benchmark/Automatic-Circuit-Discovery`; nothing
here modifies it. We only

  * import `TLACDCExperiment` (stubbing the graphviz dependency its plotting
	module pulls in at import time, and neutering the PNG it renders after every
	step -- we have our own visualization),
  * drive `exp.step()` in a loop the way `acdc/main.py` does,
  * translate the resulting `TLACDCCorrespondence` into the same grid-DAG JSON
	that `serialization.py` produces for the IPE searches.

Differences from the IPE searches that the translation has to bridge:

  * ACDC is *position agnostic* (`TLACDCExperiment` raises on `positions`), so
	every node comes out with `position: None`. The frontend therefore lays the
	x axis out by attention head instead of by token (`meta.layout == "head"`).
  * ACDC works on the hook-level graph, where one attention head is up to seven
	nodes (`hook_q/k/v`, `hook_{q,k,v}_input`, `hook_result`). Those collapse to
	the single `attn:layer:head:*` grid node, and the q/k/v distinction survives
	as the node's `variants` (the same field the IPE tooltip shows as "patched
	streams").
  * ACDC is subtractive: it starts from the complete graph and deletes edges,
	whereas the IPE searches grow one. There is no meaningful "admit" stream, so
	progress is reported as `progress` events plus periodic whole-graph
	snapshots once the graph is small enough to draw.
"""

import io
import sys
import time
import types
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path

import torch

ACDC_ROOT = Path(__file__).resolve().parents[3] / "benchmark" / "Automatic-Circuit-Discovery"

# ACDC minimizes its metric; `kl_div` needs no target token.
ACDC_METRICS = ["kl_div", "logit_diff", "nll"]
ACDC_METRICS_NEEDING_TARGETS = ["logit_diff", "nll"]
ACDC_METRICS_NEEDING_CF_TARGETS = ["logit_diff"]

_acdc = None


class _Sink(io.TextIOBase):
	"""Throw-away stdout (an in-memory buffer would grow for the whole run)."""

	def write(self, s):
		return len(s)


def _load_acdc():
	"""Import the vendored ACDC package once and hand back the few names we use."""
	global _acdc
	if _acdc is not None:
		return _acdc
	if not ACDC_ROOT.is_dir():
		raise RuntimeError(f"ACDC checkout not found at {ACDC_ROOT}")
	if str(ACDC_ROOT) not in sys.path:
		sys.path.insert(0, str(ACDC_ROOT))

	# `acdc.acdc_graphics` imports pygraphviz (and lazily cmapy) purely to render
	# its own graphs, which we never call. Stub them so the demo does not need a
	# system graphviz install.
	for name in ("pygraphviz", "cmapy"):
		if name not in sys.modules:
			try:
				__import__(name)
			except ImportError:
				stub = types.ModuleType(name)

				# acdc_graphics uses `pgv.AGraph` in a return annotation, which
				# is evaluated at import time, so the stub has to answer any
				# attribute with *something* -- but dunders must still fail, or
				# `inspect` chokes on the module later.
				def _missing(attr, _name=name):
					if attr.startswith("__"):
						raise AttributeError(attr)
					return type(attr, (), {})

				stub.__getattr__ = _missing
				sys.modules[name] = stub

	from acdc import TLACDCExperiment as experiment_module
	from acdc.TLACDCEdge import EdgeType
	from acdc.TLACDCExperiment import TLACDCExperiment
	from acdc.acdc_utils import kl_divergence, logit_diff_metric, negative_log_probs

	# `TLACDCExperiment.step` unconditionally writes ims/img_new_*.png through
	# this reference. Rebinding it in the module namespace is the least invasive
	# way to switch ACDC's own visualization off without touching the checkout.
	experiment_module.show = lambda *args, **kwargs: None

	_acdc = types.SimpleNamespace(
		TLACDCExperiment=TLACDCExperiment,
		EdgeType=EdgeType,
		kl_divergence=kl_divergence,
		logit_diff_metric=logit_diff_metric,
		negative_log_probs=negative_log_probs,
	)
	return _acdc


# ------------------------------------------------------------------- metrics

def build_metric(model, name, clean_tokens, target_ids, wrong_ids):
	"""An ACDC metric: logits -> scalar tensor, lower is better.

	`target_ids` / `wrong_ids` are per-prompt token ids for the last position."""
	acdc = _load_acdc()
	if name == "kl_div":
		with torch.no_grad():
			model.reset_hooks()
			# Keep the model's dtype: F.kl_div wants both arguments to match.
			base = model(clean_tokens)[:, -1, :]
			base_logprobs = torch.nn.functional.log_softmax(base, dim=-1)
		return partial(acdc.kl_divergence, base_model_logprobs=base_logprobs,
					   last_seq_element_only=True,
					   base_model_probs_last_seq_element_only=False)
	if name == "logit_diff":
		return partial(acdc.logit_diff_metric, correct_labels=target_ids, wrong_labels=wrong_ids)
	if name == "nll":
		return partial(acdc.negative_log_probs, labels=target_ids, last_seq_element_only=True)
	raise ValueError(f"Unknown ACDC metric '{name}'. Available: {ACDC_METRICS}")


# --------------------------------------------------------- graph translation

def _gid(key: tuple) -> str:
	"""Same string form as `serialization.grid_id`, so both sources of graphs
	speak the same node ids."""
	return ":".join("*" if p is None else str(p) for p in key)


def _parse_hook(name: str, index) -> tuple:
	"""ACDC (hook name, TorchIndex) -> (grid key, q/k/v stream or None).

	Every hook belonging to one attention head maps to the same grid key; the
	stream letter is what distinguishes `hook_q_input` from `hook_v_input`."""
	head = None
	tup = index.hashable_tuple
	if len(tup) >= 3 and isinstance(tup[2], int):
		head = tup[2]
	layer = int(name.split(".")[1]) if name.startswith("blocks.") else None

	if name in ("hook_embed", "hook_pos_embed") or name.endswith("hook_resid_pre"):
		return ("embed", None), None
	if name.endswith("hook_resid_post"):
		return ("final", None), None
	if "hook_mlp_out" in name or "hook_mlp_in" in name:
		return ("mlp", layer, None), None
	for letter in "qkv":
		if name.endswith(f"hook_{letter}") or name.endswith(f"hook_{letter}_input"):
			return ("attn", layer, head, None), letter.upper()
	if "attn.hook_result" in name:
		return ("attn", layer, head, None), None
	return ("unknown", name), None


def graph_from_correspondence(corr) -> dict:
	"""Collapse the surviving ACDC edges into the grid DAG the frontend eats.

	Edge orientation matches `serialization.py`: `source` is upstream (the ACDC
	*parent* / sender), `target` is downstream (the ACDC *child* / receiver)."""
	acdc = _load_acdc()
	nodes: dict[str, dict] = {}
	edges: dict[tuple, dict] = {}

	def touch(key, stream=None) -> str:
		nid = _gid(key)
		node = nodes.get(nid)
		if node is None:
			node = {
				"id": nid, "kind": key[0], "layer": None, "head": None, "position": None,
				"contribution": None, "merged": 1, "complete": False,
				"variants": [], "kv_positions": [],
			}
			if key[0] == "mlp":
				node["layer"] = key[1]
			elif key[0] == "attn":
				node["layer"], node["head"] = key[1], key[2]
			nodes[nid] = node
		if stream and stream not in node["variants"]:
			node["variants"].append(stream)
		return nid

	for (child_name, child_idx, parent_name, parent_idx), edge in corr.all_edges().items():
		if not edge.present or edge.edge_type == acdc.EdgeType.PLACEHOLDER:
			continue
		src_key, _ = _parse_hook(parent_name, parent_idx)
		dst_key, dst_stream = _parse_hook(child_name, child_idx)
		src = touch(src_key)
		dst = touch(dst_key, dst_stream)
		if src == dst:
			# hook_q_input -> hook_q and friends: an intra-head edge whose only
			# information (which stream is live) is already on the node.
			continue
		effect = edge.effect_size
		e = edges.get((src, dst))
		if e is None:
			e = edges[(src, dst)] = {"source": src, "target": dst, "count": 0,
									 "contribution": None, "complete": False}
		e["count"] += 1
		if effect is not None:
			if e["contribution"] is None or abs(effect) > abs(e["contribution"]):
				e["contribution"] = float(effect)
			# A component's score is the strongest effect of cutting one of the
			# edges it sends: that is what "this thing matters" means in ACDC.
			node = nodes[src]
			if node["contribution"] is None or abs(effect) > abs(node["contribution"]):
				node["contribution"] = float(effect)

	_mark_complete(nodes, edges)
	return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def _mark_complete(nodes: dict, edges: dict) -> None:
	"""Flag everything on a path back to the embeddings, the way the IPE graphs
	flag branches that reached an EMBED node. ACDC leaves plenty of dangling
	heads behind (its own README notes a0.0_q/a0.0_k hanging off the input), and
	the frontend draws those dashed."""
	forward: dict[str, list] = {}
	for e in edges.values():
		forward.setdefault(e["source"], []).append(e)

	stack = [nid for nid, n in nodes.items() if n["kind"] == "embed"]
	seen = set(stack)
	while stack:
		cur = stack.pop()
		nodes[cur]["complete"] = True
		for e in forward.get(cur, []):
			e["complete"] = True
			if e["target"] not in seen:
				seen.add(e["target"])
				stack.append(e["target"])


# -------------------------------------------------------------------- driver

def run_acdc(model, prompts, targets, cf_prompts, cf_targets, *,
			 metric_name="kl_div", threshold=0.0575, abs_value_threshold=False,
			 on_event=None, max_steps=100_000, quiet=True,
			 snapshot_interval=2.0, snapshot_max_edges=600) -> tuple[dict, dict]:
	"""Run ACDC to convergence and return (grid graph, info).

	`on_event` gets the same kind of dicts the IPE searches emit and may raise to
	cancel (that is how the server aborts a job). The model's hooks and the three
	cfg flags ACDC requires are restored on the way out, so the shared cached
	HookedTransformer stays usable by the IPE searches."""
	acdc = _load_acdc()
	emit = on_event or (lambda ev: None)

	ds = model.to_tokens(prompts)
	ref_ds = model.to_tokens(cf_prompts) if cf_prompts else None
	zero_ablation = ref_ds is None
	if ref_ds is not None and ref_ds.shape != ds.shape:
		# to_tokens right-pads silently; ACDC would then patch across padding.
		raise ValueError(f"counterfactual prompts tokenize to {tuple(ref_ds.shape)}, "
						 f"clean prompts to {tuple(ds.shape)} — they must match")

	target_ids = wrong_ids = None
	if metric_name in ACDC_METRICS_NEEDING_TARGETS:
		target_ids = torch.tensor([model.to_single_token(t) for t in targets], device=ds.device)
	if metric_name in ACDC_METRICS_NEEDING_CF_TARGETS:
		wrong_ids = torch.tensor([model.to_single_token(t) for t in cf_targets], device=ds.device)

	# ACDC needs per-head outputs and per-head q/k/v inputs to be hookable; the
	# IPE searches do not, so the shared model is not configured for it.
	old_flags = (model.cfg.use_attn_result, model.cfg.use_split_qkv_input,
				 getattr(model.cfg, "use_hook_mlp_in", None))
	model.reset_hooks()
	model.set_use_attn_result(True)
	model.set_use_split_qkv_input(True)
	if old_flags[2] is not None:
		model.set_use_hook_mlp_in(True)

	try:
		# ACDC prints one line per hook at setup and one per node visited (a few
		# thousand lines for gpt2-small). Swallow them rather than flooding the
		# server log; safe because the server serializes jobs behind one lock and
		# logs through `logging` (stderr), not print.
		with torch.no_grad(), redirect_stdout(_Sink() if quiet else sys.stdout):
			metric = build_metric(model, metric_name, ds, target_ids, wrong_ids)
			emit({"event": "status", "status": "running",
				  "message": "Building the ACDC computational graph and corrupted cache..."})

			on_gpu = model.cfg.device is not None and "cuda" in str(model.cfg.device)
			exp = acdc.TLACDCExperiment(
				model=model,
				threshold=threshold,
				ds=ds,
				ref_ds=ref_ds,
				metric=metric,
				zero_ablation=zero_ablation,
				abs_value_threshold=abs_value_threshold,
				using_wandb=False,
				verbose=False,
				hook_verbose=False,
				indices_mode="reverse",
				names_mode="normal",
				online_cache_cpu=not on_gpu,
				corrupted_cache_cpu=not on_gpu,
				add_sender_hooks=True,
				add_receiver_hooks=False,
				remove_redundant=False,
				show_full_index=False,
			)

			t0 = time.time()
			# ACDC walks its reverse-topological node order monotonically,
			# skipping the nodes that ended up disconnected. Its step counter is
			# therefore a poor progress signal (it only counts the nodes it
			# actually evaluated); how far along that order we are is the honest
			# one.
			order = {(name, index): i
					 for i, (name, index) in enumerate(
						 (name, index) for name, by_index in exp.corr.graph.items()
						 for index in by_index)}
			total_nodes = len(order)
			initial_edges = exp.count_no_edges()
			last_snapshot = 0.0
			steps = 0

			for steps in range(1, max_steps + 1):
				if exp.current_node is None:
					break
				node_name = str(exp.current_node)
				exp.step(testing=False)
				n_edges = exp.cur_edges
				at = (total_nodes if exp.current_node is None else
					  order.get((exp.current_node.name, exp.current_node.index), 0))

				event = {
					"event": "progress",
					"frac": min(1.0, at / max(total_nodes, 1)),
					"text": f"{at}/{total_nodes} nodes · {n_edges} edges left",
					"message": f"ACDC at {node_name} — {n_edges}/{initial_edges} edges, "
							   f"metric {exp.cur_metric:.4f}",
				}
				now = time.monotonic()
				if now - last_snapshot >= snapshot_interval:
					snapshot = graph_from_correspondence(exp.corr)
					# The full graph has tens of thousands of collapsed edges;
					# only start streaming it once it is drawable.
					if len(snapshot["edges"]) <= snapshot_max_edges:
						event["graph"] = snapshot
						last_snapshot = now
				emit(event)

			graph = graph_from_correspondence(exp.corr)
			info = {
				"runtime": round(time.time() - t0, 2),
				"steps": steps,
				"acdc_edges": exp.count_no_edges(),
				"initial_acdc_edges": initial_edges,
				"final_metric": float(exp.cur_metric),
				"threshold": threshold,
				"metric": metric_name,
				"ablation": "zero" if zero_ablation else "counterfactual",
			}
			return graph, info
	finally:
		model.reset_hooks()
		model.set_use_attn_result(old_flags[0])
		model.set_use_split_qkv_input(old_flags[1])
		if old_flags[2] is not None:
			model.set_use_hook_mlp_in(old_flags[2])
