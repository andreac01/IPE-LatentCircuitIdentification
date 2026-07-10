"""Local web UI for IPE circuit discovery.

Serves a single-page frontend (static/) and a small JSON/SSE API around the
path- and tree-based searches:

	POST /api/tokenize            tokenize a prompt (loads the model on first use)
	POST /api/search              start a discovery job -> {job_id}
	GET  /api/search/{id}/events  Server-Sent Events stream of search progress
	GET  /api/search/{id}         job status / final result (for reconnects)
	POST /api/search/{id}/cancel  request cancellation of a running job
	GET  /api/config              available models, metrics, methods, device

Run from the repo root:

	PYTHONPATH=src uvicorn visualization.server:app --port 8321
	# or simply
	python visualization/server.py

One search runs at a time (a single model forward pass already saturates the
device); extra jobs queue on a lock. The loaded HookedTransformer is cached per
model name and reused across jobs.
"""

import gc
import json
import math
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

# Allow `python visualization/server.py` without an installed ipe package.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
	sys.path.insert(0, str(_SRC))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformer_lens import HookedTransformer

from ipe.experiment import ExperimentManager
from ipe.paths import evaluate_tree
from ipe.graph_search import (
	PathMessagePatching,
	PathMessagePatching_LimitedLevelWidth,
	TreeMessagePatching,
	TreeMessagePatching_LimitedLevelWidth,
)
from ipe.webutils.serialization import graph_from_paths, graph_from_tree, node_to_dict

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = ["gpt2-small", "gpt2-medium", "gpt2-large", "pythia-160m", "pythia-410m"]
METRICS = ["logit_difference", "indirect_effect", "target_logit_percentage",
		   "target_probability_percentage", "kl_divergence"]

app = FastAPI(title="IPE circuit discovery")


# --------------------------------------------------------------------- models

_model_cache: dict[str, HookedTransformer] = {}
_model_lock = threading.Lock()


def get_model(name: str) -> HookedTransformer:
	"""Load (once) and cache a HookedTransformer, configured as in the
	experiments (fp16 on GPU, centered unembedding)."""
	if name not in MODELS:
		raise HTTPException(400, f"Unknown model '{name}'. Available: {MODELS}")
	with _model_lock:
		if name not in _model_cache:
			model = HookedTransformer.from_pretrained(
				name,
				device=DEVICE,
				torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
				center_unembed=True,
			)
			model.eval()
			_model_cache[name] = model
		return _model_cache[name]


# ----------------------------------------------------------------------- jobs

class SearchCancelled(BaseException):
	"""Raised inside the search's on_event callback to abort a job. Derives from
	BaseException on purpose: the search swallows Exception from observers."""


def _scrub(x):
	"""Replace non-finite floats with None. json.dumps emits bare NaN/Infinity,
	which is invalid JSON for the browser's JSON.parse — one NaN metric (easy
	with fp16) would otherwise silently kill the event that carries it."""
	if isinstance(x, float):
		return x if math.isfinite(x) else None
	if isinstance(x, dict):
		return {k: _scrub(v) for k, v in x.items()}
	if isinstance(x, (list, tuple)):
		return [_scrub(v) for v in x]
	return x


class Job:
	def __init__(self, job_id: str, params: dict):
		self.id = job_id
		self.params = params
		self.status = "queued"  # queued | running | complete | error | cancelled
		self.error: str | None = None
		self.result: dict | None = None
		self.events: list[dict] = []
		self.cond = threading.Condition()
		self.cancel_requested = False
		self.created = time.time()

	def push(self, event: dict) -> None:
		event = _scrub(event)
		with self.cond:
			self.events.append(event)
			self.cond.notify_all()


_jobs: OrderedDict[str, Job] = OrderedDict()
_jobs_lock = threading.Lock()
_run_lock = threading.Lock()  # one search at a time
MAX_JOBS = 20


def _register_job(job: Job) -> None:
	with _jobs_lock:
		_jobs[job.id] = job
		while len(_jobs) > MAX_JOBS:
			old_id, old = next(iter(_jobs.items()))
			if old.status in ("complete", "error", "cancelled"):
				_jobs.pop(old_id)
			else:
				break


def _get_job(job_id: str) -> Job:
	job = _jobs.get(job_id)
	if job is None:
		raise HTTPException(404, "Job not found")
	return job


# ------------------------------------------------------------------- requests

class TokenizeRequest(BaseModel):
	model: str
	prompt: str


class SearchRequest(BaseModel):
	model: str = "gpt2-small"
	prompts: list[str] = Field(min_length=1)
	targets: list[str] = Field(min_length=1)
	cf_prompts: list[str] = []
	cf_targets: list[str] = []
	method: str = "tree"           # tree | path
	strategy: str = "threshold"    # threshold | topk
	min_contribution: float = 0.05
	max_width: int = 20000
	metric: str = "logit_difference"
	positional: bool = True
	include_negative: bool = True


def _validate(req: SearchRequest, model: HookedTransformer) -> None:
	if req.method not in ("tree", "path"):
		raise HTTPException(400, "method must be 'tree' or 'path'")
	if req.strategy not in ("threshold", "topk"):
		raise HTTPException(400, "strategy must be 'threshold' or 'topk'")
	if req.metric not in METRICS:
		raise HTTPException(400, f"Unknown metric '{req.metric}'. Available: {METRICS}")
	if req.cf_prompts and len(req.cf_prompts) != len(req.prompts):
		raise HTTPException(400, "cf_prompts must match prompts in number")
	if len(req.targets) != len(req.prompts):
		raise HTTPException(400, "targets must match prompts in number")
	if req.cf_prompts and len(req.cf_targets) != len(req.cf_prompts):
		raise HTTPException(400, "cf_targets must match cf_prompts in number")

	ref_len = len(model.to_str_tokens(req.prompts[0], prepend_bos=True))
	for p in req.prompts + req.cf_prompts:
		n = len(model.to_str_tokens(p, prepend_bos=True))
		if req.positional and n != ref_len:
			raise HTTPException(400, f"Prompt {p!r} has {n} tokens, expected {ref_len} "
									 "(all prompts and counterfactuals must tokenize to the "
									 "same length for a positional search)")
	if req.metric != "kl_divergence":
		for t in req.targets + req.cf_targets:
			if len(model.to_str_tokens(t, prepend_bos=False)) != 1:
				raise HTTPException(400, f"Target {t!r} is not a single token "
										 "(tip: most tokens need a leading space)")


# -------------------------------------------------------------------- worker

def _run_search(job: Job, req: SearchRequest) -> None:
	last_leaf_emit = 0.0

	def on_event(event: dict) -> None:
		nonlocal last_leaf_emit
		if job.cancel_requested:
			raise SearchCancelled()
		kind = event.get("event")
		if kind == "leaf_done":
			# Progress ticks can be very frequent; throttle to ~10/s.
			now = time.monotonic()
			if now - last_leaf_emit < 0.1 and event["leaf"] != event["n_leaves"]:
				return
			last_leaf_emit = now
			job.push(event)
		elif kind == "admit":
			job.push({
				"event": "admit",
				"depth": event["depth"],
				"node": node_to_dict(event["node"]),
				"parent": node_to_dict(event["parent"]),
				"contribution": event["contribution"],
			})
		elif kind == "path_complete":
			job.push({
				"event": "path_complete",
				"depth": event["depth"],
				"path": [node_to_dict(n) for n in event["path"]],
				"contribution": event["contribution"],
			})
		else:  # depth_start / depth_end
			job.push(event)

	try:
		with _run_lock:
			job.status = "running"
			job.push({"event": "status", "status": "running", "message": f"Loading {req.model}..."})
			model = get_model(req.model)
			job.push({"event": "status", "status": "running", "message": "Running clean/counterfactual forward passes..."})

			has_cf = bool(req.cf_prompts)
			# With counterfactual denoising the counterfactual prompts are fed as
			# the clean run and vice-versa, exactly as experiments/tree_vs_path.py
			# and experiments/MIB/run_search.py do.
			experiment = ExperimentManager(
				model=model,
				prompts=req.cf_prompts if has_cf else req.prompts,
				targets=req.cf_targets if has_cf else req.targets,
				cf_prompts=req.prompts if has_cf else None,
				cf_targets=req.targets if has_cf else None,
				algorithm="PathMessagePatching",
				search_strategy="Threshold",
				algorithm_params={"min_contribution": req.min_contribution,
								  "include_negative": req.include_negative},
				metric=req.metric,
				positional_search=req.positional,
				patch_type="counterfactual" if has_cf else "zero",
				patch_clean_into_cf=has_cf,
			)
			metric = experiment.metric
			root = experiment.root

			tokens = model.to_str_tokens(req.prompts[0], prepend_bos=True)
			job.push({"event": "meta", "tokens": tokens,
					  "n_layers": model.cfg.n_layers, "n_heads": model.cfg.n_heads,
					  "positional": req.positional})

			t0 = time.time()
			if req.method == "tree":
				search = (TreeMessagePatching if req.strategy == "threshold"
						  else TreeMessagePatching_LimitedLevelWidth)
				kwargs = ({"min_contribution": req.min_contribution} if req.strategy == "threshold"
						  else {"max_width": req.max_width})
				out_root = search(model, metric, root, include_negative=req.include_negative,
								  on_event=on_event, **kwargs)
				graph = graph_from_tree(out_root)
				joint = float(evaluate_tree(out_root, metric))
			else:
				search = (PathMessagePatching if req.strategy == "threshold"
						  else PathMessagePatching_LimitedLevelWidth)
				kwargs = ({"min_contribution": req.min_contribution} if req.strategy == "threshold"
						  else {"max_width": req.max_width})
				paths = search(model, metric, root, include_negative=req.include_negative,
							   on_event=on_event, **kwargs)
				graph = graph_from_paths(paths)
				joint = None
			runtime = time.time() - t0

			job.result = _scrub({
				"graph": graph,
				"meta": {
					"tokens": tokens,
					"n_layers": model.cfg.n_layers,
					"n_heads": model.cfg.n_heads,
					"positional": req.positional,
					"method": req.method,
					"strategy": req.strategy,
					"metric": req.metric,
					"runtime": round(runtime, 2),
					"joint_tree_contribution": joint,
					"n_nodes": len(graph["nodes"]),
					"n_edges": len(graph["edges"]),
				},
			})
			job.status = "complete"
			job.push({"event": "result", **job.result})

			del experiment
			gc.collect()
			if DEVICE.type == "cuda":
				torch.cuda.empty_cache()
	except SearchCancelled:
		job.status = "cancelled"
		job.push({"event": "status", "status": "cancelled", "message": "Search cancelled."})
	except Exception as e:
		job.status = "error"
		job.error = f"{type(e).__name__}: {e}"
		job.push({"event": "status", "status": "error", "message": job.error})
	finally:
		with job.cond:
			job.cond.notify_all()


# ------------------------------------------------------------------ endpoints

@app.get("/api/config")
def config():
	return {
		"models": MODELS,
		"loaded_models": list(_model_cache),
		"metrics": METRICS,
		"methods": ["tree", "path"],
		"strategies": ["threshold", "topk"],
		"device": str(DEVICE),
	}


@app.post("/api/tokenize")
def tokenize(req: TokenizeRequest):
	model = get_model(req.model)
	tokens = model.to_str_tokens(req.prompt, prepend_bos=True)
	return {"tokens": tokens, "length": len(tokens),
			"n_layers": model.cfg.n_layers, "n_heads": model.cfg.n_heads}


@app.post("/api/search")
def start_search(req: SearchRequest):
	model = get_model(req.model)
	_validate(req, model)
	job = Job(str(uuid.uuid4()), req.model_dump())
	_register_job(job)
	threading.Thread(target=_run_search, args=(job, req), daemon=True).start()
	return {"job_id": job.id, "status": job.status}


@app.get("/api/search/{job_id}")
def job_status(job_id: str):
	job = _get_job(job_id)
	out = {"job_id": job.id, "status": job.status, "n_events": len(job.events),
		   "params": job.params, "error": job.error}
	if job.result is not None:
		out["result"] = job.result
	return out


@app.post("/api/search/{job_id}/cancel")
def cancel_job(job_id: str):
	job = _get_job(job_id)
	job.cancel_requested = True
	return {"job_id": job.id, "status": job.status, "cancel_requested": True}


@app.get("/api/search/{job_id}/events")
def job_events(job_id: str):
	job = _get_job(job_id)

	def stream():
		# Never yield while holding job.cond: the generator suspends at yield,
		# and the worker needs the lock to push events.
		idx = 0
		while True:
			with job.cond:
				if idx >= len(job.events) and job.status in ("queued", "running"):
					job.cond.wait(timeout=15.0)
				batch = job.events[idx:]
				idx += len(batch)
				done = job.status not in ("queued", "running")
			for event in batch:
				yield f"data: {json.dumps(event)}\n\n"
			if done and not batch:
				yield f"data: {json.dumps({'event': 'end', 'status': job.status})}\n\n"
				return
			if not batch:
				yield ": keep-alive\n\n"

	return StreamingResponse(stream(), media_type="text/event-stream",
							 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
	return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="127.0.0.1", port=8321)
