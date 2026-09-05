"""Model-agnostic knockout evaluation of an IPE circuit.

This is the harness of `experiments/faithfulness_completeness.ipynb`, lifted out of the
notebook and parameterised by the model, so that one implementation serves GPT-2 small,
Qwen2.5-0.5B and Llama-3-8B in the same sweep. The notebook version reads `N_LAYERS` /
`N_HEADS` from module globals, which only ever works for one model at a time.

**Faithfulness** is the definition of *Interpretability in the Wild* (Wang et al., 2023, §3):
given a circuit `C`, run the model with every component *not* in `C` mean-ablated over the
ABC distribution and read off the IOI logit difference,

    F(C) = E[ logit(IO) - logit(S) ]   with the complement of C knocked out,

normalised so that `1.0` is the full model and `0.0` the empty circuit:

    faithfulness(C) = (F(C) - F(empty)) / (F(M) - F(empty)).

Only **node** granularity is implemented here: the output of every component outside `C` is
replaced by its ablated value, and everything downstream recomputes. That is literally Wang
et al.'s knockout, it is the granularity the head-level ground truth is defined on, and it is
the one that pairs with a "percentage of nodes retained" column. Edge granularity needs
`use_split_qkv_input` / `use_hook_mlp_in`, whose activation footprint is not affordable for an
8B model on a single 24GB card; `faithfulness_completeness.ipynb` has that variant.

Two evaluation **scopes**, both reported, because the difference between them is itself a
measurement:

* ``all``        - the circuit exactly as found (heads *and* MLPs); everything else ablated.
* ``attention``  - only the circuit's heads count as the circuit and every MLP is left intact
                   as unspecified infrastructure. Ablating the MLP sublayers damages the model
                   on its own, so under ``all`` a circuit that found few MLPs scores near zero
                   however good its heads are. ``attention - all`` is the faithfulness carried
                   by the MLPs a search did *not* find.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import torch

INPUT = ("input",)
LOGITS = ("logits",)
SCOPES = ("all", "attention")


# --------------------------------------------------------------------------- topology


class ModelGraph:
    """The computational graph the knockout works on, for one model.

    Nodes are ``("attn", layer, head)``, ``("mlp", layer)`` and the two boundary nodes
    ``INPUT`` (embeddings + positional embeddings) and ``LOGITS``.
    """

    def __init__(self, n_layers: int, n_heads: int):
        self.n_layers, self.n_heads = n_layers, n_heads

    @classmethod
    def of(cls, model) -> "ModelGraph":
        return cls(model.cfg.n_layers, model.cfg.n_heads)

    @property
    def all_components(self) -> list[tuple]:
        """Every ablatable component, in computation order."""
        comps = []
        for l in range(self.n_layers):
            comps += [("attn", l, h) for h in range(self.n_heads)]
            comps.append(("mlp", l))
        return comps

    @property
    def n_components(self) -> int:
        return self.n_layers * (self.n_heads + 1)

    @property
    def all_mlps(self) -> set:
        return {("mlp", l) for l in range(self.n_layers)}

    def upstream_of(self, dest: tuple) -> list[tuple]:
        """Components (plus INPUT) whose output reaches `dest`'s input, in computation order."""
        if dest == LOGITS:
            upto_attn = upto_mlp = self.n_layers
        elif dest[0] == "attn":
            upto_attn = upto_mlp = dest[1]
        elif dest[0] == "mlp":
            upto_attn, upto_mlp = dest[1] + 1, dest[1]  # an MLP also reads its own layer's heads
        else:
            raise ValueError(f"{dest} is not a destination")
        ups = [INPUT]
        for l in range(self.n_layers):
            if l < upto_attn:
                ups += [("attn", l, h) for h in range(self.n_heads)]
            if l < upto_mlp:
                ups.append(("mlp", l))
        return ups

    @staticmethod
    def streams_of_dest(dest: tuple) -> tuple[str, ...]:
        if dest == LOGITS:
            return ("resid",)
        return ("q", "k", "v") if dest[0] == "attn" else ("mlp",)

    def fully_connected_edges(self, nodes: set) -> set:
        """Every edge the model's topology allows among `nodes` (boundary nodes always in).

        Only meaningful to an edge-granularity knockout, which this module does not implement;
        it is quadratic in the component count (~3.3M tuples for a 32x32 model), so nothing on
        the node-granularity path may call it.
        """
        keep = set(nodes) | {INPUT, LOGITS}
        edges = set()
        for v in keep:
            if v == INPUT:
                continue
            for stream in self.streams_of_dest(v):
                edges.update((u, v, stream) for u in self.upstream_of(v) if u in keep)
        return edges

    def full_circuit(self) -> "Circuit":
        """Every component. Carries no edges: the node knockout reads only `nodes`."""
        comps = set(self.all_components)
        return Circuit("full model", comps | {INPUT, LOGITS}, set())

    def empty_circuit(self) -> "Circuit":
        return Circuit("empty", {INPUT, LOGITS}, set())


def component_label(c: tuple) -> str:
    if c == INPUT:
        return "input"
    if c == LOGITS:
        return "logits"
    return f"a{c[1]}.h{c[2]}" if c[0] == "attn" else f"m{c[1]}"


# --------------------------------------------------------------------------- circuit


@dataclass
class Circuit:
    name: str
    nodes: set = field(default_factory=set)  # components, boundary nodes included
    edges: set = field(default_factory=set)  # (src, dst, stream)
    threshold: float = float("nan")
    n_paths: int = 0  # complete paths that survived
    scope: str = "all"

    @property
    def components(self) -> set:
        """Circuit nodes the knockout can actually ablate (no boundary nodes)."""
        return {n for n in self.nodes if n not in (INPUT, LOGITS)}

    @property
    def heads(self) -> set:
        return {(l, h) for _, l, h in (n for n in self.nodes if n[0] == "attn")}

    @property
    def mlps(self) -> set:
        return {n[1] for n in self.nodes if n[0] == "mlp"}


def component_of(nd: dict) -> tuple:
    """The graph component a serialised search node denotes."""
    t = nd["type"]
    if t == "attn":
        return ("attn", nd["layer"], nd["head"])
    if t == "mlp":
        return ("mlp", nd["layer"])
    if t == "embed":
        return INPUT
    if t == "final":
        return LOGITS
    raise ValueError(t)


def streams_into(nd: dict) -> tuple[str, ...]:
    """Which input stream(s) of `nd` its children feed.

    Readable even in a non-positional search: `get_expansion_candidates` emits a predecessor
    either for the query branch (`patch_query`) or for the key/value branch, so a node's own
    patch flags say which of *its* inputs its children feed.
    """
    t = nd["type"]
    if t == "mlp":
        return ("mlp",)
    if t == "final":
        return ("resid",)
    if t == "attn":
        return tuple(k for k in ("q", "k", "v") if nd.get(k)) or ("q", "k", "v")
    raise ValueError(f"{t} cannot be the destination of an edge")


def circuit_from_paths(paths: list[dict], threshold: float = None, name: str = "circuit") -> Circuit:
    """Build a circuit from serialised search output.

    `paths` is a list of ``{"contribution": float, "nodes": [leaf, ..., root]}`` records, the
    format `serialise_paths` writes. With `threshold` given, a complete path is kept only if
    *every* one of its nodes clears it in absolute value -- which reproduces exactly what a
    Threshold search at that value would have returned, since each path is scored in isolation.
    """
    nodes, edges, kept = set(), set(), 0
    for path in paths:
        chain = path["nodes"]
        if threshold is not None:
            scores = [abs(n["contribution"]) for n in chain if n.get("contribution") is not None]
            if scores and min(scores) < threshold:
                continue
        kept += 1
        for child, parent in zip(chain[:-1], chain[1:]):
            c, p = component_of(child), component_of(parent)
            nodes.add(c)
            nodes.add(p)
            edges.update((c, p, stream) for stream in streams_into(parent))
    return Circuit(name=name, nodes=nodes, edges=edges,
                   threshold=float("nan") if threshold is None else threshold, n_paths=kept)


def scope_circuit(circuit: Circuit, scope: str, graph: ModelGraph, name: str = None) -> Circuit:
    """Re-read a circuit under one of the two evaluation scopes (see the module docstring).

    Nothing about the search changes; this is a re-reading at evaluation time.
    """
    if scope == "all":
        return Circuit(name or circuit.name, set(circuit.nodes), set(circuit.edges),
                       circuit.threshold, circuit.n_paths, "all")
    if scope != "attention":
        raise ValueError(scope)

    heads = {n for n in circuit.nodes if n[0] == "attn"}
    nodes = heads | graph.all_mlps | {INPUT, LOGITS}
    # The circuit's own wiring among the non-MLP nodes. The MLPs are wired in completely, but
    # that wiring is not materialised: the node knockout reads only `nodes`, and enumerating it
    # is quadratic in the component count.
    own = {e for e in circuit.edges if e[0][0] != "mlp" and e[1][0] != "mlp"}
    return Circuit(name or circuit.name, nodes, own,
                   circuit.threshold, circuit.n_paths, "attention")


# --------------------------------------------------------------------------- data


@dataclass
class EvalSet:
    tokens: torch.Tensor      # [N, pos]  clean IOI prompts
    abc_tokens: torch.Tensor  # [N, pos]  ABC counterfactuals (the ablation distribution)
    io: torch.Tensor          # [N]       indirect-object token ids
    s: torch.Tensor           # [N]       subject token ids

    def __len__(self) -> int:
        return self.tokens.shape[0]


# --------------------------------------------------------------------------- knockout


class Knockout:
    """Runs the model with everything outside a circuit ablated, and reports the IOI logit diff.

    The ablation values are the mean of the ABC counterfactual run, accumulated **over
    minibatches** rather than by caching the whole evaluation set at once: with
    `use_attn_result=True` a 32-layer model retains `[N, pos, n_heads, d_model]` per layer,
    which is several GB for an 8B model at N=64 and is what would otherwise OOM.
    """

    def __init__(self, model, evalset: EvalSet, graph: ModelGraph,
                 ablation: str = "mean", minibatch: int = 16):
        if ablation != "mean":
            raise ValueError("only ablation='mean' (Wang et al. ABC mean-ablation) is implemented")
        self.model, self.eval, self.graph, self.minibatch = model, evalset, graph, minibatch
        self._hook_names = self._names()
        self._abl = self._mean_ablation()

    def _names(self) -> set:
        return {f"blocks.{l}.{suffix}"
                for l in range(self.graph.n_layers)
                for suffix in ("attn.hook_result", "hook_mlp_out")} | {"blocks.0.hook_resid_pre"}

    @torch.no_grad()
    def _mean_ablation(self) -> dict:
        """Per-component ablated output, averaged over the ABC set, shape [1, pos, d_model]."""
        n, mb = len(self.eval), self.minibatch
        acc: dict = {}
        for i in range(0, n, mb):
            toks = self.eval.abc_tokens[i:i + mb]
            _, cache = self.model.run_with_cache(toks, names_filter=lambda x: x in self._hook_names)
            weight = toks.shape[0] / n
            chunk = {INPUT: cache["blocks.0.hook_resid_pre"]}
            for l in range(self.graph.n_layers):
                result = cache[f"blocks.{l}.attn.hook_result"]
                for h in range(self.graph.n_heads):
                    chunk[("attn", l, h)] = result[:, :, h, :]
                chunk[("mlp", l)] = cache[f"blocks.{l}.hook_mlp_out"]
            for k, v in chunk.items():
                contribution = v.mean(0, keepdim=True) * weight
                acc[k] = contribution if k not in acc else acc[k] + contribution
            del cache, chunk
        return acc

    # ------------------------------------------------------------------ hooks
    @staticmethod
    def _drop_heads(x, hook, layer, heads, abl):
        x = x.clone()
        for h in heads:
            x[:, :, h, :] = abl[("attn", layer, h)]
        return x

    @staticmethod
    def _drop_mlp(x, hook, layer, abl):
        return abl[("mlp", layer)].expand_as(x).clone()

    def _hooks(self, circuit: Circuit) -> list:
        hooks = []
        for l in range(self.graph.n_layers):
            drop = [h for h in range(self.graph.n_heads) if ("attn", l, h) not in circuit.nodes]
            if drop:
                hooks.append((f"blocks.{l}.attn.hook_result",
                              functools.partial(self._drop_heads, layer=l, heads=drop, abl=self._abl)))
            if ("mlp", l) not in circuit.nodes:
                hooks.append((f"blocks.{l}.hook_mlp_out",
                              functools.partial(self._drop_mlp, layer=l, abl=self._abl)))
        return hooks

    # ----------------------------------------------------------------- metric
    def _logit_diff(self, logits: torch.Tensor, sl: slice) -> torch.Tensor:
        last = logits[:, -1, :]
        idx = torch.arange(last.shape[0], device=last.device)
        return last[idx, self.eval.io[sl]] - last[idx, self.eval.s[sl]]

    @torch.no_grad()
    def evaluate(self, circuit: Circuit) -> float:
        """F(C): mean IOI logit difference with every component outside `circuit` ablated."""
        n, mb = len(self.eval), self.minibatch
        hooks, out = self._hooks(circuit), []
        for i in range(0, n, mb):
            sl = slice(i, min(i + mb, n))
            logits = self.model.run_with_hooks(self.eval.tokens[sl], fwd_hooks=hooks)
            out.append(self._logit_diff(logits, sl).float())
        return float(torch.cat(out).mean())

    @torch.no_grad()
    def clean(self) -> float:
        n, mb = len(self.eval), self.minibatch
        out = []
        for i in range(0, n, mb):
            sl = slice(i, min(i + mb, n))
            out.append(self._logit_diff(self.model(self.eval.tokens[sl]), sl).float())
        return float(torch.cat(out).mean())


class Faithfulness:
    """Normalises F(C) against the full and empty circuits of one model.

    `f_full` and `f_empty` are measured once per model and reused for every circuit, so the
    per-circuit cost is a single pass over the evaluation set.
    """

    def __init__(self, knockout: Knockout, graph: ModelGraph):
        self.knockout, self.graph = knockout, graph
        self.f_full = knockout.evaluate(graph.full_circuit())
        self.f_empty = knockout.evaluate(graph.empty_circuit())
        self.denom = self.f_full - self.f_empty

    def normalise(self, f: float) -> float:
        return (f - self.f_empty) / self.denom

    def score(self, circuit: Circuit) -> dict:
        """Raw and normalised faithfulness of one circuit, under both scopes."""
        out = {}
        for scope in SCOPES:
            f = self.knockout.evaluate(scope_circuit(circuit, scope, self.graph))
            out[f"F_{scope}"] = f
            out[f"faithfulness_{scope}"] = self.normalise(f)
        return out

    def sanity(self) -> dict:
        """The full circuit must reproduce the unhooked model; report the gap."""
        clean = self.knockout.clean()
        return {"F_clean": clean, "F_full": self.f_full, "F_empty": self.f_empty,
                "denominator": self.denom, "full_vs_clean_gap": abs(self.f_full - clean)}
