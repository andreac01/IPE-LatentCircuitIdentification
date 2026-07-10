"""JSON serialization of discovered circuits for the web visualization.

The searches produce either a tree of Nodes (TreeMessagePatching*) or a list of
scored root->leaf paths (PathMessagePatching*). For the grid view (layers on the
y-axis, token positions on the x-axis) both are collapsed into the same DAG:
every tree node / path element that shares a grid identity (component + token
position, see `grid_key`) becomes one grid node, and parent->child links are
deduplicated into edges carrying a multiplicity. This mirrors the ASCII
`render_position_grid` in experiments/tree_vs_path.py.

Grid identity deliberately ignores the key/value-position and the patch flags:
a head repeated across branches (reading different positions, patched on Q vs
KV, ...) collapses to a single grid node, with its reads expressed by the edges
and the merged variants listed in the node's metadata.
"""

from collections import Counter

from ipe.nodes import Node, EMBED_Node, MLP_Node, ATTN_Node, FINAL_Node


def grid_key(n: Node) -> tuple:
	"""Identity of a node as placed on the grid: component + token position."""
	if isinstance(n, FINAL_Node):
		return ("final", n.position)
	if isinstance(n, EMBED_Node):
		return ("embed", n.position)
	if isinstance(n, MLP_Node):
		return ("mlp", n.layer, n.position)
	if isinstance(n, ATTN_Node):
		return ("attn", n.layer, n.head, n.position)
	return ("unknown", id(n))


def grid_id(n: Node) -> str:
	"""Stable string form of `grid_key`, usable as a DOM/JSON identifier."""
	return ":".join("*" if p is None else str(p) for p in grid_key(n))


def attn_variant(n: ATTN_Node) -> str:
	"""Which stream(s) of the head this tree node patches: 'Q', 'K', 'V', 'KV', ..."""
	v = ""
	if n.patch_query:
		v += "Q"
	if n.patch_key:
		v += "K"
	if n.patch_value:
		v += "V"
	return v or "?"


def node_to_dict(n: Node) -> dict:
	"""Public JSON form of a single tree node (used for live 'admit' events)."""
	kind = grid_key(n)[0]
	d = {
		"id": grid_id(n),
		"kind": kind,
		"layer": getattr(n, "layer", None) if not isinstance(n, EMBED_Node) else None,
		"position": n.position,
		"contribution": getattr(n, "contribution", None),
	}
	if isinstance(n, ATTN_Node):
		d["head"] = n.head
		d["kv_position"] = n.keyvalue_position
		d["variant"] = attn_variant(n)
	return d


def _subtree_reaches_embed(n: Node, memo: dict) -> bool:
	"""True if some root->leaf branch through `n` ends at an EMBED node, i.e. the
	branch is complete rather than pruned mid-way."""
	if id(n) in memo:
		return memo[id(n)]
	if isinstance(n, EMBED_Node):
		memo[id(n)] = True
		return True
	res = any(_subtree_reaches_embed(c, memo) for c in n.children)
	memo[id(n)] = res
	return res


class _GraphAccumulator:
	"""Accumulates (parent, child) links of tree nodes / path elements into the
	collapsed grid DAG, merging by grid identity."""

	def __init__(self):
		self.nodes: dict[str, dict] = {}
		self.edges: Counter = Counter()
		self.edge_best: dict[tuple[str, str], float] = {}
		self.edge_complete: set[tuple[str, str]] = set()

	def add_node(self, n: Node, complete: bool = True) -> str:
		nid = grid_id(n)
		contribution = getattr(n, "contribution", None)
		if nid not in self.nodes:
			d = node_to_dict(n)
			d.update({"merged": 0, "complete": False, "variants": [], "kv_positions": []})
			d.pop("kv_position", None)
			d.pop("variant", None)
			self.nodes[nid] = d
		d = self.nodes[nid]
		d["merged"] += 1
		d["complete"] = d["complete"] or complete
		if contribution is not None:
			# Keep the signed contribution of the strongest merged branch.
			if d["contribution"] is None or abs(contribution) > abs(d["contribution"]):
				d["contribution"] = contribution
		if isinstance(n, ATTN_Node):
			v = attn_variant(n)
			if v not in d["variants"]:
				d["variants"].append(v)
			if n.keyvalue_position is not None and n.keyvalue_position not in d["kv_positions"]:
				d["kv_positions"].append(n.keyvalue_position)
		return nid

	def add_edge(self, child: Node, parent: Node, complete: bool = True) -> None:
		# Data-flow orientation: the child feeds its message into the parent.
		src, dst = grid_id(child), grid_id(parent)
		if src == dst:
			return
		self.edges[(src, dst)] += 1
		if complete:
			# The edge lies on at least one branch that reaches the embeddings.
			self.edge_complete.add((src, dst))
		c = getattr(child, "contribution", None)
		if c is not None:
			best = self.edge_best.get((src, dst))
			if best is None or abs(c) > abs(best):
				self.edge_best[(src, dst)] = c

	def to_dict(self) -> dict:
		return {
			"nodes": list(self.nodes.values()),
			"edges": [
				{"source": s, "target": t, "count": c,
				 "contribution": self.edge_best.get((s, t)),
				 "complete": (s, t) in self.edge_complete}
				for (s, t), c in self.edges.items()
			],
		}


def graph_from_tree(root: Node) -> dict:
	"""Collapse the tree grown by TreeMessagePatching* into the grid DAG.

	Nodes on branches that were pruned before reaching the embeddings are marked
	`complete: false` (unless they also lie on a complete branch) so the frontend
	can render them differently."""
	acc = _GraphAccumulator()
	memo: dict = {}

	def walk(n: Node) -> None:
		acc.add_node(n, complete=_subtree_reaches_embed(n, memo))
		for child in n.children:
			acc.add_edge(child, n, complete=_subtree_reaches_embed(child, memo))
			walk(child)

	walk(root)
	return acc.to_dict()


def graph_from_paths(paths: list[tuple[float, list[Node]]]) -> dict:
	"""Collapse PathMessagePatching* results (score, [leaf, ..., root]) into the
	grid DAG. Completed paths always reach the embeddings, so every node is
	complete; a node's contribution is the strongest path score through it."""
	acc = _GraphAccumulator()
	for score, path in paths:
		for n in path:
			# The leaf-side score is the whole path's contribution; per-node
			# admission scores were stored on the nodes by the search when
			# available and take precedence in add_node.
			if getattr(n, "contribution", None) is None:
				n.contribution = float(score)
			acc.add_node(n, complete=True)
		for child, parent in zip(path[:-1], path[1:]):
			acc.add_edge(child, parent)
	return acc.to_dict()
