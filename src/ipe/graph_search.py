from transformer_lens import HookedTransformer
import torch
from ipe.nodes import Node, ATTN_Node
from ipe.paths import evaluate_path, get_path
from ipe.miscellanea import batch_iterable
from tqdm import tqdm
from typing import Callable
import gc
import heapq
import time
from loguru import logger

# --- Debug logging ----------------------------------------------------------
# TreeMessagePatching emits verbose debug records (per-candidate contribution,
# frontier, node being inspected, candidate, admit/reject) to a common log file
# so a run can be inspected offline. Call setup_tree_debug_log(path) before the
# search to (re)configure the sink; logging is a no-op until then.
TREE_DEBUG_LOG = "tree_search_debug.log"
_tree_debug_sink_id = None


def setup_tree_debug_log(path: str = TREE_DEBUG_LOG, level: str = "DEBUG") -> None:
	"""Route the tree-search debug records to `path`.

	Adds a dedicated loguru sink writing only the records tagged with
	`tree_debug=True` (see TreeMessagePatching). Safe to call repeatedly: the
	previous sink installed by this helper is removed first so the file is opened
	fresh (mode="w") for each new run.

	Args:
		path (str, default=TREE_DEBUG_LOG):
			Destination log file.
		level (str, default="DEBUG"):
			Minimum record level to write.
	"""
	global _tree_debug_sink_id
	if _tree_debug_sink_id is not None:
		try:
			logger.remove(_tree_debug_sink_id)
		except ValueError:
			pass
	_tree_debug_sink_id = logger.add(
		path,
		level=level,
		mode="w",
		enqueue=True,
		filter=lambda record: record["extra"].get("tree_debug", False),
		format="{time:HH:mm:ss.SSS} | {level: <7} | {message}",
	)
	logger.bind(tree_debug=True).info(f"Tree debug log initialised at {path}")


# Bound logger used by the tree search; records carry the tree_debug flag so the
# sink filter above picks them up without polluting the rest of the app's logs.
_tlog = logger.bind(tree_debug=True)

def find_relevant_positions(
		candidate: ATTN_Node,
		incomplete_path: list[Node],
		metric: Callable,
		min_contribution: float,
		include_negative: bool) -> list[tuple[torch.Tensor, list[Node]]]:
	"""Helper function to find relevant key-value positions for a candidate attention node.
	
	Args:
		candidate (ATTN_Node):
			The candidate attention node to evaluate.
		incomplete_path (list of Node):
			The current incomplete path to be extended.
		metric (Callable):
			A function to evaluate the contribution or importance of the path.
		min_contribution (float):
			The minimum absolute contribution score required for a path to be considered valid.
		include_negative (bool):
			If True, include paths with negative contributions.
	
	Returns:
		list of tuples: A list of tuples containing the contribution score and the corresponding extended path.
	"""
	relevant_extensions = []
	target_positions = []
	# assert candidate.keyvalue_position is None, f"Candidate keyvalue_position should be None when finding relevant positions! {candidate} - {incomplete_path}"
	assert incomplete_path[0].position is not None, f"First node in incomplete_path should have a defined position! {incomplete_path}"
	if incomplete_path[0].__class__.__name__ == 'ATTN_Node':
		if incomplete_path[0].patch_key or incomplete_path[0].patch_value:
			target_positions = [incomplete_path[0].keyvalue_position]
		if incomplete_path[0].patch_query and incomplete_path[0].position != incomplete_path[0].keyvalue_position:
			target_positions.append(incomplete_path[0].position)
	else:
		target_positions = [incomplete_path[0].position]
	assert len(target_positions) == 1, "More than one target position found in find_relevant_positions!"
	for target_position in target_positions:
		candidate.position = target_position
		if candidate.patch_key or candidate.patch_value:
			for kv_position in range(candidate.position + 1):
				candidate_pos = ATTN_Node(
					model=candidate.model,
					layer=candidate.layer,
					head=candidate.head,
					position=candidate.position,
					keyvalue_position=kv_position,
					parent=candidate.parent,
					children=set(),
					msg_cache=candidate.msg_cache,
					cf_cache=candidate.cf_cache,
					gradient=None,
					patch_query=candidate.patch_query,
					patch_key=candidate.patch_key,
					patch_value=candidate.patch_value,
					plot_patterns=False,
					patch_type=candidate.patch_type
				)
				contribution = evaluate_path([candidate_pos] + incomplete_path, metric)
				if (contribution >= min_contribution) or (include_negative and abs(contribution) >= min_contribution):
					relevant_extensions.append((contribution, [candidate_pos]+incomplete_path))
		elif candidate.patch_query:
			candidate_pos = ATTN_Node(
				model=candidate.model,
				layer=candidate.layer,
				head=candidate.head,
				position=target_position,
				keyvalue_position=None,
				parent=candidate.parent,
				children=set(),
				msg_cache=candidate.msg_cache,
				cf_cache=candidate.cf_cache,
				gradient=None,
				patch_query=candidate.patch_query,
				patch_key=candidate.patch_key,
				patch_value=candidate.patch_value,
				plot_patterns=False,
				patch_type=candidate.patch_type
			)
			contribution = evaluate_path([candidate_pos] + incomplete_path, metric)

			if (contribution >= min_contribution) or (include_negative and abs(contribution) >= min_contribution):
				relevant_extensions.append((contribution, [candidate_pos]+incomplete_path))
	assert len(relevant_extensions) == len(set([tuple(path) for _, path in relevant_extensions])), "Duplicate paths found in find_relevant_positions!"
	return relevant_extensions



def find_relevant_heads(
		candidate: ATTN_Node,
		incomplete_path: list[Node],
		metric: Callable,
		min_contribution: float,
		include_negative: bool,
		batch_positions: bool) -> list[tuple[torch.Tensor, list[Node]]]:
	"""Helper function to find relevant heads for a candidate attention node.
	
	Args:
		candidate (ATTN_Node):
			The candidate attention node to evaluate.
		incomplete_path (list of Node):
			The current incomplete path to be extended.
		metric (Callable):
			A function to evaluate the contribution or importance of the path.
		min_contribution (float):
			The minimum absolute contribution score required for a path to be considered valid.
		include_negative (bool):
			If True, include paths with negative contributions.
		batch_positions (bool):
			If True, when expanding nodes, first evaluates attentions without considering position-wise contributions, only later, if the attention has been deemed meaningful, it will be evaluated at all possible key-value positions.
	
	Returns:
		list of tuples: A list of tuples containing the contribution score and the corresponding extended path.
	"""
	relevant_extensions = []
	for head in range(candidate.model.cfg.n_heads):
		candidate_head = ATTN_Node(
			model=candidate.model,
			layer=candidate.layer,
			head=head,
			position=candidate.position,
			keyvalue_position=candidate.keyvalue_position,
			parent=candidate.parent,
			children=set(),
			msg_cache=candidate.msg_cache,
			cf_cache=candidate.cf_cache,
			gradient=None,
			patch_query=candidate.patch_query,
			patch_key=candidate.patch_key,
			patch_value=candidate.patch_value,
			plot_patterns=False,
			patch_type=candidate.patch_type
		)
		contribution = evaluate_path([candidate_head]+incomplete_path, metric)
		if (contribution >= min_contribution) or (include_negative and abs(contribution) >= min_contribution):
			if batch_positions:
				relevant_extensions.extend(find_relevant_positions(candidate_head, incomplete_path, metric, min_contribution, include_negative))
			else:
				relevant_extensions.append((contribution, [candidate_head] + incomplete_path))
	return relevant_extensions



def TreeMessagePatching(
	model: HookedTransformer,
	metric: Callable,
	root: Node,
	min_contribution: float = 0.5,
	include_negative: bool = True,
) -> Node:
	"""
	Performs a Breadth-First Search (BFS) backwards from a node, growing a single
	tree rooted at it. Each candidate is scored by its own isolated contribution
	along the branch it would create: score(C) = evaluate_path([C, leaf, ..., root]),
	i.e. the message C sends up the leaf->root chain, removed from the clean
	residual and passed to the metric. This is exactly the per-path contribution
	PathMessagePatching uses; the tree is the same set of paths materialised as a
	trie sharing common suffixes toward the root. No joint-ablation baseline is
	involved, so a candidate's score depends only on C and the chain above it, not
	on any sibling subtree.

	Args:
		model (HookedTransformer):
			The transformer model used for evaluation.
		metric (Callable):
			A function to evaluate the contribution or importance of the path. It must accept a single parameter: `corrupted_resid`.
		root (Node):
			The initial node to begin the backward search from (e.g. FINAL_Node(layer=model.cfg.n_layers - 1, position=target_pos)). Its children are populated in place.
		min_contribution (float, default=0.5):
			The minimum absolute contribution required for a candidate to be attached to the tree.
		include_negative (bool, default=True):
			If True, include candidates with negative contributions. The min_contribution is therefore interpreted as a threshold on the magnitude of the contribution.

	Returns:
		Node:
			The root of the discovered tree, with its children populated.
	"""
	with torch.no_grad():
		_tlog.info(
			f"=== TreeMessagePatching start === root={root!r} "
			f"min_contribution={min_contribution} include_negative={include_negative}"
		)
		frontier = [root]
		depth = 0
		while frontier:
			# Each candidate is scored in isolation (its own branch contribution),
			# so survivors can be attached immediately without affecting any other
			# leaf's scores.
			_tlog.debug(f"--- depth {depth} --- frontier_size={len(frontier)}")
			_tlog.debug(f"depth {depth} frontier: {[repr(n) for n in frontier]}")
			next_frontier = []
			level_scored = level_admitted = 0
			for leaf_idx, leaf in enumerate(tqdm(frontier)):
				# [leaf, parent, ..., root] reused as the suffix for every candidate.
				suffix = get_path(leaf)
				survivors = []
				candidates = leaf.get_expansion_candidates(model.cfg, include_head=True)
				_tlog.debug(
					f"[d{depth} leaf {leaf_idx + 1}/{len(frontier)}] inspecting {leaf!r} "
					f"-> {len(candidates)} candidates"
				)
				for candidate in candidates:
					# Isolated contribution of the branch C -> leaf -> ... -> root.
					score = evaluate_path([candidate] + suffix, metric)
					admitted = (score >= min_contribution) or (include_negative and abs(score) >= min_contribution)
					level_scored += 1
					_tlog.debug(
						f"[d{depth} leaf {leaf_idx + 1}/{len(frontier)}] candidate={candidate!r} "
						f"contribution={float(score):+.6f} |contribution|={abs(float(score)):.6f} "
						f"thr={min_contribution} -> {'ADMIT' if admitted else 'reject'}"
					)
					if admitted:
						survivors.append(candidate)
						leaf.children.add(candidate)
						if candidate.__class__.__name__ != 'EMBED_Node':
							next_frontier.append(candidate)
						level_admitted += 1
				_tlog.debug(
					f"[d{depth} leaf {leaf_idx + 1}/{len(frontier)}] {leaf!r} survivors="
					f"{len(survivors)}: {[repr(c) for c in survivors]}"
				)

			_tlog.info(
				f"=== depth {depth} done === scored={level_scored} admitted={level_admitted} "
				f"next_frontier_size={len(next_frontier)}"
			)
			frontier = next_frontier
			depth += 1
		_tlog.info(f"=== TreeMessagePatching end === depth_reached={depth}")
	return root

def TreeMessagePatching_LimitedLevelWidth(
	model: HookedTransformer,
	metric: Callable,
	root: Node,
	max_width: int = 20000,
	include_negative: bool = True,
) -> Node:
	"""Beam variant of TreeMessagePatching: instead of admitting every candidate
	above a threshold, score all candidate extensions of all leaves at a depth and
	keep only the top `max_width` by |contribution|. This is the tree analog of
	PathMessagePatching_LimitedLevelWidth. Candidates are still scored by their own
	isolated branch contribution (evaluate_path([C, leaf, ..., root])), so pruning
	across leaves is well-defined and the tree remains a trie of the same paths.

	Args:
		model (HookedTransformer):
			The transformer model used for evaluation.
		metric (Callable):
			A function to evaluate the contribution or importance of the path.
		root (Node):
			The initial node to begin the backward search from. Its children are populated in place.
		max_width (int, default=20000):
			The maximum number of candidates to retain at each depth of the tree.
		include_negative (bool, default=True):
			If True, rank candidates by the magnitude of their contribution and keep
			negatively-contributing ones; otherwise only positive contributions are kept.

	Returns:
		Node:
			The root of the discovered tree, with its children populated.
	"""
	with torch.no_grad():
		_tlog.info(
			f"=== TreeMessagePatching_LimitedLevelWidth start === root={root!r} "
			f"max_width={max_width} include_negative={include_negative}"
		)
		frontier = [root]
		depth = 0
		while frontier:
			_tlog.debug(f"--- depth {depth} --- frontier_size={len(frontier)}")
			# Score every candidate extension of every leaf at this depth, then keep
			# only the global top-`max_width` by |contribution|.
			scored = []  # (rank_value, leaf, candidate)
			for leaf in tqdm(frontier):
				suffix = get_path(leaf)
				for candidate in leaf.get_expansion_candidates(model.cfg, include_head=True):
					score = evaluate_path([candidate] + suffix, metric)
					if include_negative or score >= 0:
						rank_value = abs(score.item()) if include_negative else score.item()
						scored.append((rank_value, leaf, candidate))
			if not scored:
				break

			survivors = heapq.nlargest(max_width, scored, key=lambda x: x[0])
			next_frontier = []
			for _, leaf, candidate in survivors:
				leaf.children.add(candidate)
				if candidate.__class__.__name__ != 'EMBED_Node':
					next_frontier.append(candidate)
			_tlog.info(
				f"=== depth {depth} done === scored={len(scored)} admitted={len(survivors)} "
				f"next_frontier_size={len(next_frontier)}"
			)
			frontier = next_frontier
			depth += 1
		_tlog.info(f"=== TreeMessagePatching_LimitedLevelWidth end === depth_reached={depth}")
	return root

def PathMessagePatching(
	model: HookedTransformer,
	metric: Callable,
	root: Node,
	min_contribution: float = 0.5,
	include_negative: bool = True,
	return_all: bool = False,
	batch_positions: bool = False,
	batch_heads: bool = False
) -> list[tuple[torch.Tensor, list[Node]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_Node.

	Args:
		model (HookedTransformer): 
			The transformer model used for evaluation. It should be an instance
			of HookedTransformer, to ensure compatibility with cache and nodes forward methods.
		metric (Callable): 
			A function to evaluate the contribution or importance of the path. It must accept a single parameter: `corrupted_resid`.
		root (Node): 
			The initial node to begin the backward search from (e.g., FINAL_Node(layer=model.cfg.n_layers - 1, position=target_pos)).
		min_contribution (float, default=0.5):
			The minimum absolute contribution score required for a path to be considered valid.
		include_negative (bool, default=False): 
			If True, include paths with negative contributions. The min_contribution is therefore interpreted as a threshold on the magnitude of the contribution.
		return_all (bool, default=False): 
			If True, return all evaluated complete paths regardless of their contribution score. The search will still be guided by min_contribution.
		batch_positions (bool, default=False): 
			If True, when expanding nodes, first evaluates attentions without considering position-wise contributions, only later, if the attention has been deemed meaningful, it will be evaluated at all possible key-value positions.
		batch_heads (bool, default=False): 
			If True, when expanding nodes, first evaluates attentions without considering all heads at once, only later, if the attention as a whole has been deemed meaningful, it will evaluate all single heads.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, 
		sorted by contribution in descending order.
	"""
	with torch.no_grad():
		if root.position is None:
			print("Warning: Starting node has no position defined. Batch positions will not be used.")
			batch_positions = False

		last_node_contribution = evaluate_path([root], metric)
		frontier = [(last_node_contribution, [root])]
		completed_paths = []
		while frontier:
			# Cur depth frontier contains a list of all the path continuations found in the current depth
			# So all these paths have 1 more node than the paths in the frontier
			cur_depth_frontier = []
			# For each incomplete path in the frontier, find all valuable continuations
			for _, incomplete_path in tqdm(frontier):
				
				cur_path_start = incomplete_path[0]
				cur_path_continuations = []

				# Use a proxy compenent where heads and positions are not yet defined (declare a component of the same class)
				if batch_positions:
					backup_position = cur_path_start.position
					target_position = cur_path_start.position
					if cur_path_start.__class__.__name__ == 'ATTN_Node' and (cur_path_start.patch_key or cur_path_start.patch_value):
						target_position = cur_path_start.keyvalue_position
						backup_kv_position = cur_path_start.keyvalue_position
						cur_path_start.keyvalue_position = None
					cur_path_start.position = None
				
				candidate_components = cur_path_start.get_expansion_candidates(model.cfg, include_head=not batch_heads)
				if batch_positions:
					cur_path_start.position = backup_position
					if cur_path_start.__class__.__name__ == 'ATTN_Node' and (cur_path_start.patch_key or cur_path_start.patch_value):
						cur_path_start.keyvalue_position = backup_kv_position
				# Get the meaningful candidates for expansion
				for candidate in candidate_components:
					# EMBED is the base case, the path is complete and after evaluation can be added to the completed paths
					if candidate.__class__.__name__ == 'EMBED_Node':
						candidate.position = target_position if batch_positions else candidate.position
						
						contribution = evaluate_path([candidate] + incomplete_path, metric)
						if return_all:
							completed_paths.append((contribution, [candidate] + incomplete_path))
						elif (contribution >= min_contribution) or (include_negative and abs(contribution) >= min_contribution):
							completed_paths.append((contribution, [candidate] + incomplete_path))
					
					# ATTNs and MLPs are possible expansions of the current path to be added to the frontier
					elif candidate.__class__.__name__ == 'MLP_Node':
						candidate.position = target_position if batch_positions else candidate.position
						contribution = evaluate_path([candidate] + incomplete_path, metric)
						if include_negative:
							if abs(contribution) >= min_contribution:
								cur_path_continuations.append((contribution, [candidate] + incomplete_path))
						elif contribution >= min_contribution:
							cur_path_continuations.append((contribution, [candidate] + incomplete_path))
					elif candidate.__class__.__name__ == 'ATTN_Node':
						contribution = evaluate_path([candidate] + incomplete_path, metric)
						if (contribution >= min_contribution) or (include_negative and abs(contribution) >= min_contribution):
							if batch_heads:
								cur_path_continuations.extend(find_relevant_heads(candidate, incomplete_path, metric, min_contribution, include_negative, batch_positions))
							elif batch_positions:
								cur_path_continuations.extend(find_relevant_positions(candidate, incomplete_path, metric, min_contribution, include_negative))
							else:
								cur_path_continuations.append((contribution, [candidate] + incomplete_path))
				cur_depth_frontier.extend(cur_path_continuations)
			# Sort the frontier just for visualization purposes
			frontier = sorted(cur_depth_frontier, key=lambda x: x[0], reverse=True)
	return sorted(completed_paths, key=lambda x: x[0], reverse=True)

def PathMessagePatching_BestFirstSearch(
	model: HookedTransformer,
	metric: Callable,
	root: Node,
	top_n: int = 100,
	max_time: int = 300,
	include_negative: bool = True,
	batch_positions: bool = False,
	batch_heads: bool = False
) -> list[tuple[torch.Tensor, list[Node]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_Node.

	Args:
		model (HookedTransformer): 
			The transformer model used for evaluation. It should be an instance
			of HookedTransformer, to ensure compatibility with cache and nodes forward methods.
		metric (Callable): 
			A function to evaluate the contribution or importance of the path. It must accept a single parameter: `corrupted_resid`.
		root (Node): 
			The initial node to begin the backward search from (e.g., FINAL_Node(layer=model.cfg.n_layers - 1, position=target_pos)).
		top_n (int, default=100):
			The number of paths to return.
		max_time (int, default=300):
			The maximum time (in seconds) to run the search.
		include_negative (bool, default=False): 
			If True, include paths with negative contributions. The min_contribution is therefore interpreted as a threshold on the magnitude of the contribution.
		return_all (bool, default=False): 
			If True, return all evaluated complete paths regardless of their contribution score. The search will still be guided by min_contribution.
		batch_positions (bool, default=False): 
			If True, when expanding nodes, first evaluates attentions without considering position-wise contributions, only later, if the attention has been deemed meaningful, it will be evaluated at all possible key-value positions.
		batch_heads (bool, default=False): 
			If True, when expanding nodes, first evaluates attentions without considering all heads at once, only later, if the attention as a whole has been deemed meaningful, it will evaluate all single heads.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, 
		sorted by contribution in descending order.
	"""
	with torch.no_grad():
		if root.position is None:
			print("Warning: Starting node has no position defined. Batch positions will not be used.")
			batch_positions = False

		frontier = [(0, [root])]
		completed_paths = []
		start_time = time.time()
		pbar = tqdm(total=top_n, desc="Completed paths")
		while frontier and (len(completed_paths) < top_n) and (time.time() - start_time < max_time):
			# ensure the bar reflects current number of completed paths
			pbar.n = min(len(completed_paths), top_n)
			pbar.refresh()
			
			_, best_incomplete_path = heapq.heappop(frontier)
			cur_path_start = best_incomplete_path[0]

			if cur_path_start.__class__.__name__ == 'ATTN_Node':
				expansions = []
				flag = False
				if batch_heads and cur_path_start.head is None:
					expansions = find_relevant_heads(cur_path_start, best_incomplete_path[1:], metric, 0, include_negative, batch_positions)
					flag = True
				elif batch_positions and cur_path_start.keyvalue_position and (cur_path_start.patch_key or cur_path_start.patch_value) is None:
					expansions = find_relevant_positions(cur_path_start, best_incomplete_path[1:], metric, 0, include_negative)
					flag = True
				for expansion in expansions:
					if include_negative:
						heapq.heappush(frontier, (-abs(expansion[0].item()), expansion[1]))
					else:
						heapq.heappush(frontier, (-expansion[0].item(), expansion[1]))
				if flag:
					continue
			elif cur_path_start.__class__.__name__ == 'EMBED_Node':
				contribution = evaluate_path(best_incomplete_path, metric)
				if include_negative or contribution > 0:
					completed_paths.append((contribution, best_incomplete_path))
				continue

			if batch_positions:
				backup_position = cur_path_start.position
				target_position = cur_path_start.position
				if cur_path_start.__class__.__name__ == 'ATTN_Node' and (cur_path_start.patch_key or cur_path_start.patch_value):
					target_position = cur_path_start.keyvalue_position
					backup_kv_position = cur_path_start.keyvalue_position
					cur_path_start.keyvalue_position = None
				cur_path_start.position = None
			
			candidate_components = cur_path_start.get_expansion_candidates(model.cfg, include_head=not batch_heads)

			if batch_positions:
				cur_path_start.position = backup_position
				if cur_path_start.__class__.__name__ == 'ATTN_Node' and (cur_path_start.patch_key or cur_path_start.patch_value):
					cur_path_start.keyvalue_position = backup_kv_position
			
			for candidate in candidate_components:			
				candidate.position = target_position if batch_positions else candidate.position
				contribution = evaluate_path([candidate] + best_incomplete_path, metric)
				if include_negative:
					heapq.heappush(frontier, (-abs(contribution.item()), [candidate] + best_incomplete_path))
				else:
					heapq.heappush(frontier, (-contribution.item(), [candidate] + best_incomplete_path))
		pbar.n = min(len(completed_paths), top_n)
		pbar.refresh()
		pbar.close()
	return sorted(completed_paths, key=lambda x: x[0], reverse=True)


def PathMessagePatching_LimitedLevelWidth(
	model: HookedTransformer,
	metric: Callable,
	root: Node,
	max_width: int = 20000,
	include_negative: bool = True,
	batch_positions: bool = False,
	batch_heads: bool = False
) -> list[tuple[torch.Tensor, list[Node]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_Node.

	Args:
		model (HookedTransformer): 
			The transformer model used for evaluation.
		metric (Callable): 
			A function to evaluate the contribution or importance of the path.
		root (Node): 
			The initial node to begin the backward search from.
		max_width (int, default=20000):
			The maximum number of nodes to retain at each level of the search tree.
		include_negative (bool, default=False): 
			If True, include paths with negative contributions.
		batch_positions (bool, default=False): 
			If True, nodes are expanded without position-wise contributions, and only
			the top candidates are later expanded across all key-value positions.
		batch_heads (bool, default=False): 
			If True, attention contributions are evaluated for all heads at once, and
			only the top candidates are later expanded into single heads.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, 
		sorted by contribution in descending order.
	"""
	with torch.no_grad():
		if root.position is None and batch_positions:
			print("Warning: Starting node has no position defined. Batch positions will be disabled.")
			batch_positions = False

		frontier = [(1.0, [root])]
		completed_paths = []
		
		while frontier:
			current_depth_frontier = []
			
			for _, path in tqdm(frontier, desc=f"Expanding level (size {len(frontier)})"):
				cur_path_start = path[0]
				target_position = cur_path_start.position

				if batch_positions:
					assert cur_path_start.position is not None, f"Current path start must have a defined position when batch_positions is True! {path}"
					backup_position = cur_path_start.position
					if cur_path_start.__class__.__name__ == 'ATTN_Node' and (cur_path_start.patch_key or cur_path_start.patch_value):
						target_position = cur_path_start.keyvalue_position
						backup_kv_position = cur_path_start.keyvalue_position
						cur_path_start.keyvalue_position = None
					cur_path_start.position = None
				
				candidate_components = cur_path_start.get_expansion_candidates(model.cfg, include_head=not batch_heads)

				if batch_positions:
					cur_path_start.position = backup_position
					if cur_path_start.__class__.__name__ == 'ATTN_Node' and (cur_path_start.patch_key or cur_path_start.patch_value):
						cur_path_start.keyvalue_position = backup_kv_position
				assert cur_path_start.position is not None or not batch_positions, f"Current path start must have a defined position when batch_positions is True! {path}"
				for candidate in candidate_components:
					if candidate.__class__.__name__ == 'EMBED_Node':
						if batch_positions:
							candidate.position = target_position
						contribution = evaluate_path([candidate] + path, metric)
						if include_negative or contribution >= 0:
							completed_paths.append((contribution, [candidate] + path))

					elif candidate.__class__.__name__ == 'MLP_Node' or candidate.__class__.__name__ == 'ATTN_Node':
						# For batched search, position might be generic here
						if batch_positions:
							candidate.position = target_position
						
						contribution = evaluate_path([candidate] + path, metric)
						
						# Store the contribution magnitude for ranking
						contribution_val = abs(contribution.item()) if include_negative else contribution.item()

						if include_negative or contribution >= 0:
							current_depth_frontier.append((contribution_val, [candidate] + path))
			
			if not current_depth_frontier:
				break

			# First Pruning: Keep the top `max_width` general paths
			frontier = heapq.nlargest(max_width, current_depth_frontier, key=lambda x: x[0])
			
			# Second Step: If batching, expand the grouped nodes
			if batch_heads or batch_positions:
				new_frontier = []
				for _, path in frontier:
					# Check if this node is a generic ATTN node that needs expansion
					if path[0].__class__.__name__ == 'ATTN_Node':
						expansions = []
						if batch_heads and path[0].head is None:
							expansions = find_relevant_heads(path[0], path[1:], metric, 0, include_negative, batch_positions)
						elif batch_positions and path[0].position is None or (path[0].keyvalue_position is None and (path[0].patch_key or path[0].patch_value)):
							expansions = find_relevant_positions(path[0], path[1:], metric, 0, include_negative)
						else: # Already non batched ATTN node, keep as is
							new_frontier.append((abs(evaluate_path(path, metric).item()), path))

						if expansions:
							# Convert tensor contributions to floats for ranking
							for contrib, expanded_path in expansions:
								val = abs(contrib.item()) if include_negative else contrib.item()
								new_frontier.append((val, expanded_path))
					else:
						# Not an expandable ATTN node, keep it as is.
						new_frontier.append((abs(evaluate_path(path, metric).item()), path))

				# Second Pruning: Keep the top `max_width` of the newly expanded, specific paths
				frontier = heapq.nlargest(max_width, new_frontier, key=lambda x: x[0])

	return sorted(completed_paths, key=lambda x: x[0], reverse=True)


def PathAttributionPatching(
	model: HookedTransformer,
	metric: Callable,
	root: Node,
	min_contribution: float = 0.5,
	include_negative: bool = True,
	return_all: bool = False,
	confirm_relevance: bool = False
) -> list[tuple[torch.Tensor, list[Node]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_Node.

	Args:
		model (HookedTransformer): 
			The transformer model used for evaluation.
		msg_cache (ActivationCache): 
			The activation cache containing intermediate activations.
		metric (Callable):
			A function to evaluate the contribution or importance of the path.
				It must accept a single parameter corresponding to the corrupted residual stream just before the final layer norm.
		root (Node): 
			The initial node to begin the backward search from (e.g., FINAL_Node(layer=model.cfg.n_layers - 1, position=target_pos)).
		ground_truth_tokens (list of int): 
			The reference tokens used for evaluating path contributions.
		min_contribution (float, default=0.5): 
			The minimum absolute contribution score required for a path to be considered valid.
		include_negative (bool, default=False): 
			If True, include paths with negative contributions. The min_contribution is therefore interpreted as a threshold on the magnitude of the contribution.
		return_all (bool, default=False): 
			If True, return all evaluated paths regardless of their contribution score. The search will still be guided by min_contribution threshold.
		confirm_relevance (bool, default=False):
			If True, after identifying a potentially relevant component based on the linear approximation, it will also evaluate the contribution of the full path including to confirm its relevance.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, sorted by contribution in descending order.
	"""
	frontier = [root]
	completed_paths = []
	while frontier:
		cur_depth_frontier = []
		# Expand all paths in the frontier looking for meaningful continuations
		for node in tqdm(frontier):

			grad = node.calculate_gradient(use_precomputed=True)
			with torch.no_grad():
				childrens = []

				candidate_components = node.get_expansion_candidates(model.cfg, include_head=True) 

				# Get the meaningful candidates for expansion
				for candidate_batch in batch_iterable(candidate_components, 128):
					msgs_list = []
					for candidate in candidate_batch:
						backup_pos = candidate.position
						candidate.position = None
						msg = candidate.forward(message=None)
						msgs_list.append(msg)
						candidate.position = backup_pos
					candidate_contributions = torch.stack(msgs_list, dim=0)

					approx_contributions = torch.einsum('xbsd,bsd->x', candidate_contributions, grad)
					for i, candidate in enumerate(candidate_batch):
						approx_contribution = approx_contributions[i]
						# EMBED is the base case
						if candidate.__class__.__name__ == 'EMBED_Node':
							candidate_path = get_path(candidate)
							contribution = evaluate_path(candidate_path, metric)
							if return_all:
								completed_paths.append((contribution, candidate_path))
							elif include_negative:
								if abs(contribution.item()) >= min_contribution:
									if confirm_relevance:
										if abs(contribution) >= min_contribution:
											completed_paths.append((contribution, candidate_path))
									else:
										completed_paths.append((contribution, candidate_path))
							elif contribution >= min_contribution:
								if confirm_relevance:
									if contribution >= min_contribution:
										completed_paths.append((contribution, candidate_path))
								else:
									completed_paths.append((contribution, candidate_path))
								
						
						# MLP requires to check the contribution of the whole component and of the individual layers
						elif candidate.__class__.__name__ == 'MLP_Node' or candidate.__class__.__name__ == 'ATTN_Node':
							if include_negative:
								if abs(approx_contribution.item()) >= min_contribution:
									if confirm_relevance:
										contribution = evaluate_path(get_path(candidate), metric)
										if abs(contribution) >= min_contribution:
											childrens.append(candidate)
									else:
										childrens.append(candidate)
							elif approx_contribution >= min_contribution:
								if confirm_relevance:
									contribution = evaluate_path(get_path(candidate), metric)
									if contribution >= min_contribution:
										childrens.append(candidate)
								else:
									childrens.append(candidate)
				cur_depth_frontier.extend(childrens)
				node.children = childrens
				if len(childrens) == 0:
					node.gradient = None # Free the gradient of the node if it has no children to save memory
		
		for node in frontier: # Free the gradient of the parent nodes to save memory
			if node.parent is not None and node.parent.gradient is not None:
				node.parent.gradient = None
		gc.collect() # Reclaim memory
		torch.cuda.empty_cache()

		frontier = cur_depth_frontier

	return sorted(completed_paths, key=lambda x: x[0], reverse=True)


def PathAttributionPatching_BestFirstSearch(
	model: HookedTransformer,
	metric: Callable,
	root: Node,
	include_negative: bool = True,
	top_n: int = 100,
	max_time: int = 300,
) -> list[tuple[torch.Tensor, list[Node]]]:
	"""
	Performs a Best First Search starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_Node.

	Args:
		model (HookedTransformer): 
			The transformer model used for evaluation.
		msg_cache (ActivationCache): 
			The activation cache containing intermediate activations.
		metric (Callable):
			A function to evaluate the contribution or importance of the path.
				It must accept a single parameter corresponding to the corrupted residual stream just before the final layer norm.
		root (Node): 
			The initial node to begin the backward search from (e.g., FINAL_Node(layer=model.cfg.n_layers - 1, position=target_pos)).
		include_negative (bool, default=False): 
			If True, include paths with negative contributions, otherwise only return positively contributing paths. 
			Note that to save computation the negatively contributing paths are discarded even if incomplete.
		top_n (int, default=100):
			The number of paths to return.
		max_time (int, default=300):
			The maximum time (in seconds) to run the search.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, sorted by contribution in descending order.
	"""

	frontier = []
	heapq.heappush(frontier, (0, root))

	completed_paths = []
	start_time = time.time()

	# Best-first loop: pop highest-priority element, expand it, push children with priority
	pbar = tqdm(total=top_n, desc="Completed paths")
	while frontier and (time.time() - start_time) < max_time and (len(completed_paths) < top_n):
		# ensure the bar reflects current number of completed paths
		pbar.n = min(len(completed_paths), top_n)
		pbar.refresh()
		_, node = heapq.heappop(frontier)

		if node.__class__.__name__ == 'EMBED_Node':
			candidate_path = get_path(node)
			contribution = evaluate_path(candidate_path, metric)
			if include_negative or contribution > 0:
				completed_paths.append((contribution, candidate_path))
			continue

		grad = node.calculate_gradient(use_precomputed=True, save=False) # Initially do not save gradient to save memory, however increase the computation time
		with torch.no_grad():
			candidate_components = node.get_expansion_candidates(model.cfg, include_head=True) 

			# Get the meaningful candidates for expansion
			for candidate_batch in batch_iterable(candidate_components, 128):
				msgs_list = []
				for candidate in candidate_batch:
					backup_pos = candidate.position
					candidate.position = None
					msg = candidate.forward(message=None)
					msgs_list.append(msg)
					candidate.position = backup_pos
				candidate_contributions = torch.stack(msgs_list, dim=0)

				approx_contributions = torch.einsum('xbsd,bsd->x', candidate_contributions, grad)
				approx_contributions = approx_contributions.detach().cpu().numpy()
				for i, candidate in enumerate(candidate_batch):
					approx_contribution = approx_contributions[i]
					if include_negative or approx_contribution > 0:
						approx_contribution = -abs(approx_contribution)
						heapq.heappush(frontier, (approx_contribution, candidate))
	pbar.n = min(len(completed_paths), top_n)
	pbar.refresh()
	pbar.close()
	return sorted(completed_paths, key=lambda x: x[0], reverse=True)


def PathAttributionPatching_LimitedLevelWidth(
	model: HookedTransformer,
	metric: Callable,
	root: Node,
	max_width: int = 2000,
	include_negative: bool = True,
) -> list[tuple[torch.Tensor, list[Node]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify the most significant paths reaching it from an EMBED_Node. 
	At each level of the search tree, only the top `max_width` nodes (based on their approximate contribution) are retained.

	Args:
		model (HookedTransformer): 
			The transformer model used for evaluation.
		msg_cache (ActivationCache): 
			The activation cache containing intermediate activations.
		metric (Callable):
			A function to evaluate the contribution or importance of the path.
				It must accept a single parameter corresponding to the corrupted residual stream 
				just before the final layer norm.
		root (Node): 
			The initial node to begin the backward search from (e.g., FINAL_Node(layer=model.cfg.n_layers - 1, position=target_pos)).
		include_negative (bool, default=False): 
			If True, include paths with negative contributions, otherwise only return positively contributing paths. 
			Note that to save computation the negatively contributing paths are discarded even if incomplete.
		max_width (int, default=20000):
			The maximum number of nodes to retain at each level of the search tree.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, sorted by contribution in descending order.
	"""

	frontier = [(0, root)]
	completed_paths = []
	previous_level_nodes = []
	while frontier:
		cur_depth_frontier = []
		# Expand all paths in the frontier looking for meaningful continuations
		for _, node in tqdm(frontier):

			grad = node.calculate_gradient(use_precomputed=True)

			with torch.no_grad():
				candidate_components = node.get_expansion_candidates(model.cfg, include_head=True) 

				# Get the meaningful candidates for expansion
				for candidate_batch in batch_iterable(candidate_components, 128):
					msgs_list = []
					for candidate in candidate_batch:
						backup_pos = candidate.position
						candidate.position = None
						msg = candidate.forward(message=None)
						msgs_list.append(msg)
						candidate.position = backup_pos
					candidate_contributions = torch.stack(msgs_list, dim=0)

					approx_contributions = torch.einsum('xbsd,bsd->x', candidate_contributions, grad)
					for i, candidate in enumerate(candidate_batch):
						approx_contribution = approx_contributions[i]
						# EMBED is the base case
						if candidate.__class__.__name__ == 'EMBED_Node':
							candidate_path = get_path(candidate)
							contribution = evaluate_path(candidate_path, metric)
							completed_paths.append((contribution, candidate_path))
						
						# MLP requires to check the contribution of the whole component and of the individual layers
						elif candidate.__class__.__name__ == 'MLP_Node' or candidate.__class__.__name__ == 'ATTN_Node':
							if include_negative:
								cur_depth_frontier.append((abs(approx_contribution.item()), candidate))							
							else:
								cur_depth_frontier.append((approx_contribution.item(), candidate))
		cur_depth_frontier = heapq.nlargest(max_width, cur_depth_frontier, key=lambda x: x[0])

		for _, node in previous_level_nodes:
			node.gradient = None # Free the gradient of the node if it has no children to save memory
		gc.collect() # Reclaim memory
		torch.cuda.empty_cache()

		previous_level_nodes = frontier.copy()
		frontier = cur_depth_frontier

	return sorted(completed_paths, key=lambda x: x[0], reverse=True)
