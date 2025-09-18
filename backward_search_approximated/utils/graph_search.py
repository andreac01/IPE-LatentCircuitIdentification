from transformer_lens import HookedTransformer
import torch
from backward_search_approximated.utils.nodes import ApproxNode, ATTN_ApproxNode
from backward_search_approximated.utils.paths import evaluate_path, get_path
from backward_search_approximated.utils.miscellanea import batch_iterable
from tqdm import tqdm
from typing import Callable
import gc


def find_relevant_positions(candidate, incomplete_path, metric, min_contribution, include_negative):
	relevant_extensions = []
	target_positions = []
	if incomplete_path[0].__class__.__name__ == 'ATTN_ApproxNode':
		if incomplete_path[0].patch_key or incomplete_path[0].patch_value:
			target_positions = [incomplete_path[0].keyvalue_position]
		if incomplete_path[0].patch_query:
			target_positions.append(incomplete_path[0].position)
	else:
		target_positions = [incomplete_path[0].position]
	for target_position in target_positions:
		candidate.position = target_position
		if candidate.patch_key or candidate.patch_value:
			for kv_position in range(candidate.position):
				candidate_pos = ATTN_ApproxNode(
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
			candidate_pos = ATTN_ApproxNode(
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
	return relevant_extensions


def find_relevant_heads(candidate, incomplete_path, metric, min_contribution, include_negative, batch_positions):
	relevant_extensions = []
	for head in range(candidate.model.cfg.n_heads):
		candidate_head = ATTN_ApproxNode(
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



def IsolatingPathEffect_BW(
	model: HookedTransformer,
	metric: Callable,
	root: ApproxNode,
	min_contribution: float = 0.5,
	include_negative: bool = False,
	return_all: bool = False,
	batch_positions: bool = False,
	batch_heads: bool = False
) -> list[tuple[float, list[ApproxNode]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_ApproxNode.

	Args:
		model (HookedTransformer): 
			The transformer model used for evaluation. It should be an instance
			of HookedTransformer, to ensure compatibility with cache and nodes forward methods.
		metric (Callable): 
			A function to evaluate the contribution or importance of the path. It must accept a single parameter: `corrupted_resid`.
		root (ApproxNode): 
			The initial node to begin the backward search from (e.g., FINAL_ApproxNode(layer=model.cfg.n_layers - 1, position=target_pos)).
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
	if root.position is None:
		print("Warning: Starting node has no position defined. Batch positions will not be used.")
		batch_positions = False

	last_node_contribution = evaluate_path([root], metric)
	frontier = [(last_node_contribution, [root])]
	completed_paths = []
	while frontier:
		if len(frontier) > 2:
			print(f"(total {len(frontier)})    Frontier: {frontier[:2]}... ]")
		else:
			print(f"(total {len(frontier)})    Frontier: {frontier}")

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
				if cur_path_start.__class__.__name__ == 'ATTN_ApproxNode' and (cur_path_start.patch_key or cur_path_start.patch_value):
					target_position = cur_path_start.keyvalue_position
				cur_path_start.position = None
			
			candidate_components = cur_path_start.get_expansion_candidates(model.cfg, include_head=not batch_heads)

			if batch_positions:
				cur_path_start.position = backup_position
			# Get the meaningful candidates for expansion
			for candidate in candidate_components:
				# EMBED is the base case, the path is complete and after evaluation can be added to the completed paths
				if candidate.__class__.__name__ == 'EMBED_ApproxNode':
					candidate.position = target_position if batch_positions else candidate.position
					
					contribution = evaluate_path([candidate] + incomplete_path, metric)
					if return_all:
						completed_paths.append((contribution, [candidate] + incomplete_path))
					elif (contribution >= min_contribution) or (include_negative and abs(contribution) >= min_contribution):
						completed_paths.append((contribution, [candidate] + incomplete_path))
				
				# ATTNs and MLPs are possible expansions of the current path to be added to the frontier
				elif candidate.__class__.__name__ == 'MLP_ApproxNode':
					candidate.position = target_position if batch_positions else candidate.position
					contribution = evaluate_path([candidate] + incomplete_path, metric)
					if include_negative:
						if abs(contribution) >= min_contribution:
							cur_path_continuations.append((contribution, [candidate] + incomplete_path))
					elif contribution >= min_contribution:
						cur_path_continuations.append((contribution, [candidate] + incomplete_path))
				elif candidate.__class__.__name__ == 'ATTN_ApproxNode':
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

def PathAttributionPatching(
	model: HookedTransformer,
	metric: Callable,
	root: ApproxNode,
	min_contribution: float = 0.5,
	include_negative: bool = False,
	return_all: bool = False,
) -> list[tuple[float, list[ApproxNode]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_ApproxNode.

	Args:
		model (HookedTransformer): 
			The transformer model used for evaluation.
		msg_cache (ActivationCache): 
			The activation cache containing intermediate activations.
		metric (Callable):
			A function to evaluate the contribution or importance of the path.
				It must accept a single parameter corresponding to the corrupted residual stream just before the final layer norm.
		root (ApproxNode): 
			The initial node to begin the backward search from (e.g., FINAL_ApproxNode(layer=model.cfg.n_layers - 1, position=target_pos)).
		ground_truth_tokens (list of int): 
			The reference tokens used for evaluating path contributions.
		min_contribution (float, default=0.5): 
			The minimum absolute contribution score required for a path to be considered valid.
		include_negative (bool, default=False): 
			If True, include paths with negative contributions. The min_contribution is therefore interpreted as a threshold on the magnitude of the contribution.
		return_all (bool, default=False): 
			If True, return all evaluated paths regardless of their contribution score. The search will still be guided by min_contribution threshold.
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

			childrens = []

			candidate_components = node.get_expansion_candidates(model.cfg, include_head=True) 

			# Get the meaningful candidates for expansion
			for candidate_batch in batch_iterable(candidate_components, 100):
				candidate_contributions = torch.stack([candidate.forward(message=None) for candidate in candidate_batch], dim=0)

				approx_contributions = torch.einsum('xbsd,bsd->x', candidate_contributions, grad)
				for i, candidate in enumerate(candidate_batch):
					approx_contribution = approx_contributions[i]
					# EMBED is the base case
					if candidate.__class__.__name__ == 'EMBED_ApproxNode':
						candidate_path = get_path(candidate)
						if return_all:
							contribution = evaluate_path(candidate_path, metric)
							completed_paths.append((contribution, candidate_path))
						elif include_negative:
							if abs(approx_contribution.item()) >= min_contribution:
								contribution = evaluate_path(candidate_path, metric)
								completed_paths.append((contribution, candidate_path))
						elif approx_contribution >= min_contribution:
							contribution = evaluate_path(candidate_path, metric)
							completed_paths.append((contribution, candidate_path))
					
					# MLP requires to check the contribution of the whole component and of the individual layers
					elif candidate.__class__.__name__ == 'MLP_ApproxNode' or candidate.__class__.__name__ == 'ATTN_ApproxNode':
						if include_negative:
							if abs(approx_contribution.item()) >= min_contribution:
								childrens.append(candidate)
						elif approx_contribution >= min_contribution:
							childrens.append(candidate)
			cur_depth_frontier.extend(childrens)
			node.children = childrens
			if len(childrens) == 0:
				node.gradient = None # Free the gradient of the node if it has no children to save memory
		
		for node in frontier: # Free the gradient of the parent nodes to save memory
			if node.parent is not None:
				node.parent.gradient = None
		gc.collect() # Reclaim memory
		torch.cuda.empty_cache()

		frontier = cur_depth_frontier

	return sorted(completed_paths, key=lambda x: x[0], reverse=True)
