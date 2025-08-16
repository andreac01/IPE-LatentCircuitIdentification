from transformer_lens import HookedTransformer, ActivationCache
import torch
from torch import Tensor
from backward_search_approximated.utils.nodes import ApproxNode, FINAL_ApproxNode, MLP_ApproxNode, ATTN_ApproxNode, EMBED_ApproxNode
from tqdm import tqdm
from functools import partial
from typing import Callable



def evaluate_path(model, cache, path, metric, correct_tokens):
	message = None
	if len(path) == 0:
		return message

	for i in range(len(path)):
		message = path[i].forward(message=message)

	return metric(path[-1].forward(), path[-1].forward() - message, model, correct_tokens)


def breadth_first_search(
	model: HookedTransformer,
	cache: ActivationCache,
	metric: Callable,
	start_node: list[ApproxNode],
	ground_truth_tokens: list[int],
	min_contribution: float = 0.5,
) -> list[tuple[float, list[ApproxNode]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_ApproxNode.

	Args:
		model: The transformer model used for evaluation.
		cache: The activation cache containing intermediate activations.
		metric: A function to evaluate the contribution or importance of a path.
				(Assumes higher scores indicate greater importance based on evaluate_path behavior).
		start_node: The initial node to begin the backward search from (e.g., FINAL_ApproxNode(layer=model.cfg.n_layers - 1, position=target_pos)).
		ground_truth_tokens: The reference tokens used for evaluating path contributions.
		min_contribution: The minimum absolute contribution score required for a path to be considered valid.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, sorted by contribution in descending order.
	"""
	last_node_contribution = evaluate_path(model, cache, start_node, metric, ground_truth_tokens)
	frontier = [(last_node_contribution, start_node)]
	completed_paths = []
	while frontier:
		if len(frontier) > 3:
			print(f"({len(frontier)})    Frontier: {frontier[:100]}... ](total {len(frontier)})")
		else:
			print(f"({len(frontier)})    Frontier: {frontier}(total {len(frontier)})")

		cur_depth_frontier = []
		# Expand all paths in the frontier looking for meaningful continuations
		for _, incomplete_path in tqdm(frontier):
			
			cur_path_start = incomplete_path[0]
			cur_path_continuations = []

			# Use a proxy compenent where heads and positions are not yet defined (declare a component of the same class)
			candidate_components = cur_path_start.get_expansion_candidates(model.cfg, include_head=True)

			# Get the meaningful candidates for expansion
			for candidate in candidate_components:
				# EMBED is the base case
				if candidate.__class__.__name__ == 'EMBED_ApproxNode':
					contribution = evaluate_path(model, cache, [candidate] + incomplete_path, metric, ground_truth_tokens)
					if contribution >= min_contribution:
						print(f"({contribution:.2f}) {[candidate]+incomplete_path}")
						completed_paths.append((contribution, [candidate] + incomplete_path))
				
				# MLP requires to check the contribution of the whole component and of the individual layers
				elif candidate.__class__.__name__ == 'MLP_ApproxNode' or candidate.__class__.__name__ == 'ATTN_ApproxNode':
					contribution = evaluate_path(model, cache, [candidate] + incomplete_path, metric, ground_truth_tokens)
					if contribution >= min_contribution:
						cur_path_continuations.append((contribution, [candidate] + incomplete_path))
			# Sort the current path continuations by contribution and take the top-k		
			cur_path_continuations.sort(key=lambda x: x[0], reverse=True)

			# Expand the frontier meaning with the sofound meaningful components
			cur_depth_frontier.extend(cur_path_continuations)
		# Limit the number of paths in the frontier to avoid memory issues
		frontier = sorted(cur_depth_frontier, key=lambda x: x[0], reverse=True)

	return sorted(completed_paths, key=lambda x: x[0], reverse=True)


def get_path_msg(path, message=None):
	if len(path) == 0:
		return message
	message = path[0].forward(message=message)
	return get_path_msg(path[1:], message=message)

def get_path(node):
	path = [node]
	while path[-1].parent is not None:
		path.append(path[-1].parent)
	return path

def PathAttributionPatching(
	model: HookedTransformer,
	msg_cache: dict,
	metric: Callable,
	root: ApproxNode,
	ground_truth_tokens: list[int],
	min_contribution: float = 0.5,
	include_negative: bool = False,
	return_all: bool = False,
) -> list[tuple[float, list[ApproxNode]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_ApproxNode.

	Args:
		model: The transformer model used for evaluation.
		cache: The activation cache containing intermediate activations.
		metric: A function to evaluate the contribution or importance of a path.
				(Assumes higher scores indicate greater importance based on evaluate_path behavior).
		start_node: The initial node to begin the backward search from (e.g., FINAL_ApproxNode(layer=model.cfg.n_layers - 1, position=target_pos)).
		ground_truth_tokens: The reference tokens used for evaluating path contributions.
		min_contribution: The minimum absolute contribution score required for a path to be considered valid.
		include_negative: If True, include paths with negative contributions. The min_contribution is therefore interpreted as a threshold on the magnitude of the contribution.
		return_all: If True, return all evaluated paths regardless of their contribution score. The search will still be guided by min_contribution.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, sorted by contribution in descending order.
	"""
	frontier = [root]
	completed_paths = []
	while frontier:
		cur_depth_frontier = []
		# Expand all paths in the frontier looking for meaningful continuations
		for node in tqdm(frontier):

			grad = node.get_gradient()
			
			childrens = []

			candidate_components = node.get_expansion_candidates(model.cfg, include_head=True)

			# Get the meaningful candidates for expansion
			for candidate in candidate_components:
				candidate_path = get_path(candidate)
				candidate_contribution = candidate.forward(message=None)
				approx_contribution = torch.einsum('bsd,bsd->b', candidate_contribution, grad)
				# EMBED is the base case
				if candidate.__class__.__name__ == 'EMBED_ApproxNode':
					if return_all:
						contribution = evaluate_path(model, msg_cache, candidate_path, metric, ground_truth_tokens)
						completed_paths.append((contribution, candidate_path))
					elif include_negative:
						if abs(approx_contribution) >= min_contribution:
							contribution = evaluate_path(model, msg_cache, candidate_path, metric, ground_truth_tokens)
							completed_paths.append((contribution, candidate_path))
					elif contribution >= min_contribution:
						if approx_contribution >= min_contribution:
							contribution = evaluate_path(model, msg_cache, candidate_path, metric, ground_truth_tokens)
							completed_paths.append((contribution, candidate_path))
				
				# MLP requires to check the contribution of the whole component and of the individual layers
				elif candidate.__class__.__name__ == 'MLP_ApproxNode' or candidate.__class__.__name__ == 'ATTN_ApproxNode':
					candidate_contribution = candidate.forward(message=None)
					approx_contribution = torch.einsum('bsd,bsd->b', candidate_contribution, grad)

					if include_negative:
						if abs(approx_contribution) >= min_contribution:
							childrens.append(candidate)
					elif approx_contribution >= min_contribution:
						childrens.append(candidate)
			cur_depth_frontier.extend(childrens)
			node.children = childrens

		frontier = cur_depth_frontier

	return sorted(completed_paths, key=lambda x: x[0], reverse=True)

