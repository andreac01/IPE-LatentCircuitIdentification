from transformer_lens import HookedTransformer, ActivationCache
import torch
from torch import Tensor
from backward_search_approximated.utils.nodes import ApproxNode, FINAL_ApproxNode, MLP_ApproxNode, ATTN_ApproxNode, EMBED_ApproxNode
from tqdm import tqdm
from functools import partial
from typing import Callable
from itertools import islice
import gc
from collections import defaultdict
import weakref



def evaluate_path(model, cache, path, metric, correct_tokens):
	"""
	Evaluates the contribution of a given path by executing the forward methods of each node in the path and then applying the provided metric function to the final output.
	
	Args:
		model (HookedTransformer): The transformer model used for evaluation.
		cache (ActivationCache): The activation cache containing intermediate activations.
		path (list of ApproxNode): The sequence of nodes representing the path to be evaluated.
		metric (Callable): A function to evaluate the contribution or importance of the path.
			It must accept three parameters:
			- The output of the last node in the path.
			- The output of the last node when the path is removed
			- The model itself
		correct_tokens (list of int): The reference tokens used for evaluating path contributions.
	
	Returns:
		float: 
			The contribution score of the path as determined by the metric function.
	"""
	message = None
	if len(path) == 0:
		return message

	for i in range(len(path)):
		message = path[i].forward(message=message)

	return metric(corrupted_resid=path[-1].forward() - message)



def IsolatingPathEffect_BW(
	model: HookedTransformer,
	metric: Callable,
	start_node: ApproxNode,
	min_contribution: float = 0.5,
	include_negative: bool = False,
	return_all: bool = False,
	counterfactual_cache: ActivationCache = None,
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
		start_node (ApproxNode): 
			The initial node to begin the backward search from (e.g., FINAL_ApproxNode(layer=model.cfg.n_layers - 1, position=target_pos)).
		min_contribution (float, default=0.5):
			The minimum absolute contribution score required for a path to be considered valid.
		include_negative (bool, default=False): 
			If True, include paths with negative contributions. The min_contribution is therefore interpreted as a threshold on the magnitude of the contribution.
		return_all (bool, default=False): 
			If True, return all evaluated complete paths regardless of their contribution score. The search will still be guided by min_contribution.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, 
		sorted by contribution in descending order.
	"""
	last_node_contribution = evaluate_path(model, [start_node], metric)
	frontier = [(last_node_contribution, [start_node])]
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
			candidate_components = cur_path_start.get_expansion_candidates(model.cfg, include_head=True)

			# Get the meaningful candidates for expansion
			for candidate in candidate_components:
				# EMBED is the base case, the path is complete and after evaluation can be added to the completed paths
				if candidate.__class__.__name__ == 'EMBED_ApproxNode':
					# If the candidate is 
					if counterfactual_cache is not None:
						initial_message = counterfactual_cache[candidate.input_name]
					else:
						initial_message = None
					contribution = evaluate_path(model, [candidate] + incomplete_path, metric, initial_message)
					if return_all:
						completed_paths.append((contribution, [candidate] + incomplete_path))
					elif include_negative:
						if abs(contribution) >= min_contribution:
							completed_paths.append((contribution, [candidate] + incomplete_path))
					elif contribution >= min_contribution:
						completed_paths.append((contribution, [candidate] + incomplete_path))
				
				# ATTNs and MLPs are possible expansions of the current path to be added to the frontier
				elif candidate.__class__.__name__ == 'MLP_ApproxNode' or candidate.__class__.__name__ == 'ATTN_ApproxNode':
					if counterfactual_cache is not None:
						initial_message = counterfactual_cache[candidate.input_name]
					else:
						initial_message = None
					contribution = evaluate_path(model, [candidate] + incomplete_path, metric, initial_message)
					if include_negative:
						if abs(contribution) >= min_contribution:
							cur_path_continuations.append((contribution, [candidate] + incomplete_path))
					elif contribution >= min_contribution:
						cur_path_continuations.append((contribution, [candidate] + incomplete_path))

			cur_depth_frontier.extend(cur_path_continuations)
		# Sort the frontier just for visualization purposes
		frontier = sorted(cur_depth_frontier, key=lambda x: x[0], reverse=True)

	return sorted(completed_paths, key=lambda x: x[0], reverse=True)


def get_path_msg(path, message=None):
	"""
	Recursively computes the message by applying the forward method of each node in the path.

	Args:
		path (list of ApproxNode): 
			The sequence of nodes representing the path.
		message (torch.Tensor, default=None):
			Initial message to be passed to the first node in the path.

	Returns:
		torch.Tensor:
			The final message after applying all nodes in the path.
	"""
	if len(path) == 0:
		return message
	message = path[0].forward(message=message)
	return get_path_msg(path[1:], message=message)

def get_path(node):
	"""
	Constructs the path from the given node back to the root by following parent links.

	Args:
		node (ApproxNode): The node from which to start constructing the path.
	
	Returns:
		list of ApproxNode:
			The sequence of nodes representing the path from the root to the given node.
	"""
	path = [node]
	while path[-1].parent is not None:
		path.append(path[-1].parent)
	return path


def batch_iterable(iterable, batch_size):
	it = iter(iterable)
	while True:
		chunk = list(islice(it, batch_size))
		if not chunk:
			break
		yield chunk



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
				candidate_contributions = torch.stack([candidate.forward(message=None) for candidate in candidate_batch], dim=0) # TODO: make this somewhat batched by position if possible

				approx_contributions = torch.einsum('xbsd,bsd->x', candidate_contributions, grad) # TODO: make this batched (up to 40% of the total time)
				for i, candidate in enumerate(candidate_batch):
					approx_contribution = approx_contributions[i]
					# EMBED is the base case
					if candidate.__class__.__name__ == 'EMBED_ApproxNode':
						candidate_path = get_path(candidate)
						if return_all:
							contribution = evaluate_path(model, msg_cache, candidate_path, metric, ground_truth_tokens)
							completed_paths.append((contribution, candidate_path))
						elif include_negative:
							if abs(approx_contribution.item()) >= min_contribution:
								contribution = evaluate_path(model, msg_cache, candidate_path, metric, ground_truth_tokens)
								completed_paths.append((contribution, candidate_path))
						elif approx_contribution >= min_contribution:
							contribution = evaluate_path(model, msg_cache, candidate_path, metric, ground_truth_tokens)
							completed_paths.append((contribution, candidate_path))
					
					# MLP requires to check the contribution of the whole component and of the individual layers
					elif candidate.__class__.__name__ == 'MLP_ApproxNode' or candidate.__class__.__name__ == 'ATTN_ApproxNode':
						if include_negative:
							if abs(approx_contribution.item()) >= min_contribution: # TODO: make this faster batched? np.abs?
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

		frontier = cur_depth_frontier

	return sorted(completed_paths, key=lambda x: x[0], reverse=True)

