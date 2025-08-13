from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Float
import torch
from torch import Tensor
from typing import List, Optional
from backward_search.utils.nodes import Node, EMBED_Node, FINAL_Node, ATTN_Node, MLP_Node
import heapq
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple
from collections import deque
import gc
from functools import partial

def breadth_first_search(
	model: HookedTransformer,
	cache: ActivationCache,
	metric: Callable,
	start_node: list[Node],
	ground_truth_tokens: list[int],
	max_depth: int = 5,
	max_branching_factor: int = 8,
	min_contribution: float = 0.5,
	min_contribution_percentage: float = 5.0,
	inibition_task: bool = False,
) -> List[Tuple[float, List[Node]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_Node.

	Args:
		model: The transformer model used for evaluation.
		cache: The activation cache containing intermediate activations.
		metric: A function to evaluate the contribution or importance of a path.
				(Assumes higher scores indicate greater importance based on evaluate_path behavior).
		start_node: The initial node to begin the backward search from (e.g., FINAL_Node(layer=model.cfg.n_layers - 1, position=target_pos)).
		ground_truth_tokens: The reference tokens used for evaluating path contributions.
		max_depth: The maximum depth of paths to explore during the search.
		max_branching_factor: The maximum number of child nodes to expand from each node.
		min_contribution: The minimum absolute contribution score required for a path to be considered valid.
		min_contribution_percentage: The minimum percentage of the previous node's contribution required for expansion.
		inibition_task: If True, reverses the contribution metric to evaluate indirect effects.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, sorted by contribution in descending order.
	"""
	last_node_contribution = evaluate_path(model, cache, start_node, metric, ground_truth_tokens, invert_value=inibition_task)
	frontier = [(last_node_contribution, start_node)]
	completed_paths = []
	
	pbar = tqdm(range(max_depth), desc=f"BFS search")
	for depth in pbar:
		if not frontier:
			break
		print(f"Exploring depth {depth + 1} with {len(frontier)} paths in the frontier")
		if len(frontier) > 4:
			print(f"    Frontier: {frontier[:4]}... ](total {len(frontier)})")
		else:
			print(f"    Frontier: {frontier}(total {len(frontier)})")

		cur_depth_frontier = []
		# Expand all paths in the frontier looking for meaningful continuations
		for prev_contrib, incomplete_path in frontier:
			required_contribution = max(min_contribution_percentage * abs(prev_contrib) / 100.0, min_contribution)
			
			cur_path_start = incomplete_path[0]
			cur_path_continuations = []

			# Use a proxy compenent where heads and positions are not yet defined (declare a component of the same class)
			proxy_component = cur_path_start.__class__(cur_path_start.layer)
			candidate_components = proxy_component.get_prev_nodes(
				model.cfg, include_head=False, include_bos=True)

			# Target positions are the position relevant for patching the first node of the path
			target_position = cur_path_start.position
			if cur_path_start.__class__.__name__ == 'ATTN_Node' and cur_path_start.patch_keyvalue: # Assuming that we either patch the query or the keyvalue
				target_position = cur_path_start.keyvalue_position
			
			# Get the meaningful candidates for expansion
			for candidate in candidate_components:

				# EMBED is the base case
				if candidate.__class__.__name__ == 'EMBED_Node':
					contribution = evaluate_path(model, cache, [candidate] + incomplete_path, metric, ground_truth_tokens, invert_value=inibition_task)
					if contribution >= required_contribution:
						candidate.position = target_position
						completed_paths.append((contribution, [candidate] + incomplete_path))
				
				# ATTN requires to chech the contribution of the whole compoenent and of the individual heads-positions
				elif candidate.__class__.__name__ == 'ATTN_Node':
					
					whole_component_message = candidate.forward(model, cache, patch=None)
					whole_component_contribution = evaluate_path(model, cache, incomplete_path, metric, ground_truth_tokens, message=whole_component_message, invert_value=inibition_task)
	 
					# If the attention component is meaningful:
					#   1. Check which heads are contributing
					#   2. For important heads check which positions are meaningful
					if whole_component_contribution >= required_contribution:
						# 1. Check which heads are contributing
						for head in range(model.cfg.n_heads):
							node = ATTN_Node(candidate.layer, head=head, keyvalue_position=None, position=target_position, patch_keyvalue=candidate.patch_keyvalue, patch_query=candidate.patch_query)
							message = node.forward(model, cache, patch=None)
							contribution = evaluate_path(model, cache, incomplete_path, metric, ground_truth_tokens, message=message, invert_value=inibition_task)

							# 2. If this head is meaningful we go on checking the single positions
							if contribution >= required_contribution:
								# 2a. Patching the keyvalue
								if candidate.patch_keyvalue:
									for position in range(target_position+1):
										node = ATTN_Node(candidate.layer, head=head, keyvalue_position=position, position=target_position, patch_keyvalue=True, patch_query=False)
										message = node.forward(model, cache, patch=None)
										contribution = evaluate_path(model, cache, incomplete_path, metric, ground_truth_tokens, message=message, invert_value=inibition_task)
										# If the contribution is meaningful we add the node to the path
										if contribution >= required_contribution:
											cur_path_continuations.append((contribution, [node] + incomplete_path))
								if candidate.patch_query:
									# 2b. Patching the query
									node = ATTN_Node(candidate.layer, head=head, keyvalue_position=None, position=target_position, patch_keyvalue=False, patch_query=True)
									message = node.forward(model, cache, patch=None)
									contribution = evaluate_path(model, cache, incomplete_path, metric, ground_truth_tokens, message=message, invert_value=inibition_task)
									if contribution >= required_contribution:
										cur_path_continuations.append((contribution, [node] + incomplete_path))
		
				# MLP requires to check the contribution of the whole component and of the individual layers
				elif candidate.__class__.__name__ == 'MLP_Node':
					message = MLP_Node(candidate.layer, target_position).forward(model, cache, patch=None)
					contribution = evaluate_path(model, cache, incomplete_path, metric, ground_truth_tokens, message=message, invert_value=inibition_task)
					if contribution >= required_contribution:
						cur_path_continuations.append((contribution, [MLP_Node(candidate.layer, target_position)] + incomplete_path))
			
			# Sort the current path continuations by contribution and take the top-k		
			cur_path_continuations.sort(key=lambda x: x[0], reverse=True)
			cur_path_continuations = cur_path_continuations[:max_branching_factor]

			# Expand the frontier meaning with the sofound meaningful components
			cur_depth_frontier.extend(cur_path_continuations)
		frontier = cur_depth_frontier
		pbar.set_postfix({"completed_paths": len(completed_paths), "frontier_size": len(frontier)})

	# For the last step we don't need to evaluate all the nodes in the frontier, we just need to evaluate the contribution of input emebeddings
	for contrib, path in frontier:
		# 1. Get the position we are patching the first node of the path
		target_position = path[0].position
		if path[0].__class__.__name__ == 'ATTN_Node' and path[0].patch_keyvalue: # Assuming that we either patch the query or the keyvalue
			target_position = path[0].keyvalue_position
		# 2. Evaluate the contribution of the EMBED_Node at that position
		embed_node = EMBED_Node(layer=0, position=target_position)
		contribution = evaluate_path(model, cache, [embed_node] + path, metric, ground_truth_tokens, invert_value=inibition_task)
		required_contribution = max(min_contribution_percentage * abs(contrib) / 100.0, min_contribution)
		if contribution >= required_contribution:
			completed_paths.append((contribution, [embed_node] + path))
	
	# Sort the completed paths by contribution and return them
	return sorted(completed_paths, key=lambda x: x[0], reverse=True), sorted(cur_depth_frontier, key=lambda x: x[0], reverse=True)
		



def evaluate_path(model, cache, path, metric, correct_tokens, message=None, invert_value=False):
	"""
	Evaluates the contribution of a path in the transformer model using a given metric.
	Args:
		model: The transformer model.
		cache: The activation cache.
		path: A list of nodes representing the path.
		metric: A function to evaluate the path's contribution/score.
		correct_tokens: The ground truth tokens to be used for evaluation.
		message: An optional message tensor to be used to patch the first node of the path.
		invert_value: If True, inverts the value of the metric. Used as an example for indirect contribution.
	Returns:
		The contribution score of the path.
	"""
	message = path_message(model, cache, path, message=message)
	if invert_value:
		return -metric(path[-1].forward(model, cache), path[-1].forward(model, cache) - message, model, correct_tokens)
	return metric(path[-1].forward(model, cache), path[-1].forward(model, cache) - message, model, correct_tokens)




def path_message(model: HookedTransformer, cache: ActivationCache, path: list[Node], message: Tensor = None) -> Tensor:
	"""
	Computes the message through a path of nodes in the transformer.
 
	Args:
		model: The transformer model.
		cache: The activation cache.
		path: A list of nodes representing the path.
	
	Returns:
		The final message tensor after passing through the path.
	"""
	if len(path) == 0:
		return message
	# Initialize the message with the input residual
	if message is None:
		message = path[0].forward(model, cache, patch=None)
	else:
		if not isinstance(path[0], FINAL_Node):			
			message = path[0].forward(model, cache, patch=message)

	if len(path) > 1:
		return path_message(model, cache, path[1:], message=message)
 
	return message

def evaluate_path_with_cache(model, cache, path, metric, correct_tokens, message_cache, max_cached_length=0):
	message = None
	if len(path) == 0:
		return message
	if message_cache is not None:
		for i in range(max_cached_length, -1, -1):
			if i == 0:
				break
			if message_cache.get(str(path[:i]), None) is not None:
				if path[i-1].position is not None:
					message = torch.zeros_like(cache['hook_embed'])
					message[:, path[i-1].position] = message_cache[str(path[:i])]
				else:
					message = message_cache[str(path[:i])]
				break

	for j in range(i, len(path)):
		message = path[j].forward(model, cache, patch=message)
		if j <= max_cached_length:
			if path[j].position is not None:
				message_cache[str(path[:j+1])] = message[:, path[j].position, :].detach().clone()
			else:
				message_cache[str(path[:j+1])] = message.detach().clone()

	return metric(path[-1].forward(model, cache), path[-1].forward(model, cache) - message, model, correct_tokens)

def ablation_hook(residual, hook, pos=None, head=None):
	if head is not None:
		if pos is not None:
			residual[:, pos, head, :] = torch.zeros_like(residual[:, pos, head, :])
		else:
			residual[:, :, head, :] = torch.zeros_like(residual[:, :, head, :])
	else:
		if pos is not None:
			residual[:, pos, :] = torch.zeros_like(residual[:, pos, :])
		else:
			residual = torch.zeros_like(residual)
	return residual

def add_ablation_hook(model, component):
	if component.__class__.__name__ == "ATTN_Node":
		if component.position is not None:
			hook_fn = partial(ablation_hook, pos=component.position, head=component.head)
			model.add_perma_hook(f"blocks.{component.layer}.attn.hook_z", hook_fn)
	elif component.__class__.__name__ == "MLP_Node":
		hook_fn = partial(ablation_hook, pos=component.position)
		model.add_perma_hook(f"blocks.{component.layer}.hook_mlp_out", hook_fn)		
		

def breadth_first_search_recursive(
	model: HookedTransformer,
	sample_prompts: ActivationCache,
	metric: Callable,
	start_node: list[Node],
	ground_truth_tokens: list[int],
	min_contribution: float = 0.5,
) -> List[Tuple[float, List[Node]]]:
	_, cache = model.run_with_cache(sample_prompts, prepend_bos=True)
	cur_paths =  breadth_first_search_cached(
		model,
		cache,
		metric,
		start_node,
		ground_truth_tokens,
		min_contribution,
	)
	all_paths = cur_paths.copy()
	
	for layer in range(model.cfg.n_layers-1, -1, -1):
		print(f"Searching for paths in layer {layer}")
		model.reset_hooks(including_permanent=True)
		flag = False
		for _, path in all_paths:
			for component in path[:-1]:
				if component.layer == layer:
					add_ablation_hook(model, component)
					flag = True
		if flag:
			_, cache = model.run_with_cache(sample_prompts, prepend_bos=True)
			cur_paths =  breadth_first_search_cached(
				model,
				cache,
				metric,
				start_node,
				ground_truth_tokens,
				min_contribution,
			)
			all_paths.extend(cur_paths)
			merged_paths = {}
			for path in [p for _, p in all_paths]:
				if merged_paths.get(str(path), None) is None:
					contributions = [c for c, p in all_paths if p == path]
					merged_paths[str(path)] = (sum(contributions), path)
			all_paths = list(merged_paths.values())
	print(f"Found {len(merged_paths)} unique paths after merging.")
	print("Paths:", merged_paths.values())
	return list(merged_paths.values())


def breadth_first_search_cached(
	model: HookedTransformer,
	cache: ActivationCache,
	metric: Callable,
	start_node: list[Node],
	ground_truth_tokens: list[int],
	min_contribution: float = 0.5,
) -> List[Tuple[float, List[Node]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_Node.

	Args:
		model: The transformer model used for evaluation.
		cache: The activation cache containing intermediate activations.
		metric: A function to evaluate the contribution or importance of a path.
				(Assumes higher scores indicate greater importance based on evaluate_path behavior).
		start_node: The initial node to begin the backward search from (e.g., FINAL_Node(layer=model.cfg.n_layers - 1, position=target_pos)).
		ground_truth_tokens: The reference tokens used for evaluating path contributions.
		min_contribution: The minimum absolute contribution score required for a path to be considered valid.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, sorted by contribution in descending order.
	"""
	message_cache = {}
	last_node_contribution = evaluate_path_with_cache(model, cache, start_node, metric, ground_truth_tokens, message_cache)
	frontier = [(last_node_contribution, start_node)]
	completed_paths = []
	while frontier:
		if len(frontier) > 4:
			print(f"({len(frontier)})    Frontier: {frontier[:4]}... ](total {len(frontier)})")
		else:
			print(f"({len(frontier)})    Frontier: {frontier}(total {len(frontier)})")
		print("Cache has {} entries".format(len(message_cache)))
		cur_depth_frontier = []
		# Expand all paths in the frontier looking for meaningful continuations
		for prev_contrib, incomplete_path in frontier:
			
			cur_path_start = incomplete_path[0]
			cur_path_continuations = []

			# Use a proxy compenent where heads and positions are not yet defined (declare a component of the same class)
			proxy_component = cur_path_start.__class__(cur_path_start.layer)
			candidate_components = proxy_component.get_prev_nodes(
				model.cfg, include_head=False, include_bos=True)

			# Target positions are the position relevant for patching the first node of the path
			target_position = cur_path_start.position
			if cur_path_start.__class__.__name__ == 'ATTN_Node' and cur_path_start.patch_keyvalue: # Assuming that we either patch the query or the keyvalue
				target_position = cur_path_start.keyvalue_position
			
			# Get the meaningful candidates for expansion
			for candidate in candidate_components:
				# EMBED is the base case
				if candidate.__class__.__name__ == 'EMBED_Node':
					contribution = evaluate_path_with_cache(model, cache, [candidate] + incomplete_path, metric, ground_truth_tokens, message_cache)
					if contribution >= min_contribution:
						candidate.position = target_position
						completed_paths.append((contribution, [candidate] + incomplete_path))
				
				# ATTN requires to chech the contribution of the whole compoenent and of the individual heads-positions
				elif candidate.__class__.__name__ == 'ATTN_Node':
					
					whole_component_contribution = evaluate_path_with_cache(model, cache, [candidate] + incomplete_path, metric, ground_truth_tokens, message_cache)
					# If the attention component is meaningful:
					#   1. Check which heads are contributing
					#   2. For important heads check which positions are meaningful
					if whole_component_contribution >= min_contribution:
						# 1. Check which heads are contributing
						for head in range(model.cfg.n_heads):
							node = ATTN_Node(candidate.layer, head=head, keyvalue_position=None, position=target_position, patch_keyvalue=candidate.patch_keyvalue, patch_query=candidate.patch_query)
							contribution = evaluate_path_with_cache(model, cache, [node] + incomplete_path, metric, ground_truth_tokens, message_cache)

							# 2. If this head is meaningful we go on checking the single positions
							if contribution >= min_contribution:
								# 2a. Patching the keyvalue
								if candidate.patch_keyvalue:
									for position in range(target_position+1):
										node = ATTN_Node(candidate.layer, head=head, keyvalue_position=position, position=target_position, patch_keyvalue=True, patch_query=False)
										contribution = evaluate_path_with_cache(model, cache, [node] + incomplete_path, metric, ground_truth_tokens, message_cache)
										# If the contribution is meaningful we add the node to the path
										if contribution >= min_contribution:
											cur_path_continuations.append((contribution, [node] + incomplete_path))
								if candidate.patch_query:
									# 2b. Patching the query
									node = ATTN_Node(candidate.layer, head=head, keyvalue_position=None, position=target_position, patch_keyvalue=False, patch_query=True)
									contribution = evaluate_path_with_cache(model, cache, [node] + incomplete_path, metric, ground_truth_tokens, message_cache)
									if contribution >= min_contribution:
										cur_path_continuations.append((contribution, [node] + incomplete_path))
		
				# MLP requires to check the contribution of the whole component and of the individual layers
				elif candidate.__class__.__name__ == 'MLP_Node':
					node = MLP_Node(candidate.layer, target_position)
					contribution = evaluate_path_with_cache(model, cache, [node] + incomplete_path, metric, ground_truth_tokens, message_cache)
					if contribution >= min_contribution:
						cur_path_continuations.append((contribution, [node] + incomplete_path))
			# Sort the current path continuations by contribution and take the top-k		
			cur_path_continuations.sort(key=lambda x: x[0], reverse=True)

			# Expand the frontier meaning with the sofound meaningful components
			cur_depth_frontier.extend(cur_path_continuations)
		# Limit the number of paths in the frontier to avoid memory issues
		frontier = cur_depth_frontier
	message_cache.clear()  # Clear the message cache to free memory
	torch.cuda.empty_cache()  # Clear CUDA memory if using GPU
	gc.collect()  # Run garbage collection to free up memory
	# Sort the completed paths by contribution and return them
	return sorted(completed_paths, key=lambda x: x[0], reverse=True)



def breadth_first_search_cached_no_pos(
	model: HookedTransformer,
	cache: ActivationCache,
	metric: Callable,
	start_node: list[Node],
	ground_truth_tokens: list[int],
	min_contribution: float = 0.5,
	cached_path_lenght: int = 1,
) -> List[Tuple[float, List[Node]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_Node.

	Args:
		model: The transformer model used for evaluation.
		cache: The activation cache containing intermediate activations.
		metric: A function to evaluate the contribution or importance of a path.
				(Assumes higher scores indicate greater importance based on evaluate_path behavior).
		start_node: The initial node to begin the backward search from (e.g., FINAL_Node(layer=model.cfg.n_layers - 1, position=target_pos)).
		ground_truth_tokens: The reference tokens used for evaluating path contributions.
		min_contribution: The minimum absolute contribution score required for a path to be considered valid.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, sorted by contribution in descending order.
	"""
	print("Starting breadth_first_search_cached_no_pos")
	message_cache = {}
	last_node_contribution = evaluate_path_with_cache(model, cache, start_node, metric, ground_truth_tokens, message_cache, max_cached_length=cached_path_lenght)
	frontier = [(last_node_contribution, start_node)]
	completed_paths = []
	while frontier:
		if len(frontier) > 2:
			print(f"(complete: {len(completed_paths)} - frontier: {len(frontier)})    Frontier: {frontier[:2]}... ]")
		else:
			print(f"(complete: {len(completed_paths)} - frontier: {len(frontier)})   Frontier: {frontier}")
		cur_depth_frontier = []
		# Expand all paths in the frontier looking for meaningful continuations
		for prev_contrib, incomplete_path in frontier:
			
			cur_path_start = incomplete_path[0]
			cur_path_continuations = []

			# Use a proxy compenent where heads and positions are not yet defined (declare a component of the same class)
			candidate_components = cur_path_start.get_prev_nodes(
				model.cfg, include_head=False, include_bos=True)

			# Get the meaningful candidates for expansion
			for candidate in candidate_components:
				assert candidate.position is None, "This function does not support nodes with positions"

				# EMBED is the base case
				if candidate.__class__.__name__ == 'EMBED_Node':
					contribution = evaluate_path_with_cache(model, cache, [candidate] + incomplete_path, metric, ground_truth_tokens, message_cache, max_cached_length=cached_path_lenght)
					if contribution >= min_contribution:
						completed_paths.append((contribution, [candidate] + incomplete_path))

				# ATTN requires to chech the contribution of the whole compoenent and of the individual heads-positions
				elif candidate.__class__.__name__ == 'ATTN_Node':
					
					whole_component_contribution = evaluate_path_with_cache(model, cache, [candidate] + incomplete_path, metric, ground_truth_tokens, message_cache, max_cached_length=cached_path_lenght)
					# If the attention component is meaningful:
					#   1. Check which heads are contributing
					#   2. For important heads check which positions are meaningful
					if whole_component_contribution >= min_contribution:
						# 1. Check which heads are contributing
						for head in range(model.cfg.n_heads):
							node = ATTN_Node(candidate.layer, head=head, keyvalue_position=None, position=None, patch_keyvalue=candidate.patch_keyvalue, patch_query=candidate.patch_query)
							contribution = evaluate_path_with_cache(model, cache, [node] + incomplete_path, metric, ground_truth_tokens, message_cache, max_cached_length=cached_path_lenght)

							# 2. If this head is meaningful we add it to the path
							if contribution >= min_contribution:
								cur_path_continuations.append((contribution, [node] + incomplete_path))
		
				# MLP requires to check the contribution of the whole component and of the individual layers
				elif candidate.__class__.__name__ == 'MLP_Node':
					contribution = evaluate_path_with_cache(model, cache, [candidate] + incomplete_path, metric, ground_truth_tokens, message_cache, max_cached_length=cached_path_lenght)
					if contribution >= min_contribution:
						cur_path_continuations.append((contribution, [candidate] + incomplete_path))
			# Sort the current path continuations by contribution and take the top-k		
			cur_path_continuations.sort(key=lambda x: x[0], reverse=True)

			# Expand the frontier meaning with the sofound meaningful components
			cur_depth_frontier.extend(cur_path_continuations)
		# Limit the number of paths in the frontier to avoid memory issues
		frontier = cur_depth_frontier
	message_cache.clear()  # Clear the message cache to free memory
	torch.cuda.empty_cache()  # Clear CUDA memory if using GPU
	gc.collect()  # Run garbage collection to free up memory
	# Sort the completed paths by contribution and return them
	return sorted(completed_paths, key=lambda x: x[0], reverse=True)
		

def evaluate_path_with_counterfactual(model, cache, counterfactual_cache, path, metric, 
                                     correct_tokens, counterfactual_tokens=None, 
                                     message=None, invert_value=False, 
                                     take_message_from_clean=True):
	"""
	Evaluates the contribution of a path in the transformer model using a given metric.
	This function can handle both ablation-based and explicit counterfactual comparisons.
	
	Args:
		model: The transformer model.
		cache: The activation cache for clean inputs.
		counterfactual_cache: Optional activation cache for counterfactual inputs.
		path: A list of nodes representing the path.
		metric: A function to evaluate the path's contribution/score.
		correct_tokens: The ground truth tokens for clean inputs.
		counterfactual_tokens: Optional ground truth tokens for counterfactual inputs.
		message: An optional message tensor to be used to patch the first node of the path.
		invert_value: If True, inverts the value of the metric. Used for indirect contribution.
		take_message_from_clean: If True, takes initial message from clean_cache and applies path in counterfactual_cache.
		                        If False, takes initial message from counterfactual_cache and applies path in clean_cache.
	Returns:
		The contribution score of the path.
	"""
	if counterfactual_cache is None:
		# Ablation mode: use current approach
		message = path_message(model, cache, path, message=message)
		if invert_value:
			return -metric(path[-1].forward(model, cache), path[-1].forward(model, cache) - message, model, correct_tokens, use_ablation_mode=True)
		return metric(path[-1].forward(model, cache), path[-1].forward(model, cache) - message, model, correct_tokens, use_ablation_mode=True)
	else:
		# Explicit counterfactual mode
		if counterfactual_tokens is None:
			raise ValueError("counterfactual_tokens must be provided when counterfactual_cache is provided")
		
		if take_message_from_clean:
			# Take message from clean_cache, apply path logic in counterfactual_cache
			# Get the initial message from the clean cache
			if message is None:
				message = path[0].forward(model, cache, patch=None)
			else:
				if not isinstance(path[0], FINAL_Node):			
					message = path[0].forward(model, cache, patch=message)
			
			# Apply the path logic in the counterfactual cache
			message = path_message(model, counterfactual_cache, path, message=message)
			
			# Compare: clean final output vs counterfactual final output with path applied
			clean_resid = path[-1].forward(model, cache)
			counterfactual_resid_with_path = path[-1].forward(model, counterfactual_cache, patch=message)
			
		else:
			# Take message from counterfactual_cache, apply path logic in clean_cache
			# Get the initial message from the counterfactual cache
			if message is None:
				message = path[0].forward(model, counterfactual_cache, patch=None)
			else:
				if not isinstance(path[0], FINAL_Node):			
					message = path[0].forward(model, counterfactual_cache, patch=message)
			
			# Apply the path logic in the clean cache
			message = path_message(model, cache, path, message=message)
			
			# Compare: counterfactual final output vs clean final output with path applied
			counterfactual_resid = path[-1].forward(model, counterfactual_cache)
			clean_resid_with_path = path[-1].forward(model, cache, patch=message)
			
			# Swap for consistent comparison (clean vs counterfactual)
			clean_resid = counterfactual_resid
			counterfactual_resid_with_path = clean_resid_with_path
		
		if invert_value:
			return -metric(clean_resid, counterfactual_resid_with_path, model, correct_tokens, counterfactual_tokens, use_ablation_mode=False)
		return metric(clean_resid, counterfactual_resid_with_path, model, correct_tokens, counterfactual_tokens, use_ablation_mode=False)
		


def evaluate_path_with_counterfactual_cached_no_pos(model, cache, counterfactual_cache, path, metric, 
                                                   correct_tokens, counterfactual_tokens, 
                                                   message_cache, max_cached_length=2,
                                                   invert_value=True, take_message_from_clean=True):
	"""
	Evaluates the contribution of a path in the transformer model using counterfactual logic
	but working on all positions simultaneously (no position distinction).
	This function uses message caching for efficiency.
	
	Args:
		model: The transformer model.
		cache: The activation cache for clean inputs.
		counterfactual_cache: The activation cache for counterfactual inputs.
		path: A list of nodes representing the path.
		metric: A function to evaluate the path's contribution/score.
		correct_tokens: The ground truth tokens for clean inputs.
		counterfactual_tokens: The ground truth tokens for counterfactual inputs.
		message_cache: Cache for storing intermediate messages.
		max_cached_length: Maximum length of paths to cache.
		invert_value: If True, inverts the value of the metric.
		take_message_from_clean: If True, takes initial message from clean_cache and applies path in counterfactual_cache.
		                        If False, takes initial message from counterfactual_cache and applies path in clean_cache.
	Returns:
		The contribution score of the path.
	"""
	if counterfactual_cache is None or counterfactual_tokens is None:
		raise ValueError("Both counterfactual_cache and counterfactual_tokens must be provided")
	
	# Initialize message from cache if available
	message = None
	cached_length = 0
	
	# Try to find cached message for the path
	if message_cache is not None:
		for i in range(max_cached_length, -1, -1):
			if i == 0:
				break
			if message_cache.get(str(path[:i]), None) is not None:
				message = message_cache[str(path[:i])]
				cached_length = i
				break
	
	# Process the path starting from cached_length
	for j in range(cached_length, len(path)):
		if take_message_from_clean:
			# Take message from clean_cache, apply path logic in counterfactual_cache
			if j == 0:
				# Get initial message from clean cache
				message = path[j].forward(model, counterfactual_cache, patch=None) - path[j].forward(model, cache, patch=None)
			else:
				# Apply path logic in counterfactual cache
				message = path[j].forward(model, counterfactual_cache, patch=message)
		else:
			# Take message from counterfactual_cache, apply path logic in clean_cache
			if j == 0:
				# Get initial message from counterfactual cache
				message = path[j].forward(model, cache, patch=None) - path[j].forward(model, counterfactual_cache, patch=None)
			else:
				# Apply path logic in clean cache
				message = path[j].forward(model, cache, patch=message)
		
		# Cache the message if within cacheable length
		if j < max_cached_length and message_cache is not None:
			message_cache[str(path[:j+1])] = message.detach().clone()
	
	# Perform counterfactual comparison
	if take_message_from_clean:
		# Compare: clean final output vs counterfactual final output with path applied
		clean_resid = path[-1].forward(model, cache)
		counterfactual_resid_with_path = path[-1].forward(model, counterfactual_cache) - message 
	else:
		# Compare: counterfactual final output vs clean final output with path applied
		counterfactual_resid = path[-1].forward(model, counterfactual_cache)
		clean_resid_with_path = path[-1].forward(model, cache) - message
		
		# Swap for consistent comparison (clean vs counterfactual)
		clean_resid = clean_resid_with_path
		counterfactual_resid_with_path = counterfactual_resid
	
	if invert_value:
		return -metric(clean_resid, counterfactual_resid_with_path, model, correct_tokens, counterfactual_tokens, use_ablation_mode=False)
	return metric(clean_resid, counterfactual_resid_with_path, model, correct_tokens, counterfactual_tokens, use_ablation_mode=False)



def breadth_first_search_with_counterfactual(
	model: HookedTransformer,
	cache: ActivationCache,
	counterfactual_cache: Optional[ActivationCache],
	metric: Callable,
	start_node: list[Node],
	ground_truth_tokens: list[int],
	counterfactual_tokens: Optional[list[int]] = None,
	max_depth: int = 5,
	max_branching_factor: int = 8,
	min_contribution: float = 0.5,
	min_contribution_percentage: float = 5.0,
	inibition_task: bool = False,
	take_message_from_clean: bool = True,
) -> List[Tuple[float, List[Node]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_Node.
	This function can handle both ablation-based and explicit counterfactual comparisons.

	Args:
		model: The transformer model used for evaluation.
		cache: The activation cache containing intermediate activations for clean inputs.
		counterfactual_cache: Optional activation cache for counterfactual inputs.
		metric: A function to evaluate the contribution or importance of a path.
				(Assumes higher scores indicate greater importance based on evaluate_path behavior).
		start_node: The initial node to begin the backward search from (e.g., FINAL_Node(layer=model.cfg.n_layers - 1, position=target_pos)).
		ground_truth_tokens: The reference tokens for clean inputs.
		counterfactual_tokens: Optional reference tokens for counterfactual inputs.
		max_depth: The maximum depth of paths to explore during the search.
		max_branching_factor: The maximum number of child nodes to expand from each node.
		min_contribution: The minimum absolute contribution score required for a path to be considered valid.
		min_contribution_percentage: The minimum percentage of the previous node's contribution required for expansion.
		inibition_task: If True, reverses the contribution metric to evaluate indirect effects.
		take_message_from_clean: If True, takes initial message from clean_cache and applies path in counterfactual_cache.
		                        If False, takes initial message from counterfactual_cache and applies path in clean_cache.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, sorted by contribution in descending order.
	"""
	# Validate inputs
	if counterfactual_cache is not None and counterfactual_tokens is None:
		raise ValueError("counterfactual_tokens must be provided when counterfactual_cache is provided")
	if counterfactual_cache is None and counterfactual_tokens is not None:
		raise ValueError("counterfactual_cache must be provided when counterfactual_tokens is provided")
	
	last_node_contribution = evaluate_path_with_counterfactual(model, cache, counterfactual_cache, start_node, metric, ground_truth_tokens, counterfactual_tokens, invert_value=inibition_task, take_message_from_clean=take_message_from_clean)
	frontier = [(last_node_contribution, start_node)]
	completed_paths = []
	
	pbar = tqdm(range(max_depth), desc=f"BFS search with counterfactual")
	for depth in pbar:
		if not frontier:
			break
		print(f"Exploring depth {depth + 1} with {len(frontier)} paths in the frontier")
		if len(frontier) > 4:
			print(f"    Frontier: {frontier[:4]}... ](total {len(frontier)})")
		else:
			print(f"    Frontier: {frontier}(total {len(frontier)})")

		cur_depth_frontier = []
		# Expand all paths in the frontier looking for meaningful continuations
		for prev_contrib, incomplete_path in frontier:
			required_contribution = max(min_contribution_percentage * abs(prev_contrib) / 100.0, min_contribution)
			
			cur_path_start = incomplete_path[0]
			cur_path_continuations = []

			# Use a proxy compenent where heads and positions are not yet defined (declare a component of the same class)
			proxy_component = cur_path_start.__class__(cur_path_start.layer)
			candidate_components = proxy_component.get_prev_nodes(
				model.cfg, include_head=False, include_bos=True)

			# Target positions are the position relevant for patching the first node of the path
			target_position = cur_path_start.position
			if cur_path_start.__class__.__name__ == 'ATTN_Node' and cur_path_start.patch_keyvalue: # Assuming that we either patch the query or the keyvalue
				target_position = cur_path_start.keyvalue_position
			
			# Get the meaningful candidates for expansion
			for candidate in candidate_components:

				# EMBED is the base case
				if candidate.__class__.__name__ == 'EMBED_Node':
					contribution = evaluate_path_with_counterfactual(model, cache, counterfactual_cache, [candidate] + incomplete_path, metric, ground_truth_tokens, counterfactual_tokens, invert_value=inibition_task, take_message_from_clean=take_message_from_clean)
					if abs(contribution) >= required_contribution:
						candidate.position = target_position
						completed_paths.append((contribution, [candidate] + incomplete_path))
				
				# ATTN requires to chech the contribution of the whole compoenent and of the individual heads-positions
				elif candidate.__class__.__name__ == 'ATTN_Node':
					
					whole_component_message = candidate.forward(model, cache, patch=None)
					whole_component_contribution = evaluate_path_with_counterfactual(model, cache, counterfactual_cache, incomplete_path, metric, ground_truth_tokens, counterfactual_tokens, message=whole_component_message, invert_value=inibition_task, take_message_from_clean=take_message_from_clean)
	 
					# If the attention component is meaningful:
					#   1. Check which heads are contributing
					#   2. For important heads check which positions are meaningful
					if abs(whole_component_contribution) >= required_contribution:
						# 1. Check which heads are contributing
						for head in range(model.cfg.n_heads):
							node = ATTN_Node(candidate.layer, head=head, keyvalue_position=None, position=target_position, patch_keyvalue=candidate.patch_keyvalue, patch_query=candidate.patch_query)
							message = node.forward(model, cache, patch=None)
							contribution = evaluate_path_with_counterfactual(model, cache, counterfactual_cache, incomplete_path, metric, ground_truth_tokens, counterfactual_tokens, message=message, invert_value=inibition_task, take_message_from_clean=take_message_from_clean)

							# 2. If this head is meaningful we go on checking the single positions
							if abs(contribution) >= required_contribution:
								# 2a. Patching the keyvalue
								if candidate.patch_keyvalue:
									for position in range(target_position+1):
										node = ATTN_Node(candidate.layer, head=head, keyvalue_position=position, position=target_position, patch_keyvalue=True, patch_query=False)
										message = node.forward(model, cache, patch=None)
										contribution = evaluate_path_with_counterfactual(model, cache, counterfactual_cache, incomplete_path, metric, ground_truth_tokens, counterfactual_tokens, message=message, invert_value=inibition_task, take_message_from_clean=take_message_from_clean)
										# If the contribution is meaningful we add the node to the path
										if abs(contribution) >= required_contribution:
											cur_path_continuations.append((contribution, [node] + incomplete_path))
								if candidate.patch_query:
									# 2b. Patching the query
									node = ATTN_Node(candidate.layer, head=head, keyvalue_position=None, position=target_position, patch_keyvalue=False, patch_query=True)
									message = node.forward(model, cache, patch=None)
									contribution = evaluate_path_with_counterfactual(model, cache, counterfactual_cache, incomplete_path, metric, ground_truth_tokens, counterfactual_tokens, message=message, invert_value=inibition_task, take_message_from_clean=take_message_from_clean)
									if abs(contribution) >= required_contribution:
										cur_path_continuations.append((contribution, [node] + incomplete_path))
		
				# MLP requires to check the contribution of the whole component and of the individual layers
				elif candidate.__class__.__name__ == 'MLP_Node':
					message = MLP_Node(candidate.layer, target_position).forward(model, cache, patch=None)
					contribution = evaluate_path_with_counterfactual(model, cache, counterfactual_cache, incomplete_path, metric, ground_truth_tokens, counterfactual_tokens, message=message, invert_value=inibition_task, take_message_from_clean=take_message_from_clean)
					if abs(contribution) >= required_contribution:
						cur_path_continuations.append((contribution, [MLP_Node(candidate.layer, target_position)] + incomplete_path))
			
			# Sort the current path continuations by contribution and take the top-k		
			cur_path_continuations.sort(key=lambda x: x[0], reverse=True)
			cur_path_continuations = cur_path_continuations[:max_branching_factor]

			# Expand the frontier meaning with the sofound meaningful components
			cur_depth_frontier.extend(cur_path_continuations)
		frontier = cur_depth_frontier
		pbar.set_postfix({"completed_paths": len(completed_paths), "frontier_size": len(frontier)})

	# For the last step we don't need to evaluate all the nodes in the frontier, we just need to evaluate the contribution of input emebeddings
	for contrib, path in frontier:
		# 1. Get the position we are patching the first node of the path
		target_position = path[0].position
		if path[0].__class__.__name__ == 'ATTN_Node' and path[0].patch_keyvalue: # Assuming that we either patch the query or the keyvalue
			target_position = path[0].keyvalue_position
		# 2. Evaluate the contribution of the EMBED_Node at that position
		embed_node = EMBED_Node(layer=0, position=target_position)
		contribution = evaluate_path_with_counterfactual(model, cache, counterfactual_cache, [embed_node] + path, metric, ground_truth_tokens, counterfactual_tokens, invert_value=inibition_task, take_message_from_clean=take_message_from_clean)
		required_contribution = max(min_contribution_percentage * abs(contrib) / 100.0, min_contribution)
		if abs(contribution) >= required_contribution:
			completed_paths.append((contribution, [embed_node] + path))
	
	# Sort the completed paths by contribution and return them
	return sorted(completed_paths, key=lambda x: x[0], reverse=True)
		

def breadth_first_search_with_counterfactual_cached_no_pos(
	model: HookedTransformer,
	cache: ActivationCache,
	counterfactual_cache: ActivationCache,
	metric: Callable,
	start_node: list[Node],
	ground_truth_tokens: list[int],
	counterfactual_tokens: list[int],
	max_depth: int = 10,
	max_branching_factor: int = 8,
	min_contribution: float = 10,
	min_contribution_percentage: float = 0.0,
	inibition_task: bool = False,
	take_message_from_clean: bool = True,
	cached_path_length: int = 2,
	absolute: bool = True
) -> List[Tuple[float, List[Node]]]:
	"""
	Performs a Breadth-First Search (BFS) starting from a node backwards to identify
	the most significant paths reaching it from an EMBED_Node.
	This function combines counterfactual logic with position-agnostic processing and message caching.

	Args:
		model: The transformer model used for evaluation.
		cache: The activation cache containing intermediate activations for clean inputs.
		counterfactual_cache: The activation cache for counterfactual inputs.
		metric: A function to evaluate the contribution or importance of a path.
				(Assumes higher scores indicate greater importance based on evaluate_path behavior).
		start_node: The initial node to begin the backward search from (e.g., FINAL_Node(layer=model.cfg.n_layers - 1, position=target_pos)).
		ground_truth_tokens: The reference tokens for clean inputs.
		counterfactual_tokens: The reference tokens for counterfactual inputs.
		max_depth: The maximum depth of paths to explore during the search.
		max_branching_factor: The maximum number of child nodes to expand from each node.
		min_contribution: The minimum absolute contribution score required for a path to be considered valid.
		min_contribution_percentage: The minimum percentage of the previous node's contribution required for expansion.
		inibition_task: If True, reverses the contribution metric to evaluate indirect effects.
		take_message_from_clean: If True, takes initial message from clean_cache and applies path in counterfactual_cache.
		                        If False, takes initial message from counterfactual_cache and applies path in clean_cache.
		cached_path_length: Maximum length of paths to cache for efficiency.
	Returns:
		A list of tuples containing the contribution score and the corresponding path, sorted by contribution in descending order.
	"""
	print("Starting breadth_first_search_with_counterfactual_cached_no_pos")
	message_cache = {}
	last_node_contribution = evaluate_path_with_counterfactual_cached_no_pos(
		model, cache, counterfactual_cache, start_node, metric, ground_truth_tokens, 
		counterfactual_tokens, message_cache, max_cached_length=cached_path_length,
		invert_value=inibition_task, take_message_from_clean=take_message_from_clean
	)
	frontier = [(last_node_contribution, start_node)]
	completed_paths = []
	
	pbar = tqdm(range(max_depth), desc=f"BFS search with counterfactual (no pos)")
	for depth in pbar:
		if not frontier:
			break
		print(f"Exploring depth {depth + 1} with {len(frontier)} paths in the frontier")
		if len(frontier) > 4:
			print(f"    Frontier: {frontier[:4]}... ](total {len(frontier)})")
		else:
			print(f"    Frontier: {frontier}(total {len(frontier)})")

		cur_depth_frontier = []
		completed_in_depth = 0
		expanded_paths = 0
		# Expand all paths in the frontier looking for meaningful continuations
		path_pbar = tqdm(frontier, desc=f"Expanding paths at depth {depth + 1}", leave=False)
		for prev_contrib, incomplete_path in path_pbar:
			expanded_paths += 1
			path_pbar.set_postfix({"expanded": expanded_paths, "completed_in_depth": completed_in_depth, "total_completed": len(completed_paths)})
			required_contribution = max(min_contribution_percentage * abs(prev_contrib) / 100.0, min_contribution)
			
			cur_path_start = incomplete_path[0]
			cur_path_continuations = []

			# Use a proxy component where heads and positions are not yet defined (declare a component of the same class)
			candidate_components = cur_path_start.get_prev_nodes(
				model.cfg, include_head=False, include_bos=True)

			# Get the meaningful candidates for expansion
			for candidate in candidate_components:
				assert candidate.position is None, "This function does not support nodes with positions"

				# EMBED is the base case
				if candidate.__class__.__name__ == 'EMBED_Node':
					contribution = evaluate_path_with_counterfactual_cached_no_pos(
						model, cache, counterfactual_cache, [candidate] + incomplete_path, metric, 
						ground_truth_tokens, counterfactual_tokens, message_cache, 
						max_cached_length=cached_path_length, invert_value=inibition_task, 
						take_message_from_clean=take_message_from_clean
					)
					flag = abs(contribution) >= required_contribution if absolute else contribution >= required_contribution
					if flag:
						completed_paths.append((contribution, [candidate] + incomplete_path))
						completed_in_depth += 1

				# ATTN requires to check the contribution of the whole component and of the individual heads
				elif candidate.__class__.__name__ == 'ATTN_Node':
					
					whole_component_contribution = evaluate_path_with_counterfactual_cached_no_pos(
						model, cache, counterfactual_cache, [candidate] + incomplete_path, metric, 
						ground_truth_tokens, counterfactual_tokens, message_cache, 
						max_cached_length=cached_path_length, invert_value=inibition_task, 
						take_message_from_clean=take_message_from_clean
					)
					
					# If the attention component is meaningful:
					#   1. Check which heads are contributing
					flag = abs(whole_component_contribution) >= required_contribution if absolute else whole_component_contribution >= required_contribution
					if flag:
						# 1. Check which heads are contributing
						for head in range(model.cfg.n_heads):
							node = ATTN_Node(candidate.layer, head=head, keyvalue_position=None, position=None, 
											patch_keyvalue=candidate.patch_keyvalue, patch_query=candidate.patch_query)
							contribution = evaluate_path_with_counterfactual_cached_no_pos(
								model, cache, counterfactual_cache, [node] + incomplete_path, metric, 
								ground_truth_tokens, counterfactual_tokens, message_cache, 
								max_cached_length=cached_path_length, invert_value=inibition_task, 
								take_message_from_clean=take_message_from_clean
							)

							# 2. If this head is meaningful we add it to the path
							flag = abs(contribution) >= required_contribution if absolute else contribution >= required_contribution
							if flag:
								cur_path_continuations.append((contribution, [node] + incomplete_path))
		
				# MLP requires to check the contribution of the whole component
				elif candidate.__class__.__name__ == 'MLP_Node':
					contribution = evaluate_path_with_counterfactual_cached_no_pos(
						model, cache, counterfactual_cache, [candidate] + incomplete_path, metric, 
						ground_truth_tokens, counterfactual_tokens, message_cache, 
						max_cached_length=cached_path_length, invert_value=inibition_task, 
						take_message_from_clean=take_message_from_clean
					)
					flag = abs(contribution) >= required_contribution if absolute else contribution >= required_contribution
					if flag:
						cur_path_continuations.append((contribution, [candidate] + incomplete_path))
			
			# Sort the current path continuations by contribution and take the top-k		
			cur_path_continuations.sort(key=lambda x: x[0], reverse=True)
			cur_path_continuations = cur_path_continuations[:max_branching_factor]

			# Expand the frontier with the found meaningful components
			cur_depth_frontier.extend(cur_path_continuations)
		frontier = cur_depth_frontier
		pbar.set_postfix({"completed_paths": len(completed_paths), "frontier_size": len(frontier), "completed_in_depth": completed_in_depth, "expanded_paths": expanded_paths})
		print(f"Depth {depth + 1} completed: {expanded_paths} paths expanded, {completed_in_depth} paths completed, {len(frontier)} paths in next frontier")

	# For the last step we don't need to evaluate all the nodes in the frontier, we just need to evaluate the contribution of input embeddings
	final_pbar = tqdm(frontier, desc="Processing final frontier paths", leave=False)
	for contrib, path in final_pbar:
		# Evaluate the contribution of the EMBED_Node (no position distinction)
		embed_node = EMBED_Node(layer=0, position=None)
		contribution = evaluate_path_with_counterfactual_cached_no_pos(
			model, cache, counterfactual_cache, [embed_node] + path, metric, 
			ground_truth_tokens, counterfactual_tokens, message_cache, 
			max_cached_length=cached_path_length, invert_value=inibition_task, 
			take_message_from_clean=take_message_from_clean
		)
		required_contribution = max(min_contribution_percentage * abs(contrib) / 100.0, min_contribution)
		flag = abs(contribution) >= required_contribution if absolute else contribution >= required_contribution
		if flag:
			completed_paths.append((contribution, [embed_node] + path))
			final_pbar.set_postfix({"total_completed": len(completed_paths)})
	
	# Clean up memory
	message_cache.clear()  # Clear the message cache to free memory
	torch.cuda.empty_cache()  # Clear CUDA memory if using GPU
	gc.collect()  # Run garbage collection to free up memory
	
	print(f"Search completed: {len(completed_paths)} total paths found")
	
	# Sort the completed paths by contribution and return them
	return sorted(completed_paths, key=lambda x: x[0], reverse=True)
		
