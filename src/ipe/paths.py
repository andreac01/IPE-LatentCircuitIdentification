from ipe.nodes import Node, FINAL_Node
import torch
from transformer_lens import HookedTransformer
from typing import Callable

def evaluate_path(path: list[Node], metric: Callable[[torch.Tensor], float]) -> float:
	"""
	Evaluates the contribution of a given path by executing the forward methods of each node in the path and then applying the provided metric function to the final output.
	
	Args:
		path (list of Node): The sequence of nodes representing the path to be evaluated.
		metric (Callable): A function to evaluate the contribution or importance of the path. It must accept a single parameter, the output of the last node when the path is removed.
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

def get_path(node: Node) -> list[Node]:
	"""
	Constructs the path from the given node back to the root by following parent links.

	Args:
		node (Node): The node from which to start constructing the path.
	
	Returns:
		list of Node:
			The sequence of nodes representing the path from the root to the given node.
	"""
	path = [node]
	while path[-1].parent is not None:
		path.append(path[-1].parent)
	return path


def get_path_msg(path: list[Node], message: torch.Tensor = None) -> torch.Tensor:
	"""
	Recursively computes the message by applying the forward method of each node in the path.

	Args:
		path (list of Node): 
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

def get_path_msgs(path: list[Node], messages: list[torch.Tensor] = [], msg_cache: dict = None, model: HookedTransformer = None) -> list[torch.Tensor]:
	"""
	Recursively computes and collects messages by applying the forward method of each node in the path.
	
	Args:
		path (list of Node): 
			The sequence of nodes representing the path.
		messages (list of torch.Tensor, default=[]):
			list to collect messages at each step.
		msg_cache (dict, optional):
			Cache for messages to optimize computation.
		model (HookedTransformer, optional):
			The transformer model to be used in the nodes.
	Returns:
		list of torch.Tensor:
			The list of all messages flowing through the path including intermediate ones."""
	if not path:
		return messages
	
	# Determine the patch tensor
	message = messages[-1] if messages else None
	
	# Compute the next message and append it
	if msg_cache is not None:
		path[0].msg_cache = msg_cache
	if model is not None:
		path[0].model = model
	next_message = path[0].forward(message=message)
	messages.append(next_message)

	# Recurse with the rest of the path
	return get_path_msgs(path[1:], messages=messages, msg_cache=msg_cache, model=model)

def clean_paths(paths: list[tuple[float, list]]) -> list[tuple[float, list]]:
	"""Cleans up the paths by removing references to models, parents, children, and caches to save memory. It is useful to call this function before saving the outputs of graph search to a file, avoiding saving cache, gradients, and model.
	
	Args:
		paths (list[tuple[float, list]]): 
			A list of tuples where each tuple contains a path weight and a list of Node instances representing the path.
	Returns:
		list[tuple[float, list]]: 
			The cleaned list of paths with unnecessary references removed.
	"""
	cleaned = []
	for c, path in paths:
		for node in path:
			node.model = None  # remove model reference to save memory
			node.parent = None  # remove parent reference to save memory
			node.children = []  # remove children reference to save memory
			node.msg_cache = None  # remove msg_cache reference to save memory
			node.cf_cache = None  # remove cf_cache reference to save memorys
			node.gradient = None  # remove gradient reference to save memory
			if isinstance(node, FINAL_Node):
				node.metric = None  # remove metric reference to save memory
		if isinstance(c, torch.Tensor):
			c = c.item()  # convert tensor to float for serialization
		cleaned.append((c, path))
	return cleaned