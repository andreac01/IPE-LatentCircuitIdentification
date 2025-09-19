def evaluate_path(path, metric):
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

def get_path(node):
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


def get_path_msg(path, message=None):
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