import networkx as nx
from typing import Union
from math import ceil
import torch

class ImgNode:
	"""
	Represents a node in the visualization graph.

	Attributes:
		cmpt (str): Component type (e.g., 'emb', 'sa', 'mlp', 'lmh').
		layer (int): Layer index of the component.
		head_idx (int | None): Head index for attention components, or None.
		position (int | None): Token position in the sequence, or None.
		in_type (str | None): Input type for attention components ('query' or 'key-value'), or None.
	"""
	def __init__(self, cmpt: str, layer: int, head_idx: Union[int, None], position: Union[int, None], in_type: str = None) -> None:
		"""
		Initialize an ImgNode.

		Args:
			cmpt: Component type (e.g., 'emb', 'sa', 'mlp', 'lmh').
			layer: Layer index of the component.
			head_idx: Head index for attention components, or None.
			position: Token position in the sequence, or None.
			in_type: Input type for attention components ('query' or 'key-value'), or None.
		"""
		self.cmpt = cmpt
		self.layer = layer
		self.head_idx = head_idx
		self.position = position
		self.in_type = in_type
	
	def __repr__(self) -> str:
		""" A detailed string representation of the node for debugging purposes.
		
		Returns:
			str: A detailed string representation of the ImgNode instance.
		"""
		return f"ImgNode(cmpt={self.cmpt}, layer={self.layer}, head_idx={self.head_idx}, position={self.position}, in_type={self.in_type})"
	
	def __str__(self) -> str:
		"""Return a concise human-readable identifier for the node.
		
		Returns:
			str: A concise string identifier for the ImgNode instance.
		"""
		head_str = f"h{self.head_idx}" if self.head_idx is not None else ""
		pos_str = f"p{self.position}" if self.position is not None else ""
		type_str = f"_{self.in_type}" if self.in_type else ""
		return f"{self.cmpt}_l{self.layer}{head_str}{pos_str}{type_str}"

	def __lt__(self, other) -> bool:
		"""
		Define ordering for ImgNode instances.

		Nodes are compared by (layer, cmpt, position, head_idx). Useful for sorting.

		Args:
			other (ImgNode): Another ImgNode instance to compare against.
		
		Returns:
			bool: True if self is less than other, False otherwise.
		"""
		if not isinstance(other, ImgNode):
			return NotImplemented
		return (self.layer, self.cmpt, self.position, self.head_idx) < (other.layer, other.cmpt, other.position, other.head_idx)

	def __eq__(self, other) -> bool:
		"""Equality based on the string identifier produced by __str__.
		
		Args:
			other (ImgNode): Another ImgNode instance to compare against.
		
		Returns:
			bool: True if both ImgNode instances are equal, False otherwise."""
		if not isinstance(other, ImgNode):
			return False
		return str(self) == str(other)

	def __hash__(self) -> int:
		"""Hash based on the string identifier, matching __eq__ semantics.
		
		Returns:
			int: Hash value of the ImgNode instance.
		"""
		return hash(str(self))

def make_graph_from_paths(paths: list[tuple[float, list[ImgNode]]],
						  n_layers: int,
						  n_heads: int,
						  n_positions: int,
						  divide_heads: bool = True) -> nx.MultiDiGraph:
	"""Create a directed multigraph from a list of weighted paths for visualization.

	Args:
		paths (list[tuple[float, list[ImgNode]]]):
			Each element is a tuple (weight, path_nodes), where weight is a float
			or a single-element torch.Tensor, and path_nodes is a list of ImgNode
			instances ordered along the path.
		n_layers (int):
			Total number of transformer layers.
		n_heads (int):
			Number of attention heads per layer (used when divide_heads is True).
		n_positions (int):
			Number of token positions in the sequence (used to populate context nodes).
		divide_heads (bool, optional):
			If True, represent individual attention heads as separate nodes ('sa').
			If False, represent attention as a single block ('attn'). Defaults to True.

	Returns:
		nx.MultiDiGraph:
			A directed multigraph whose nodes are ImgNode instances and whose edges
			have attributes:
				- weight: path contribution (float)
				- path_idx: index of the source path in the input list
				- in_type: input type carried by the destination node ('query'/'key-value' or None)
			The graph object also contains summary attributes in G.graph:
				- max_weight, min_weight, max_abs_weight, num_paths
	"""

def place_node(node: ImgNode, 
			   n_layers: int, 
			   layer_spacing: float, 
			   pos_spacing: float = 1.0, 
			   divide_heads: bool = True, 
			   n_heads: int = 0, 
			   heads_per_row: int = 4) -> tuple[float, float]:
	"""Determines the (x, y) position of a node in the graph for visualization purposes.
	
	Args:
		node (ImgNode): 
			The ImgNode instance to be placed.
		n_layers (int): 
			The total number of layers in the model.
		layer_spacing (float): 
			The vertical spacing to leave in between layers.
		pos_spacing (float, optional):
			The horizontal spacing to leave in between token positions. Defaults to 1.0.
		divide_heads (bool, optional):
			Whether to represent attention heads separately (True) or as a single attention block (False). Defaults to True.
		n_heads (int, optional):
			The total number of attention heads in each attention module. Required if divide_heads is True. Defaults to 0.
		heads_per_row (int, optional):
			The number of attention heads to display per row when dividing heads. Defaults to 4. The attention block will be divided into ceil(n_heads / heads_per_row) rows.
	
	Returns:
		tuple[float, float]:"""
	
	base_x = (node.position or 0) * pos_spacing
	
	if node.cmpt == 'emb':
		return base_x, - layer_spacing * 0.5
	if node.cmpt == 'lmh':
		return base_x, (n_layers + 0.5) * layer_spacing

	base_y = (node.layer) * layer_spacing

	if divide_heads:
		rows = ceil(n_heads / heads_per_row) if n_heads > 0 else 1
		head_row_height = layer_spacing * 0.4 / rows
		
		if node.cmpt == 'mlp':
			return base_x, base_y + layer_spacing * 0.7
			
		if node.cmpt == 'sa':
			col = (node.head_idx or 0) % heads_per_row
			row_idx = (node.head_idx or 0) // heads_per_row
			x_offset = (col - (heads_per_row - 1) / 2) * (pos_spacing / (heads_per_row + 2))
			return base_x + x_offset, base_y + row_idx * head_row_height + 0.1 * layer_spacing
			
	else: # Full attention blocks
		if node.cmpt == 'mlp':
			return base_x, base_y + layer_spacing * 0.7
		if node.cmpt == 'attn':
			return base_x, base_y + layer_spacing * 0.3
			
	return base_x, base_y

def get_image_path(contrib_and_path: tuple[float, list], divide_heads=True) -> tuple[float, list[ImgNode]]:
	"""Converts a path represented as a list of nodes into a format suitable for visualization, using ImgNode instances.
	
	Args:
		contrib_and_path (tuple[float, list]): 
			A tuple containing the path contribution (weight) and a list of nodes representing the path. It also accepts contribution as single-element torch.Tensor.
		divide_heads (bool, optional):
			Whether to represent attention heads separately (True) or as a single attention block (False). Defaults to True.
	
	Returns:
		tuple[float, list[ImgNode]]: 
			A tuple containing the path contribution and a list of ImgNode instances representing the path.
	"""
	contrib, path = contrib_and_path
	img_nodes = []
	for idx, node in enumerate(path):
		name = node.__class__.__name__.split('_')[0].lower()
		if 'final' in name:
			name = 'lmh'
		if 'emb' in name:
			name = 'emb'
		
		head_idx = None
		in_type = None
		position = node.position if node.position is not None else 0
		
		if name == 'attn':
			head_idx = node.head
			in_type = "query" if node.patch_query else "key-value"
		if divide_heads:
			if name == 'attn':
				name = 'sa'
		
		img_nodes.append(ImgNode(name, node.layer, head_idx, position, in_type=in_type))
	if isinstance(contrib, torch.Tensor):
		contrib = contrib.item()
	return (contrib, img_nodes)

def create_graph_data(
		img_node_paths: list[tuple[float, list[ImgNode]]],
		n_layers: int,
		n_heads: int,
		n_positions: int,
		divide_heads: bool,
		prompt_str_tokens: list[str],
		output_str_tokens: list[str]
	) -> dict:
	"""
	Generates graph data for visualizing a neural network's attention pathways.

	This function processes a list of node paths from a neural network and constructs
	a graph representation with properly positioned nodes and connected edges.
	The resulting data is used for visualization in the web application.

	Args:
		img_node_paths (list[tuple[float, list[ImgNode]]]):
			A list of paths through the network nodes to visualize. Each path is a tuple
			containing a weight and a list of ImgNode instances.
		n_layers (int):
			The number of transformer layers in the model.
		n_heads (int):
			The number of attention heads per layer.
		n_positions (int):
			The number of token positions in the sequence.
		divide_heads (bool):
			Whether to visually separate attention heads in the layout.
		prompt_str_tokens (list[str]):
			A list of input tokens for labeling embedding nodes.
		output_str_tokens (list[str]):
			A list of output tokens for labeling LMH (language model head) nodes.

	Returns:
		dict:
			A dictionary containing the following keys:
			- 'nodes': A list of node data with positions and attributes.
			- 'edges': A list of edge connections with weights and path indices.
			- 'max_abs_weight': The maximum absolute weight in the graph.
			- 'num_paths': The number of paths in the graph.
			- 'n_positions': The number of token positions.
			- 'n_layers': The number of layers in the model.
			- 'n_heads': The number of attention heads.
			- 'tokenized_prompt': A list of input tokens.
			- 'tokenized_target': A list containing the last output token.
	"""
	G = make_graph_from_paths(img_node_paths, n_layers, n_heads, n_positions, divide_heads=divide_heads)
	
	involved_nodes = {u for u, _, _ in G.edges(data=True)} | {v for _, v, _ in G.edges(data=True)}
	
	# Node placement logic
	pos_spacing = 1.5
	heads_per_row = 4
	
	if divide_heads:
		layer_spacing_multiplier = (ceil(n_heads / heads_per_row) if n_heads > 0 else 1) + 2
	else:
		layer_spacing_multiplier = 4.0
	
	layer_spacing = pos_spacing * layer_spacing_multiplier

	pos_dict = {
		node: place_node(node, n_layers, layer_spacing=layer_spacing, pos_spacing=pos_spacing, 
							divide_heads=divide_heads, n_heads=n_heads, heads_per_row=heads_per_row)
		for node in G.nodes()
	}
	
	# Flip Y coordinates to place outputs at the top
	for node, (x, y) in pos_dict.items():
		pos_dict[node] = (x, -y)

	nodes_data = [
		{
			'id': str(n),
			'x': pos[0],
			'y': pos[1],
			'cmpt': n.cmpt,
			'layer': n.layer,
			'head': n.head_idx,
			'position': n.position,
			'in_type': n.in_type,
			'involved': n in involved_nodes,
			'label': (
				prompt_str_tokens[n.position].replace('\u0120', '_') if n.cmpt == 'emb' and n.position is not None and n.position < len(prompt_str_tokens) else
				output_str_tokens[n.position].replace('\u0120', '_') if n.cmpt == 'lmh' and n.position is not None and n.position < len(output_str_tokens) else
				str(n.head_idx) if n.cmpt == 'sa' else ''
			)
		} for n, pos in pos_dict.items()
	]
	
	edges_data = []
	path_idx_to_start_pos = {i: p[1][0].position for i, p in enumerate(img_node_paths)}

	for u, v, data in G.edges(data=True):
		path_idx = data['path_idx']
		edges_data.append({
			'source': str(u),
			'target': str(v),
			'weight': data['weight'],
			'path_idx': path_idx,
			'in_type': data.get('in_type'),
			'start_pos': path_idx_to_start_pos.get(path_idx, 0)
		})
	return {
		'nodes': nodes_data,
		'edges': edges_data,
		'max_abs_weight': G.graph['max_abs_weight'],
		'num_paths': G.graph['num_paths'],
		'n_positions': n_positions,
		'n_layers': n_layers,
		'n_heads': n_heads,
		'tokenized_prompt': prompt_str_tokens,
		'tokenized_target': [output_str_tokens[-1]]
	}
