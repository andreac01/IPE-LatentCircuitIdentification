import networkx as nx
from math import ceil
from typing import List, Tuple, Set, Union



class ImgNode:
	def __init__(self, cmpt: str, layer: int, head_idx: Union[int, None], position: Union[int, None], in_type: str = None):
		self.cmpt = cmpt
		self.layer = layer
		self.head_idx = head_idx
		self.position = position
		self.in_type = in_type
	
	def __repr__(self):
		return f"ImgNode(cmpt={self.cmpt}, layer={self.layer}, head_idx={self.head_idx}, position={self.position}, in_type={self.in_type})"
	
	def __str__(self):
		# A more concise string representation for use as a unique ID in the graph
		head_str = f"h{self.head_idx}" if self.head_idx is not None else ""
		pos_str = f"p{self.position}" if self.position is not None else ""
		type_str = f"_{self.in_type}" if self.in_type else ""
		return f"{self.cmpt}_l{self.layer}{head_str}{pos_str}{type_str}"

	def __lt__(self, other):
		if not isinstance(other, ImgNode):
			return NotImplemented
		return (self.layer, self.cmpt, self.position, self.head_idx) < (other.layer, other.cmpt, other.position, other.head_idx)

	def __eq__(self, other):
		if not isinstance(other, ImgNode):
			return False
		return str(self) == str(other)

	def __hash__(self):
		return hash(str(self))

def make_graph_from_paths(paths: List[Tuple[float, List[ImgNode]]],
						  n_layers: int,
						  n_heads: int,
						  n_positions: int,
						  divide_heads: bool = True) -> nx.MultiDiGraph:
	G = nx.MultiDiGraph()
	all_nodes: Set[ImgNode] = set()
	all_edge_weights = []

	for path_idx, (path_weight, path_nodes) in enumerate(paths):
		if not path_nodes:
			continue
		for node in path_nodes:
			all_nodes.add(node)
		for i in range(len(path_nodes) - 1):
			src_node = path_nodes[i]
			dst_node = path_nodes[i+1]
			G.add_edge(src_node, dst_node, weight=path_weight.item(), path_idx=path_idx, in_type=dst_node.in_type)
			all_edge_weights.append(path_weight.item())

	possible_nodes_context = {
		ImgNode('emb', 0, None, pos) for pos in range(n_positions)
	}
	possible_nodes_context |= {
		ImgNode('lmh', n_layers, None, pos) for pos in range(n_positions)
	}
	possible_nodes_context |= {
		ImgNode('mlp', layer, None, pos)
		for layer in range(n_layers)
		for pos in range(n_positions)
	}

	if divide_heads:
		possible_nodes_context |= {
			ImgNode('sa', layer, head, pos, in_type=t) 
			for layer in range(n_layers)
			for head in range(n_heads)
			for pos in range(n_positions)
			for t in ['query', 'key-value']
		}
	else:
		possible_nodes_context |= {
			ImgNode('attn', layer, head, pos, in_type=t)
			for layer in range(n_layers)
			for head in range(n_heads)
			for pos in range(n_positions)
			for t in ['query', 'key-value']
		}
		
	G.add_nodes_from(all_nodes)
	G.add_nodes_from(possible_nodes_context)
	G.graph['max_weight'] = max(all_edge_weights) if all_edge_weights else 1.0
	G.graph['min_weight'] = min(all_edge_weights) if all_edge_weights else 0.0
	G.graph['max_abs_weight'] = max(abs(w) for w in all_edge_weights) if all_edge_weights else 1.0
	G.graph['num_paths'] = len(paths)
	return G

def place_node(node: ImgNode, 
			   n_layers: int, 
			   layer_spacing: float, 
			   pos_spacing: float = 1.0, 
			   divide_heads: bool = True, 
			   n_heads: int = 0, 
			   heads_per_row: int = 4) -> tuple[float, float]:
	
	base_x = (node.position or 0) * pos_spacing
	
	if node.cmpt == 'emb':
		return base_x, - layer_spacing * 0.75
	if node.cmpt == 'lmh':
		return base_x, (n_layers + 0.25) * layer_spacing

	base_y = (node.layer) * layer_spacing

	if divide_heads:
		rows = ceil(n_heads / heads_per_row) if n_heads > 0 else 1
		head_row_height = layer_spacing / (2*rows)
		
		if node.cmpt == 'mlp':
			return base_x, base_y + head_row_height * rows
			
		if node.cmpt == 'sa':
			col = (node.head_idx or 0) % heads_per_row
			row_idx = (node.head_idx or 0) // heads_per_row
			x_offset = (col - (heads_per_row - 1) / 2) * (pos_spacing / (heads_per_row + 2))
			return base_x + x_offset, base_y + row_idx * head_row_height - layer_spacing * 0.1
			
	else: # Full attention blocks
		if node.cmpt == 'mlp':
			return base_x, base_y + layer_spacing * 0.2
		if node.cmpt == 'attn':
			return base_x, base_y - layer_spacing * 0.2
			
	return base_x, base_y

def get_image_paths(contrib_and_path: Tuple[float, List], divide_heads=True) -> Tuple[float, List[ImgNode]]:
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
		position = node.position
		
		if name == 'attn':
			head_idx = node.head
			in_type = "query" if node.patch_query else "key-value"
		if divide_heads:
			if name == 'attn':
				name = 'sa'
		
		img_nodes.append(ImgNode(name, node.layer, head_idx, position, in_type=in_type))
	return (contrib, img_nodes)

def create_graph_data(img_node_paths, n_layers, n_heads, n_positions, divide_heads, prompt_str_tokens, output_str_tokens):
	"""Helper function to create graph data from paths."""
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
	print(G.graph['max_abs_weight'])
	print("returning graph data")
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
