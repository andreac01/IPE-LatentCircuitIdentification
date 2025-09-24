import matplotlib.pyplot as plt
import networkx as nx
from math import ceil
import numpy as np
from ipe.webutils.image_nodes import ImgNode, place_node

def plot_transformer_paths(G: nx.MultiDiGraph,
							n_layers: int,
							n_heads: int,
							n_positions: int,
							example_input: list[str] = [],
							example_output: list[str] = [],
							cmap_name: str = 'coolwarm', 
							heads_per_row: int = 4,
							save_fig: bool = False,
							save_path: str = 'transformer_paths.png',
							max_w: float = None, # User-defined normalization cap for weight/contribution
							color_scheme: str = 'path_weight', # 'path_index', 'path_weight', 'input_position'
							divide_heads: bool = True
						) -> None:
	"""
	Visualize the transformer multigraph G.
	The plot shows the nodes and edges of the graph over the transformer architecture.
	Components are divided by input position, layer and eventually head.

	Args:
		G (nx.MultiDiGraph): 
			The graph to visualize.
		n_layers (int): 
			Total number of layers in the model.
		n_heads (int): 
			Number of attention heads (used if divide_heads=True).
		n_positions (int): 
			Sequence length or number of positions.
		example_input (list[str]): 
			List of input tokens for labeling the embeddings. Default is empty list.
		example_output (list[str]):
			List of output tokens for labeling the lmh outputs. Default is empty list.
		cmap_name (str):
			Colormap name for edge coloring. Default is 'viridis'.
		heads_per_row (int):
			Number of sa heads per row (used if divide_heads=True). Default is 4.
		save_fig (bool):
			If True, save the figure to a file. Default is False.
		save_path (str): 
			Path to save the figure if save_fig is True. Default is 'transformer_paths.png'.
		max_w (float or None): 
			Maximum absolute contribution for color/width normalization. If None, derived from data.
		color_scheme (str):
			Color scheme for edges ('path_index', 'path_weight', or 'input_position').
		divide_heads (bool): 
			If True, attention is shown per head ('sa' nodes).
			If False, attention is shown as full blocks ('attn' nodes).
	Returns:
		None. Displays the plot.
	"""
	if len(example_input) < n_positions:
		example_input = [''] * (n_positions - len(example_input)) + example_input
	if len(example_output) < n_positions:
		example_output = [''] * (n_positions - len(example_output)) + example_output

	if divide_heads:
		layer_spacing_multiplier = (ceil( (0.5*n_heads) / heads_per_row) if n_heads > 0 else 1)
	else:
		layer_spacing_multiplier = 1.0
		n_heads = 1
		heads_per_row = 1

	layer_spacing = layer_spacing_multiplier
	pos_spacing = max(heads_per_row/2, 2)

	pos_dict = {
		node: place_node(node, n_layers, layer_spacing, pos_spacing=pos_spacing,
						 divide_heads=divide_heads,
						 n_heads=n_heads, heads_per_row=heads_per_row)
		for node in G.nodes()
	}
			

	height = layer_spacing * (n_layers + 2) 
	width = max(n_positions * pos_spacing, height/2)
	fig, ax = plt.subplots(figsize=(width, height))

	involved = {u for u, v, data in G.edges(data=True)} | {v for u, v, data in G.edges(data=True)}
	uninvolved = set(G.nodes()) - involved

	involved_emb = [n for n in involved if isinstance(n, ImgNode) and n.cmpt == 'emb']
	involved_lmh = [n for n in involved if isinstance(n, ImgNode) and n.cmpt == 'lmh']
	involved_mlp = [n for n in involved if isinstance(n, ImgNode) and n.cmpt == 'mlp']
	
	uninvolved_emb = [n for n in uninvolved if isinstance(n, ImgNode) and n.cmpt == 'emb']
	uninvolved_lmh = [n for n in uninvolved if isinstance(n, ImgNode) and n.cmpt == 'lmh']
	uninvolved_mlp = [n for n in uninvolved if isinstance(n, ImgNode) and n.cmpt == 'mlp']

	if divide_heads:
		involved_attn_sa = [n for n in involved if isinstance(n, ImgNode) and n.cmpt == 'sa']
		uninvolved_sa_bg = [n for n in uninvolved if isinstance(n, ImgNode) and n.cmpt == 'sa']
	else:
		involved_attn_full = [n for n in involved if isinstance(n, ImgNode) and n.cmpt == 'attn']
		uninvolved_attn_full_bg = [n for n in uninvolved if isinstance(n, ImgNode) and n.cmpt == 'attn']

	def draw_nodes(nodes, img_layer=0, **opts):
		if nodes:
			nx.draw_networkx_nodes(G, pos_dict, nodelist=nodes, ax=ax, **opts).set_zorder(img_layer)
	
	attn_size = min(112.5, 450*heads_per_row/n_heads) if divide_heads else 450

	draw_nodes(involved_emb, node_shape='s', node_size=2400, node_color='white', edgecolors='black', linewidths=1.0, alpha=0.75, img_layer=10)
	draw_nodes(involved_lmh, node_shape='s', node_size=2400, node_color='white', edgecolors='black', linewidths=1.0, alpha=0.75, img_layer=10)
	draw_nodes(involved_mlp, node_shape='s', node_size=450, node_color='white', edgecolors='black', linewidths=1.0, alpha=0.75, img_layer=10)

	draw_nodes(uninvolved_mlp, node_shape='s', node_size=450, node_color='white', edgecolors='grey',  linewidths=0.5, alpha=0.55)
	draw_nodes(uninvolved_emb, node_shape='s', node_size=2400, node_color='white', edgecolors='grey',  linewidths=0.5, alpha=0.55)
	draw_nodes(uninvolved_lmh, node_shape='s', node_size=2400, node_color='white', edgecolors='grey',  linewidths=0.5, alpha=0.55)

	if divide_heads:
		draw_nodes(involved_attn_sa, node_shape='o', node_size=attn_size, node_color='white', edgecolors='black', linewidths=1.0, alpha=0.75, img_layer=10)
		draw_nodes(uninvolved_sa_bg, node_shape='o', node_size=attn_size, node_color='white', edgecolors='grey',  linewidths=0.5, alpha=0.55)
		labels_attn_inv = {n: str(n.head_idx) for n in involved_attn_sa}
		text_items = nx.draw_networkx_labels(G, pos_dict, labels=labels_attn_inv, font_size=8, font_weight='normal', ax=ax, font_color='black')
		for _, text_obj in text_items.items():
			text_obj.set_zorder(11)
	else: 
		draw_nodes(involved_attn_full, node_shape='o', node_size=attn_size, node_color='white', edgecolors='black', linewidths=1.0, alpha=0.75, img_layer=10)
		draw_nodes(uninvolved_attn_full_bg, node_shape='o', node_size=attn_size, node_color='white', edgecolors='grey',  linewidths=0.5, alpha=0.55)

	labels_emb_inv = {n: example_input[n.position].replace(' ', '_') for n in involved_emb}
	labels_emb_uninv = {n: example_input[n.position].replace(' ', '_') for n in uninvolved_emb}
	text_items = nx.draw_networkx_labels(G, pos_dict, labels=labels_emb_inv, font_size=12, font_weight='normal', ax=ax)
	for _, text_obj in text_items.items():
		text_obj.set_zorder(11)
	nx.draw_networkx_labels(G, pos_dict, labels=labels_emb_uninv, font_size=12, font_weight='light', ax=ax)

	labels_lmh_inv = {n: example_output[n.position].replace(' ', '_') for n in involved_lmh}
	labels_lmh_uninv = {n: example_output[n.position].replace(' ', '_') for n in uninvolved_lmh}
	text_items = nx.draw_networkx_labels(G, pos_dict, labels=labels_lmh_inv, font_size=12, font_weight='normal', ax=ax)
	for _, text_obj in text_items.items():
		text_obj.set_zorder(11)
	nx.draw_networkx_labels(G, pos_dict, labels=labels_lmh_uninv, font_size=12, font_weight='light', ax=ax)

	sorted_edges = sorted(G.edges(data=True, keys=True), key=lambda x: x[3]['weight'], reverse=True)

	num_paths = G.graph.get('num_paths', 1)
	
	# Determine normalization factor for contribution values (weights)
	# This will be used for both color intensity and edge width.
	graph_max_abs_weight = G.graph.get('max_abs_weight', 1.0)
	if graph_max_abs_weight == 0: 
		graph_max_abs_weight = 1.0 # Avoid division by zero

	# Use user-provided max_w for normalization cap if available, otherwise use graph's max_abs_weight
	norm_cap = max_w if max_w is not None else graph_max_abs_weight
	if norm_cap == 0:
		norm_cap = 1.0
	
	max_width = G.graph.get('max_weight', 1)
	min_width = G.graph.get('min_weight', 0)



	if color_scheme == 'path_index':
		cmap_obj = plt.get_cmap(cmap_name, num_paths if num_paths > 0 else 1)
		# This list is indexed by path_idx later
		edge_colors_by_path_idx = [cmap_obj(i) for i in np.arange(num_paths if num_paths > 0 else 1)]
	elif color_scheme == 'path_weight':
		cmap_obj = plt.get_cmap(cmap_name)
	elif color_scheme == 'input_position':
		cmap_obj_pos = plt.get_cmap(cmap_name, n_positions if n_positions > 0 else 1)
		path_idx_to_color = {}
		default_cmap_obj_pos = plt.get_cmap(cmap_name, num_paths if num_paths > 0 else 1)
		
		for in_node, _, _, data in sorted_edges:
			path_idx = data['path_idx']
			if in_node.cmpt == 'emb' and in_node.position is not None:
				path_idx_to_color[path_idx] = cmap_obj_pos(in_node.position % (n_positions if n_positions > 0 else 1))
	else:
		raise ValueError(f"Unknown color scheme: {color_scheme}. Use 'path_index', 'path_weight', or 'input_position'.")

	all_drawn_edges = []
	parallel_edge_drawn = {} 
	width_scale = 18 
	alpha = 0.6

	for i, (u, v, key, data) in enumerate(sorted_edges):
		path_idx = data['path_idx']
		contribution = data['weight'] # Renaming 'w' to 'contribution' for clarity
		
		# Edge Width: Proportional to the absolute value of the contribution
		current_width = (abs(contribution) / norm_cap) * width_scale
		current_width = max(0.1, current_width) # Ensure minimum width for visibility

		# Edge Color
		if color_scheme == 'path_weight':
			normalized_abs_contribution = (contribution - min_width) / (max_width - min_width)

			current_color = cmap_obj(normalized_abs_contribution) # Use cmap for positive
		
		elif color_scheme == 'path_index':
			current_color = edge_colors_by_path_idx[path_idx % len(edge_colors_by_path_idx)]
		
		elif color_scheme == 'input_position':
			current_color = path_idx_to_color.get(path_idx, default_cmap_obj_pos(0.5)) # Default to middle color if not found


		edge_type = data.get('in_type', None)

		if (u,v) not in parallel_edge_drawn:
			parallel_edge_drawn[(u,v)] = 0
		rad_sign = 1 if parallel_edge_drawn[(u,v)] % 2 == 0 else -1
		rad_magnitude = 0.05 + 0.1 * (parallel_edge_drawn[(u,v)] // 2) 
		current_rad = rad_sign * rad_magnitude
		parallel_edge_drawn[(u,v)] += 1
		
		linestyle = 'solid' 
		if edge_type == 'query':
			linestyle = 'dotted'
		
		connectionstyle = f'arc3,rad={current_rad:.2f}'

		edge_patches = nx.draw_networkx_edges(
			G, pos_dict,
			edgelist=[(u, v, key)],
			width=current_width,
			edge_color=[current_color], # Pass as a list with one color
			alpha=alpha, 
			connectionstyle=connectionstyle,
			arrowstyle='-', 
			style=linestyle
		)
		if edge_patches:
			if hasattr(edge_patches, '__iter__'): 
				all_drawn_edges.extend(edge_patches)
			else:
				all_drawn_edges.append(edge_patches)

	edge_patch_map = {}
	for i, (u, v, key, data) in enumerate(sorted_edges):
		if i < len(all_drawn_edges):
			edge_patch_map[(u, v, key)] = all_drawn_edges[i]

	for u, v, key, data in sorted_edges:
		patch = edge_patch_map.get((u, v, key))
		if patch and hasattr(patch, 'set_zorder'):
			weight = data['weight']
			z_intensity = abs(weight) / (norm_cap + 1e-9) # Normalized 0-1
			z = 1 + z_intensity * 3 # Scale to 1-9
			patch.set_zorder(z)

	ax.set_yticks([layer_spacing * (i - 0.5) for i in range(0, n_layers + 3)], minor=False)
	ax.set_yticklabels(['EMB'] + list(range(n_layers)) + ['LMH'] + [''], fontsize=16)
	ax.tick_params(axis='y', left=True, labelleft=True)

	ax.set_xticks([(i - 0.5)*pos_spacing for i in range(n_positions + 1)], minor=True)
	ax.set_yticks([layer_spacing * (i) for i in range(0, n_layers + 1)], minor=True) 
	ax.grid(False, which='major')
	ax.grid(True, which='minor', linestyle='-', alpha=0.8)

	xs, ys = zip(*pos_dict.values()) if pos_dict else ([0],[0])
	
	# Include edge positions in xlim calculation using the bounding box of drawn elements
	bbox = ax.dataLim
	min_x = min(min(xs), bbox.x0)
	max_x = max(max(xs), bbox.x1)
	
	ax.set_xlim(min_x - 0.05 * max_x, 1.05 * max_x)
	ax.set_ylim(max(ys) + 1 * layer_spacing, min(ys) - 1 * layer_spacing) 
	ax.invert_yaxis()

	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)

	fig.tight_layout()
	if save_fig:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()