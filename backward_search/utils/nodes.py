import abc
import torch
from torch import Tensor
from jaxtyping import Float
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from backward_search.utils.component_contributions import contribution_mlp, contribution_attn, contribution_attn_fast, contribution_attn2
from functools import total_ordering
from typing import List, Optional



@total_ordering
class Node(abc.ABC):
	"""
	Abstract base class for a component node in the transformer network.
	Includes parent and children links for use in tree structures.
	"""
	def __init__(self, name: str, layer: int, input_residual: str, position: int = None):
		self.name = name
		self.layer = layer
		self.position = position
		self.input_residual = input_residual
		self.parent: Optional['Node'] = None
		self.children: List['Node'] = []

	def add_child(self, child: 'Node'):
		"""Adds a node as a child of this node and sets its parent."""
		if child not in self.children:
			self.children.append(child)
		child.parent = self

	@abc.abstractmethod
	def forward(self, model: HookedTransformer, cache: ActivationCache, patch: Tensor = None) -> Tensor:
		"""
		Performs the forward pass for this specific node.

		Args:
			model: The transformer model.
			cache: The activation cache from a forward pass.
			patch: The patch to apply/remove related to this node's contribution.

		Returns:
			The output tensor representing the contribution of this node.
		"""
		pass

	@abc.abstractmethod
	def get_next_nodes(self, model_cfg: HookedTransformerConfig, sequence_length: int, include_head: bool = False, divide_pos: bool = True) -> list['Node']:
		"""
		Returns a list of *potential* next nodes in the computational graph
		originating from this node. These are not automatically added as children.

		Args:
			model_cfg: The configuration of the transformer model.
			sequence_length: The sequence length of the input.
			include_head: Whether to generate specific head nodes for ATTN.

		Returns:
			A list of potential next nodes.
		"""
		pass

	@abc.abstractmethod
	def get_prev_nodes(self, model_cfg: HookedTransformerConfig, include_head: bool = False, include_bos: bool = True) -> list['Node']:
		"""
		Returns a list of *potential* previous nodes in the computational graph
		that contribute to this node. These are not automatically set as the parent.

		Args:
			model_cfg: The configuration of the transformer model.
			include_head: Whether to consider specific head nodes for ATTN.
			include_bos: Whether to include position 0 (usually BOS token).

		Returns:
			A list of potential previous nodes.
		"""
		pass

	@abc.abstractmethod
	def __repr__(self) -> str:
		"""
		Returns a string representation of the node.

		Returns:
			A string representation of the node.
		"""
		pass

	def _get_sort_key(self):
		"""Helper method to return a tuple for sorting."""
		# Define an order for node types
		type_order = {EMBED_Node: 0, ATTN_Node: 1, MLP_Node: 2, FINAL_Node: 3}
		node_type = type(self)
		layer = self.layer if self.layer is not None else -1
		pos = self.position if self.position is not None else -1
		keyvalue_position = getattr(self, 'keyvalue_position', None)
		keyvalue_position = keyvalue_position if keyvalue_position is not None else -1
		head = getattr(self, 'head', None)
		head = head if head is not None else -1

		return (
			layer,
			pos,
			type_order.get(node_type, 99), # Place unknown types last
			keyvalue_position,
			head
		)

	def __lt__(self, other):
		if not isinstance(other, Node):
			return NotImplemented
		return self._get_sort_key() < other._get_sort_key()

	def __eq__(self, other):
		if not isinstance(other, Node):
			return NotImplemented
		if (self.layer != other.layer or
			self.position != other.position or
			type(self) != type(other)):
			return False
		if isinstance(self, ATTN_Node) and isinstance(other, ATTN_Node):
			return self.head == other.head and self.position == other.position and self.keyvalue_position == other.keyvalue_position and self.patch_keyvalue == other.patch_keyvalue and self.patch_query == other.patch_query
		return True

	def __hash__(self):
		head_val = getattr(self, 'head', None)
		return hash((type(self).__name__, self.layer, self.position, head_val))

class MLP_Node(Node):
	"""Represents an MLP node in the transformer."""
	def __init__(self, layer: int, position: int = None):
		super().__init__(name=f"MLP_{layer}", layer=layer, input_residual=f"blocks.{layer}.hook_resid_mid", position=position)

	def forward(self, model: HookedTransformer, cache: ActivationCache, patch: Tensor) -> Tensor:
		if patch is None:
			patch = cache[self.input_residual].clone()
		return contribution_mlp(model, self.layer, cache[self.input_residual].clone(), position=self.position) - contribution_mlp(model, self.layer, cache[self.input_residual].clone(), patch_to_remove=patch, position=self.position)

	def get_next_nodes(self, model_cfg: HookedTransformerConfig, sequence_length: int, include_head: bool = False, divide_pos: bool = True) -> list[Node]:
		next_nodes = []
		if self.position is not None:
			positions_to_iterate = range(self.position, sequence_length) if divide_pos else [None]
		else:
			positions_to_iterate = range(sequence_length) if divide_pos else [None]

		for l in range(self.layer + 1, model_cfg.n_layers):
			for p in positions_to_iterate:
				if include_head:
					for h in range(model_cfg.n_heads):
						next_nodes.append(ATTN_Node(layer=l, head=h, position=p, keyvalue_position=self.position, patch_keyvalue=True, patch_query=False))
				else:
					next_nodes.append(ATTN_Node(layer=l, head=None, position=p, keyvalue_position=self.position, patch_keyvalue=True, patch_query=False))
			if include_head:
				for h in range(model_cfg.n_heads):
					next_nodes.append(ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))
			else:
				next_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))
			next_nodes.append(MLP_Node(layer=l, position=self.position))
		next_nodes.append(FINAL_Node(layer=model_cfg.n_layers - 1, position=self.position))
		return next_nodes

	def get_prev_nodes(self, model_cfg: HookedTransformerConfig, include_head: bool = False, include_bos: bool = True) -> list[Node]:
		"""Returns a list of potential previous nodes that contribute to this MLP node.
		Previous nodes are:
			- MLP, EMBED and ATTN nodes in self.position from previous layers.
			- ATTN nodes in all previous positions from current layers.
		ATTN nodes are patched both in query and key-value positions separately.
		Args:
			model_cfg: The configuration of the transformer model.
			include_head: Whether to consider specific head nodes for ATTN.
			include_bos: Whether to include position 0 (usually BOS token).
		Returns:
			A list of potential previous nodes.
   		"""
		prev_nodes = []
		start_pos = 0 if include_bos else 1
		if self.position is not None:
			positions_to_iterate = range(start_pos, self.position + 1)
		else:
			positions_to_iterate = [None]

		# MLP nodes from previous layers
		# ATTN nodes from previous layers
		for l in range(self.layer):
			prev_nodes.append(MLP_Node(layer=l, position=self.position))
			for p in positions_to_iterate:
				if include_head:
					prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=p, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=p, patch_keyvalue=True, patch_query=False))
			if include_head:
				prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True) for h in range(model_cfg.n_heads)])
			else:
				prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))
		# ATTN nodes from current layer
		for p in positions_to_iterate:
			if include_head:
				prev_nodes.extend([ATTN_Node(layer=self.layer, head=h, position=self.position, keyvalue_position=p, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
			else:
				prev_nodes.append(ATTN_Node(layer=self.layer, head=None, position=self.position, keyvalue_position=p, patch_keyvalue=True, patch_query=False))
		if include_head:
			prev_nodes.extend([ATTN_Node(layer=self.layer, head=h, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True) for h in range(model_cfg.n_heads)])
		else:
			prev_nodes.append(ATTN_Node(layer=self.layer, head=None, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))
		# EMBED node
		prev_nodes.append(EMBED_Node(layer=0, position=self.position))
		# Remove duplicates
		prev_nodes = list(set(prev_nodes))
		return prev_nodes

	def __repr__(self):
		return f"MLP_Node(layer={self.layer}, position={self.position})"

	def __hash__(self):
		return hash((type(self).__name__, self.layer, self.position))

class ATTN_Node(Node):
	"""Represents an Attention node (potentially a specific head) in the transformer."""
	def __init__(self, layer: int, head: int = None, position: int = None, keyvalue_position: int = None, patch_query: bool = True, patch_keyvalue: bool = True):
		super().__init__(name=f"ATTN_{layer}_{head}" if head is not None else f"ATTN_{layer}", layer=layer, input_residual=f"blocks.{layer}.hook_resid_pre", position=position)
		self.head = head
		self.position = position
		self.keyvalue_position = keyvalue_position
		self.patch_keyvalue = patch_keyvalue
		self.patch_query = patch_query

		if self.position is not None and self.keyvalue_position is not None:
			assert self.position >= self.keyvalue_position, "query position must be greater than or equal to keyvalue position"

	def forward(self, model: HookedTransformer, cache: ActivationCache, patch: Tensor) -> Tensor:
		# if patch is None:
		# 	patch = cache[self.input_residual].clone()

		non_patched = contribution_attn2(model, self.layer, cache, head=self.head, in_position=self.keyvalue_position, out_position=self.position)
		if patch is None:
			if self.head is not None:
				return non_patched - model.blocks[self.layer].attn.b_O
			else:
				return non_patched
		patched = contribution_attn2(model, self.layer, cache, head=self.head, in_position=self.keyvalue_position, out_position=self.position, patch_to_remove=patch, patch_query=self.patch_query, patch_key=self.patch_keyvalue, patch_value=self.patch_keyvalue)

		return non_patched - patched
		
	def get_next_nodes(self, model_cfg: HookedTransformerConfig, sequence_length: int, include_head: bool = False, divide_pos: bool = True) -> list[Node]:
		next_nodes = []
		if self.position is not None:
			positions_to_iterate = range(self.position, sequence_length) if divide_pos else [None]
		else:
			positions_to_iterate = range(sequence_length) if divide_pos else [None]
	
		next_nodes.append(MLP_Node(layer=self.layer, position=self.position))

		for l in range(self.layer + 1, model_cfg.n_layers):
			for p in positions_to_iterate:
				if include_head:
					for h in range(model_cfg.n_heads):
						next_nodes.append(ATTN_Node(layer=l, head=h, position=p, keyvalue_position=self.position, patch_keyvalue=True, patch_query=False))
				else:
					next_nodes.append(ATTN_Node(layer=l, head=None, position=p, keyvalue_position=self.position, patch_keyvalue=True, patch_query=False))
			if include_head:
				for h in range(model_cfg.n_heads):
					next_nodes.append(ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))
			else:
				next_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))
			next_nodes.append(MLP_Node(layer=l, position=self.position))
		next_nodes.append(FINAL_Node(layer=model_cfg.n_layers - 1, position=self.position))
		return next_nodes

	def get_prev_nodes(self, model_cfg: HookedTransformerConfig, include_head: bool = False, include_bos: bool = True) -> list[Node]:
		"""Returns a list of potential previous nodes that contribute to this ATTN node.
		Previous nodes are:
			- MLP, EMBED and ATTN nodes in self.position from previous layers if patch_query=True.
			- MLP, EMBED and ATTN nodes in all previous positions from previous layers if patch_keyvalue=True.
		Args:
			model_cfg: The configuration of the transformer model.
			include_head: Whether to consider specific head nodes for ATTN.
			include_bos: Whether to include position 0 (usually BOS token).
		Returns:
			A list of potential previous nodes."""
		prev_nodes = []
		start_pos = 0 if include_bos else 1

		# MLPs
		for l in range(self.layer):
			if self.patch_query:
				prev_nodes.append(MLP_Node(layer=l, position=self.position))
			if self.patch_keyvalue and (not self.patch_query or self.position != self.keyvalue_position): # Note that if self.position is None than also self.keyvalue_position is None
				prev_nodes.append(MLP_Node(layer=l, position=self.keyvalue_position))    
	
 		# EMBED node
		if self.patch_query:
			prev_nodes.append(EMBED_Node(layer=0, position=self.position))
		if self.patch_keyvalue and (not self.patch_query or self.position != self.keyvalue_position):
			prev_nodes.append(EMBED_Node(layer=0, position=self.keyvalue_position))
   
		# ATTN nodes patching current query position
		if self.patch_query:
			for l in range(self.layer):
				# prev ATTN query positions
				if include_head:
					prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))
				
				# prev ATTN key-value positions
				if self.position is not None:
					for keyvalue_position in range(start_pos, self.position + 1):
						if include_head:
							prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
						else:
							prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_keyvalue=True, patch_query=False))
				else:
					if include_head:
						prev_nodes.extend([ATTN_Node(layer=l, head=h, position=None, keyvalue_position=None, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
					else:
						prev_nodes.append(ATTN_Node(layer=l, head=None, position=None, keyvalue_position=None, patch_keyvalue=True, patch_query=False))

		# ATTN nodes patching current key-value position
		if self.patch_keyvalue:
			keyvalue_positions = range(start_pos, self.keyvalue_position+1) if self.keyvalue_position is not None else [None]
			for l in range(self.layer):
				# prev ATTN query positions
				if include_head:
					prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.keyvalue_position, keyvalue_position=None, patch_keyvalue=False, patch_query=True) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.keyvalue_position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))

				# prev ATTN key-value positions				
				for prev_keyvalue_position in keyvalue_positions:
					if include_head:
						prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.keyvalue_position, keyvalue_position=prev_keyvalue_position, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
					else:
						prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.keyvalue_position, keyvalue_position=prev_keyvalue_position, patch_keyvalue=True, patch_query=False))

		# Remove duplicates
		prev_nodes = list(set(prev_nodes))
		return prev_nodes

	def __repr__(self):
		return f"ATTN_Node(layer={self.layer}, head={self.head}, position={self.position}, keyvalue_position={self.keyvalue_position}, patch_query={self.patch_query}, patch_keyvalue={self.patch_keyvalue})"

	def __hash__(self):
		return hash((type(self).__name__, self.layer, self.head, self.position, self.keyvalue_position, self.patch_query, self.patch_keyvalue))
class EMBED_Node(Node):
	"""Represents the embedding node in the transformer."""
	def __init__(self, input_residual: str = 'hook_embed', layer: int = 0, position: int = None):
		super().__init__(name=f"EMB", layer=0, input_residual=input_residual, position=position)

	def forward(self, model: HookedTransformer, cache: ActivationCache, patch: Tensor = None) -> Tensor:
		embedding = cache["hook_embed"].clone() #+ cache["hook_pos_embed"].clone()
		if patch is not None:
			embedding = embedding - patch
		if self.position is not None:
			embedding[:, :self.position, :] = torch.zeros_like(embedding[:, :self.position, :])
			embedding[:, self.position + 1:, :] = torch.zeros_like(embedding[:, self.position + 1:, :])
		return embedding

	def get_next_nodes(self, model_cfg: HookedTransformerConfig, sequence_length: int, include_head: bool = False, divide_pos: bool = True) -> list[Node]:
		next_nodes = []
		if self.position is not None:
			positions_to_iterate = range(self.position, sequence_length) if divide_pos else [None]
		else:
			positions_to_iterate = range(sequence_length) if divide_pos else [None]

		for l in range(model_cfg.n_layers):
			for p in positions_to_iterate:
				if include_head:
					for h in range(model_cfg.n_heads):
						next_nodes.append(ATTN_Node(layer=l, head=h, position=p, keyvalue_position=self.position, patch_keyvalue=True, patch_query=False))
				else:
					next_nodes.append(ATTN_Node(layer=l, head=None, position=p, keyvalue_position=self.position, patch_keyvalue=True, patch_query=False))
			if include_head:
				for h in range(model_cfg.n_heads):
					next_nodes.append(ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))
			else:
				next_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))
			next_nodes.append(MLP_Node(layer=l, position=self.position))
		next_nodes.append(FINAL_Node(layer=model_cfg.n_layers - 1, position=self.position))

		return next_nodes

	def get_prev_nodes(self, model_cfg: HookedTransformerConfig, sequence_length: int, include_head: bool = False, include_bos: bool = True) -> list[Node]:
		return []

	def __repr__(self):
		return f"EMBED_Node(layer={self.layer}, position={self.position})"

	def __hash__(self):
		return hash((type(self).__name__, self.layer, self.position))

class FINAL_Node(Node):
	"""Represents the final node in the transformer."""
	def __init__(self, layer: int, position: int = None):
		super().__init__(name=f"FINAL", layer=layer, input_residual=f"blocks.{layer}.hook_resid_post", position=position)

	def forward(self, model: HookedTransformer, cache: ActivationCache, patch: Tensor = None) -> Tensor:
		res = cache[self.input_residual].clone()
		if patch is not None:
			res = patch
		if self.position is not None:
			res_zeroed = torch.zeros_like(res)
			res_zeroed[:, self.position, :] = res[:, self.position, :]
			res = res_zeroed
		return res

	def get_next_nodes(self, model_cfg: HookedTransformerConfig, sequence_length: int, include_head: bool = False, divide_pos: bool = True) -> list[Node]:
		return []

	def get_prev_nodes(self, model_cfg: HookedTransformerConfig, include_head: bool = True, include_bos: bool = True) -> list[Node]:
		"""
		Returns a list of potential previous nodes that contribute to this FINAL node.
		Previous nodes are:
			- MLP, EMBED and ATTN nodes in self.position from all layers.
		Args:
			model_cfg: The configuration of the transformer model.
			include_head: Whether to consider specific head nodes for ATTN.
			include_bos: Whether to include position 0 (usually BOS token).
		Returns:
			A list of potential previous nodes.
		"""
		prev_nodes = []
		start_pos = 0 if include_bos else 1
	
   
		for l in range(model_cfg.n_layers):
			# MLPs
			prev_nodes.append(MLP_Node(layer=l, position=self.position))
			
			# ATTN query positions
			if include_head:
				prev_nodes.extend([ATTN_Node(layer=l, head=h, keyvalue_position=None, position=self.position, patch_keyvalue=False, patch_query=True) for h in range(model_cfg.n_heads)])
			else:
				prev_nodes.append(ATTN_Node(layer=l, head=None, keyvalue_position=None, position=self.position, patch_keyvalue=False, patch_query=True))
			
			# ATTN key-value positions
			if self.position is not None:
				for keyvalue_position in range(self.position + 1):
					if include_head:
						prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
					else:
						prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_keyvalue=True, patch_query=False))
			else:
				if include_head:
					prev_nodes.extend([ATTN_Node(layer=l, head=h, position=None, keyvalue_position=None, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=None, patch_keyvalue=True, patch_query=False))
		
		prev_nodes.append(EMBED_Node(layer=0, position=self.position))
		# Remove duplicates
		prev_nodes = list(set(prev_nodes))
		return prev_nodes

	def __repr__(self):
		pos_str = f", position={self.position}" if self.position is not None else ""
		return f"FINAL_Node(layer={self.layer}{pos_str})"

	def __hash__(self):
		return hash((type(self).__name__, self.layer, self.position))