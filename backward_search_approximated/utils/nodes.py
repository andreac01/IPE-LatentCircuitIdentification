import abc
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from backward_search_approximated.utils.attention import custom_attention_forward
from functools import total_ordering
from typing import Optional



@total_ordering
class ApproxNode(abc.ABC):
	def __init__(self, model: HookedTransformer, layer: int, position: int = None, parent = None, children = set(), msg_cache = {}, grad_cache = {}):
		self.model = model
		self.layer = layer
		self.position = position
		self.parent= parent
		self.children = children
		self.msg_cache = msg_cache
		self.grad_cache = grad_cache

	def add_child(self, child: 'ApproxNode'):
		"""Adds a node as a child of this node and sets its parent."""
		self.children.add(child)
		child.parent.add(self)

	def add_parent(self, parent: 'ApproxNode'):
		"""Adds a node as a parent of this node and sets its child."""
		self.parent.add(parent)
		parent.children.add(self)

	@abc.abstractmethod
	def forward(self, message: Tensor = None) -> Tensor:
		"""
		Performs the forward pass for this specific node.

		Args:
			model: The transformer model.
			cache: The activation cache from a forward pass.
			message: The input to evaluate the indirect contribution for

		Returns:
			The output tensor representing the contribution of this node.
		"""
		pass

	# @abc.abstractmethod
	# def forward_with_grad(self, message: Tensor = None) -> tuple[Tensor, Tensor]:
	# 	"""
	# 	Performs the forward pass for this specific node and computes the gradient.

	# 	Args:
	# 		model: The transformer model.
	# 		cache: The activation cache from a forward pass.
	# 		message: The input to evaluate the indirect contribution for

	# 	Returns:
	# 		A tuple containing the output tensor and the gradient tensor.
	# 	"""
	# 	pass

	@abc.abstractmethod
	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, include_head: bool = False) -> list['ApproxNode']:
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
		type_order = {EMBED_ApproxNode: 0, ATTN_ApproxNode: 1, MLP_ApproxNode: 2, FINAL_ApproxNode: 3}
		node_type = type(self)
		layer = self.layer if self.layer is not None else -1
		pos = self.position if self.position is not None else -1
		keyvalue_position = getattr(self, 'keyvalue_position', -1)
		head = getattr(self, 'head', None)
		head = head if head is not None else -1

		return (
			layer,
			pos,
			type_order.get(node_type, 99),
			keyvalue_position,
			head
		)

	def __lt__(self, other):
		if not isinstance(other, ApproxNode):
			return NotImplemented
		return self._get_sort_key() < other._get_sort_key()

	def __eq__(self, other):
		if not isinstance(other, ApproxNode):
			return NotImplemented
		if (self.layer != other.layer or self.position != other.position or type(self) is not type(other)):
			return False
		if isinstance(self, ATTN_ApproxNode) and isinstance(other, ATTN_ApproxNode):
			return self.head == other.head and self.position == other.position and self.patched_keyvalue_position == other.keyvalue_position and self.patch_keyvalue == other.patch_keyvalue and self.patch_query == other.patch_query
		return True

	def __hash__(self):
		head_val = getattr(self, 'head', None)
		return hash((type(self).__name__, self.layer, self.position, head_val))

class MLP_ApproxNode(ApproxNode):
	"""Represents an MLP node in the transformer."""
	def __init__(self, model: HookedTransformer, layer: int, position: int = None, parent = ApproxNode, children = set(), msg_cache = {}, grad_cache = {}):
		super().__init__(model=model, layer=layer, position=position, parent=parent, children=children, msg_cache=msg_cache, grad_cache=grad_cache)
		self.input_name = f"blocks.{layer}.hook_resid_mid"
		self.output_name = f"blocks.{layer}.hook_mlp_out"
	

	def forward(self, message: Tensor) -> Tensor:
		if message is None:
			if self.position is None:
				return self.msg_cache[self.output_name].detach().clone()
			else:
				return self.msg_cache[self.output_name][:, self.position, :].detach().clone()
		else:
			if self.position is None:
				residual = self.msg_cache[self.input_name].detach().clone() - message
				residual = self.model.blocks[self.layer].ln2(residual)
				return self.msg_cache[self.output_name].detach().clone() - self.model.blocks[self.layer].mlp.forward(residual, message)
			else:
				residual = self.msg_cache[self.input_name][:, self.position, :].detach().clone() - message[:, self.position, :]
				residual = self.model.blocks[self.layer].ln2(residual)
				return self.msg_cache[self.output_name][:, self.position, :].detach().clone() - self.model.blocks[self.layer].mlp.forward(residual, message)

	# def forward_with_grad(self, message: Tensor) -> tuple[Tensor, Tensor]:
	# 	if message is None:
	# 		if self.position is None:
	# 			return self.msg_cache[self.output_name].detach().clone(), self.grad_cache[self.output_name].detach().clone()
	# 		else:
	# 			return self.msg_cache[self.output_name][:, self.position, :].detach().clone(), self.grad_cache[self.output_name][:, self.position, :].detach().clone()
	# 	else:
	# 		message.requires_grad_(True)
	# 		with torch.enable_grad():
	# 			if self.position is None:
	# 				residual = self.msg_cache[self.input_name].detach().clone() - message
	# 				output = self.msg_cache[self.output_name].detach().clone()
	# 			else:
	# 				residual = self.msg_cache[self.input_name][:, self.position, :].detach().clone() - message[:, self.position, :]
	# 				output = self.msg_cache[self.output_name][:, self.position, :].detach().clone()
	# 			normalized_residual = self.model.blocks[self.layer].ln2(residual)
	# 			output = output - self.model.blocks[self.layer].mlp.forward_with_grad(normalized_residual, message)
	# 			grad = torch.autograd.grad(outputs=output, inputs=message, grad_outputs=torch.ones_like(output))[0]
	# 			return output, grad


	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, include_head: bool = False, include_bos: bool = True) -> list[ApproxNode]:
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
			prev_nodes.append(MLP_ApproxNode(layer=l, position=self.position))
			for p in positions_to_iterate:
				if include_head:
					prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=p, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=p, patch_keyvalue=True, patch_query=False))
			if include_head:
				prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True) for h in range(model_cfg.n_heads)])
			else:
				prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))
		# ATTN nodes from current layer
		for p in positions_to_iterate:
			if include_head:
				prev_nodes.extend([ATTN_ApproxNode(layer=self.layer, head=h, position=self.position, keyvalue_position=p, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
			else:
				prev_nodes.append(ATTN_ApproxNode(layer=self.layer, head=None, position=self.position, keyvalue_position=p, patch_keyvalue=True, patch_query=False))
		if include_head:
			prev_nodes.extend([ATTN_ApproxNode(layer=self.layer, head=h, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True) for h in range(model_cfg.n_heads)])
		else:
			prev_nodes.append(ATTN_ApproxNode(layer=self.layer, head=None, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))
		# EMBED node
		prev_nodes.append(EMBED_ApproxNode(layer=0, position=self.position))
		# Remove duplicates
		prev_nodes = list(set(prev_nodes))
		return prev_nodes

	def __repr__(self):
		return f"MLP_ApproxNode(layer={self.layer}, position={self.position})"

	def __hash__(self):
		return hash((type(self).__name__, self.layer, self.position))

class ATTN_ApproxNode(ApproxNode):
	"""Represents an Attention node (potentially a specific head) in the transformer."""
	def __init__(self, model: HookedTransformer, layer: int, head: int = None, position: int = None, patched_keyvalue_position: int = None, parent= ApproxNode, children = set(), msg_cache = {}, grad_cache = {}, patch_query: bool = True, patch_keyvalue: bool = True):
		super().__init__(model=model, layer=layer, position=position, parent=parent, children=children, msg_cache=msg_cache, grad_cache=grad_cache)
		self.head = head
		self.patched_keyvalue_position = patched_keyvalue_position
		self.patch_keyvalue = patch_keyvalue
		self.patch_query = patch_query
		self.input_name = f"blocks.{layer}.hook_resid_pre"
		self.output_name = f"blocks.{layer}.head.{head}.hook_out" if head is not None else f"blocks.{layer}.hook_attn_out"
		self.attn_scores = f"blocks.{layer}.attn.hook_attn_scores"

		if self.position is not None and self.patched_keyvalue_position is not None:
			assert self.position >= self.patched_keyvalue_position, "query position must be greater than or equal to keyvalue position"

	def forward(self, message: Tensor) -> Tensor:
		length = self.position+1 if self.position is not None else self.msg_cache[self.input_name].shape[1]
		if message is None:
			if self.output_name in self.msg_cache:
				if self.position is None:
					return self.msg_cache[self.output_name].detach().clone()
				else:
					return self.msg_cache[self.output_name][:, self.position, :].detach().clone()
			else:
				if self.position is None:
					query_residual = self.msg_cache[self.input_name].detach().clone()
				else:
					query_residual = self.msg_cache[self.input_name][:, self.position, :].detach().clone().unsqueeze(1)
				if self.patched_keyvalue_position is None:
					key_residual = self.msg_cache[self.input_name][:, :length].detach().clone()
				else:
					key_residual = self.msg_cache[self.input_name][:, self.patched_keyvalue_position, :].detach().clone().unsqueeze(1)
		else:
			if self.patch_query:
				if self.position is None:
					query_residual = self.msg_cache[self.input_name].detach().clone() - message
				else:
					query_residual = self.msg_cache[self.input_name][:, self.position, :].detach().clone() - message[:, self.position, :]
					query_residual = query_residual.unsqueeze(1)
			else:
				if self.position is None:
					query_residual = self.msg_cache[self.input_name].detach().clone()
				else:
					query_residual = self.msg_cache[self.input_name][:, self.position, :].detach().clone().unsqueeze(1)
			
			if self.patch_keyvalue:
				if self.patched_keyvalue_position is None:
					key_residual = self.msg_cache[self.input_name][:,:length].detach().clone() - message[:,:length]
				else:
					key_residual = self.msg_cache[self.input_name][:, self.patched_keyvalue_position, :].detach().clone() - message[:, self.patched_keyvalue_position, :]
					key_residual = key_residual.unsqueeze(1)
			else:
				if self.patched_keyvalue_position is None:
					key_residual = self.msg_cache[self.input_name][:,:length].detach().clone()
				else:
					key_residual = self.msg_cache[self.input_name][:, self.patched_keyvalue_position, :].detach().clone()
					key_residual = key_residual

		key_residual = self.model.blocks[self.layer].ln1(key_residual)
		value_residual = self.model.blocks[self.layer].ln1(self.msg_cache[self.input_name].detach().clone())
		query_residual = self.model.blocks[self.layer].ln1(query_residual)
		if self.head is not None:
			W_Q = self.model.blocks[self.layer].attn.W_Q[self.head].unsqueeze(0)
			W_K = self.model.blocks[self.layer].attn.W_K[self.head].unsqueeze(0)
			W_V = self.model.blocks[self.layer].attn.W_V[self.head].unsqueeze(0)
			b_Q = self.model.blocks[self.layer].attn.b_Q[self.head].unsqueeze(0)
			b_K = self.model.blocks[self.layer].attn.b_K[self.head].unsqueeze(0)
			b_V = self.model.blocks[self.layer].attn.b_V[self.head].unsqueeze(0)
			query = torch.einsum('bsd,ndh->bsnh', query_residual, W_Q) + b_Q[None, None, :, :]
			key = torch.einsum('bsd,ndh->bsnh', key_residual, W_K) + b_K[None, None, :, :]
			value = torch.einsum('bsd,ndh->bsnh', value_residual, W_V) + b_V[None, None, :, :]
		else:
			W_Q = self.model.blocks[self.layer].attn.W_Q
			W_K = self.model.blocks[self.layer].attn.W_K
			W_V = self.model.blocks[self.layer].attn.W_V
			b_Q = self.model.blocks[self.layer].attn.b_Q
			b_K = self.model.blocks[self.layer].attn.b_K
			b_V = self.model.blocks[self.layer].attn.b_V
			query = torch.einsum('bsd,ndh->bsnh', query_residual, W_Q) + b_Q[None, None, :, :]
			key = torch.einsum('bsd,ndh->bsnh', key_residual, W_K) + b_K[None, None, :, :]
			value = torch.einsum('bsd,ndh->bsnh', value_residual, W_V) + b_V[None, None, :, :]
		out = custom_attention_forward(
			attention_module=self.model.blocks[self.layer].attn,
			head=self.head,
			q=query,
			k=key,
			v=value,
			precomputed_attention_scores=self.msg_cache.get(self.attn_scores, None).detach().clone(),
			query_position=self.position,
			keyvalue_position=self.patched_keyvalue_position,
		)
		
		if message is None:
			self.msg_cache[self.output_name] = out
			return out
		if self.msg_cache.get(self.output_name, None) is None:
			ATTN_ApproxNode(self.model, layer=self.layer, head=self.head, msg_cache=self.msg_cache, grad_cache=self.grad_cache).forward(message=None)
		
		if self.position is not None:
			return self.msg_cache[self.output_name][:, self.position] - out
		return self.msg_cache[self.output_name] - out


	# def forward_with_grad(self, message: Tensor = None) -> tuple[Tensor, Tensor]:
	# 	"""
	# 	Performs the forward pass for this specific node and computes the gradient.

	# 	Args:
	# 		model: The transformer model.
	# 		cache: The activation cache from a forward pass.
	# 		message: The input to evaluate the indirect contribution for

	# 	Returns:
	# 		A tuple containing the output tensor and the gradient tensor.
	# 	"""
	# 	pass

	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, include_head: bool = False, include_bos: bool = True) -> list[ApproxNode]:
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
				prev_nodes.append(MLP_ApproxNode(layer=l, position=self.position))
			if self.patch_keyvalue and (not self.patch_query or self.position != self.patched_keyvalue_position): # Note that if self.position is None than also self.patched_keyvalue_position is None
				prev_nodes.append(MLP_ApproxNode(layer=l, position=self.patched_keyvalue_position))    
	
 		# EMBED node
		if self.patch_query:
			prev_nodes.append(EMBED_ApproxNode(layer=0, position=self.position))
		if self.patch_keyvalue and (not self.patch_query or self.position != self.patched_keyvalue_position):
			prev_nodes.append(EMBED_ApproxNode(layer=0, position=self.patched_keyvalue_position))
   
		# ATTN nodes patching current query position
		if self.patch_query:
			for l in range(self.layer):
				# prev ATTN query positions
				if include_head:
					prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))
				
				# prev ATTN key-value positions
				if self.position is not None:
					for keyvalue_position in range(start_pos, self.position + 1):
						if include_head:
							prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
						else:
							prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_keyvalue=True, patch_query=False))
				else:
					if include_head:
						prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=None, keyvalue_position=None, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
					else:
						prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=None, keyvalue_position=None, patch_keyvalue=True, patch_query=False))

		# ATTN nodes patching current key-value position
		if self.patch_keyvalue:
			keyvalue_positions = range(start_pos, self.patched_keyvalue_position+1) if self.patched_keyvalue_position is not None else [None]
			for l in range(self.layer):
				# prev ATTN query positions
				if include_head:
					prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.patched_keyvalue_position, keyvalue_position=None, patch_keyvalue=False, patch_query=True) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.patched_keyvalue_position, keyvalue_position=None, patch_keyvalue=False, patch_query=True))

				# prev ATTN key-value positions				
				for prev_keyvalue_position in keyvalue_positions:
					if include_head:
						prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.patched_keyvalue_position, keyvalue_position=prev_keyvalue_position, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
					else:
						prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.patched_keyvalue_position, keyvalue_position=prev_keyvalue_position, patch_keyvalue=True, patch_query=False))

		# Remove duplicates
		prev_nodes = list(set(prev_nodes))
		return prev_nodes

	def __repr__(self):
		return f"ATTN_ApproxNode(layer={self.layer}, head={self.head}, position={self.position}, keyvalue_position={self.patched_keyvalue_position}, patch_query={self.patch_query}, patch_keyvalue={self.patch_keyvalue})"

	def __hash__(self):
		return hash((type(self).__name__, self.layer, self.head, self.position, self.patched_keyvalue_position, self.patch_query, self.patch_keyvalue))


class EMBED_ApproxNode(ApproxNode):
	"""Represents the embedding node in the transformer."""
	def __init__(self, model: HookedTransformer, layer: int = 0, position: int = None, parent= ApproxNode, children = set(), msg_cache = {}, grad_cache = {}):
		super().__init__(model=model, layer=layer, position=position, parent=parent, children=children, msg_cache=msg_cache, grad_cache=grad_cache)
		self.input_residual = "hook_embed"

	def forward(self, message: Tensor = None) -> Tensor:
		embedding = self.msg_cache["hook_embed"].clone() #+ cache["hook_pos_embed"].clone()
		if message is not None:
			embedding = embedding - message
		if self.position is not None:
			embedding[:, :self.position, :] = torch.zeros_like(embedding[:, :self.position, :])
			embedding[:, self.position + 1:, :] = torch.zeros_like(embedding[:, self.position + 1:, :])
		return embedding


	def get_prev_nodes(self, model_cfg: HookedTransformerConfig, sequence_length: int, include_head: bool = False, include_bos: bool = True) -> list[ApproxNode]:
		return []

	def __repr__(self):
		return f"EMBED_ApproxNode(layer={self.layer}, position={self.position})"

	def __hash__(self):
		return hash((type(self).__name__, self.layer, self.position))

class FINAL_ApproxNode(ApproxNode):
	"""Represents the final node in the transformer."""
	def __init__(self, model: HookedTransformer, layer: int, position: Optional[int] = None, parent= ApproxNode, children = set(), msg_cache = {}, grad_cache = {}):
		super().__init__(model=model, layer=layer, position=position, parent=parent, children=children, msg_cache=msg_cache, grad_cache=grad_cache)
		self.input_residual = f"blocks.{layer}.hook_resid_post"
		self.output_name = f"blocks.{layer}.hook_output"

	def forward(self, message: Tensor = None) -> Tensor:
		res = self.msg_cache[self.input_residual].clone()
		if message is not None:
			res = message
		if self.position is not None:
			res_zeroed = torch.zeros_like(res)
			res_zeroed[:, self.position, :] = res[:, self.position, :]
			res = res_zeroed
		return res

	def get_prev_nodes(self, model_cfg: HookedTransformerConfig, include_head: bool = True, include_bos: bool = True) -> list[ApproxNode]:
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
			prev_nodes.append(MLP_ApproxNode(layer=l, position=self.position))
			
			# ATTN query positions
			if include_head:
				prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, keyvalue_position=None, position=self.position, patch_keyvalue=False, patch_query=True) for h in range(model_cfg.n_heads)])
			else:
				prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, keyvalue_position=None, position=self.position, patch_keyvalue=False, patch_query=True))
			
			# ATTN key-value positions
			if self.position is not None:
				for keyvalue_position in range(self.position + 1):
					if include_head:
						prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
					else:
						prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_keyvalue=True, patch_query=False))
			else:
				if include_head:
					prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=None, keyvalue_position=None, patch_keyvalue=True, patch_query=False) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=None, patch_keyvalue=True, patch_query=False))
		
		prev_nodes.append(EMBED_ApproxNode(layer=0, position=self.position))
		# Remove duplicates
		prev_nodes = list(set(prev_nodes))
		return prev_nodes

	def __repr__(self):
		pos_str = f", position={self.position}" if self.position is not None else ""
		return f"FINAL_ApproxNode(layer={self.layer}{pos_str})"

	def __hash__(self):
		return hash((type(self).__name__, self.layer, self.position)) 