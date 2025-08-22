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
	def __init__(self, model: HookedTransformer, layer: int, input_name: str, output_name: str, position: int = None, msg_cache: dict = {}, cf_cache: dict = {}, parent = None, children = set(), gradient = None, patch_type = 'zero'):
		self.model = model
		self.layer = layer
		self.position = position
		self.parent= parent
		self.children = children
		self.msg_cache = msg_cache
		self.cf_cache = cf_cache
		self.gradient = gradient
		self.input_name = input_name
		self.output_name = output_name
		self.patch_type = patch_type
		if patch_type not in ['zero', 'counterfactual']:
			raise ValueError(f"Unknown patch type: {patch_type}")

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
	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, include_head: bool = False, separate_kv: bool = False) -> list['ApproxNode']:
		"""
		Returns a list of *potential* previous nodes in the computational graph
		that contribute to this node. These are not automatically set as the parent.

		Args:
			model_cfg: The configuration of the transformer model.
			include_head: Whether to consider specific head nodes for ATTN.
		Returns:
			A list of potential previous nodes.
		"""
		pass

	@abc.abstractmethod
	def calculate_gradient(self, grad_outputs=None, save=True, use_precomputed=False) -> Tensor:
		"""
		Returns the gradient of this node with respect to the output of the model.
		Returns:
			A tensor representing the gradient of this node.
		"""


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
			return self.head == other.head and self.position == other.position and self.keyvalue_position == other.keyvalue_position and self.patch_key == other.patch_key and self.patch_query == other.patch_query and self.patch_value == other.patch_value
		return True
		

	def __hash__(self):
		head_val = getattr(self, 'head', None)
		return hash((type(self).__name__, self.layer, self.position, head_val))

class MLP_ApproxNode(ApproxNode):
	"""Represents an MLP node in the transformer."""
	def __init__(self, model: HookedTransformer, layer: int, position: int = None, parent: ApproxNode = None, children = set(), msg_cache = {}, cf_cache = {}, gradient = None, patch_type = 'zero'):
		super().__init__(model=model, layer=layer, position=position, parent=parent, children=children, msg_cache=msg_cache, cf_cache=cf_cache, gradient=gradient, input_name=f"blocks.{layer}.hook_resid_mid", output_name=f"blocks.{layer}.hook_mlp_out", patch_type=patch_type)
	

	def forward(self, message: Tensor) -> Tensor:
		if message is None:
			if self.patch_type == 'zero':
				if self.position is None:
					return self.msg_cache[self.output_name].detach().clone()
				else:
					out = torch.zeros_like(self.msg_cache[self.input_name], device=self.msg_cache[self.input_name].device)
					out[:, self.position, :] = self.msg_cache[self.output_name][:, self.position, :].detach().clone()
					return out
			elif self.patch_type == 'counterfactual':
				if self.position is None:
					return self.msg_cache[self.output_name].detach().clone() - self.cf_cache[self.output_name].detach().clone()
				else:
					out = torch.zeros_like(self.msg_cache[self.input_name], device=self.msg_cache[self.input_name].device)
					out[:, self.position, :] =  self.msg_cache[self.output_name][:, self.position, :].detach().clone() - self.cf_cache[self.output_name][:, self.position, :].detach().clone()
					return out
		else:
			if self.position is None:
				residual = self.msg_cache[self.input_name].detach().clone() - message
				residual = self.model.blocks[self.layer].ln2(residual)
				return self.msg_cache[self.output_name].detach().clone() - self.model.blocks[self.layer].mlp.forward(residual)
			else:
				residual = self.msg_cache[self.input_name][:, self.position, :].detach().clone() - message[:, self.position, :]
				residual = self.model.blocks[self.layer].ln2(residual)
				out = torch.zeros_like(self.msg_cache[self.input_name], device=self.msg_cache[self.input_name].device)
				out[:, self.position, :] = self.msg_cache[self.output_name][:, self.position, :].detach().clone() - self.model.blocks[self.layer].mlp.forward(residual)
				return out


	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, include_head: bool = False, separate_kv: bool = False) -> list[ApproxNode]:
		"""Returns a list of potential previous nodes that contribute to this MLP node.
		Previous nodes are:
			- MLP, EMBED and ATTN nodes in self.position from previous layers.
			- ATTN nodes in all previous positions from current layers.
		ATTN nodes are patched both in query and key-value positions separately.
		Args:
			model_cfg: The configuration of the transformer model.
			include_head: Whether to consider specific head nodes for ATTN.
		Returns:
			A list of potential previous nodes.
		"""
		prev_nodes = []
		common_args = {"model": self.model, "msg_cache": self.msg_cache, "cf_cache": self.cf_cache, "parent": self, "patch_type": self.patch_type}
		if self.position is not None:
			positions_to_iterate = range(self.position + 1)
		else:
			positions_to_iterate = [None]

		# MLP and ATTN nodes from previous layers
		for l in range(self.layer):
			prev_nodes.append(MLP_ApproxNode(layer=l, position=self.position, **common_args))
			for p in positions_to_iterate:
				if include_head:
					if separate_kv:
						prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=p, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
						prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=p, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
					else:
						prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=p, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
				else:
					if separate_kv:
						prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=p, patch_key=True, patch_value=False, patch_query=False, **common_args))
						prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=p, patch_key=False, patch_value=True, patch_query=False, **common_args))
					else:
						prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=p, patch_key=True, patch_value=True, patch_query=False, **common_args))
			if include_head:
				prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args) for h in range(model_cfg.n_heads)])
			else:
				prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args))

		# ATTN nodes from current layer
		for p in positions_to_iterate:
			if include_head:
				if separate_kv:
					prev_nodes.extend([ATTN_ApproxNode(layer=self.layer, head=h, position=self.position, keyvalue_position=p, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
					prev_nodes.extend([ATTN_ApproxNode(layer=self.layer, head=h, position=self.position, keyvalue_position=p, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.extend([ATTN_ApproxNode(layer=self.layer, head=h, position=self.position, keyvalue_position=p, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
			else:
				if separate_kv:
					prev_nodes.append(ATTN_ApproxNode(layer=self.layer, head=None, position=self.position, keyvalue_position=p, patch_key=True, patch_value=False, patch_query=False, **common_args))
					prev_nodes.append(ATTN_ApproxNode(layer=self.layer, head=None, position=self.position, keyvalue_position=p, patch_key=False, patch_value=True, patch_query=False, **common_args))
				else:
					prev_nodes.append(ATTN_ApproxNode(layer=self.layer, head=None, position=self.position, keyvalue_position=p, patch_key=True, patch_value=True, patch_query=False, **common_args))
		if include_head:
			prev_nodes.extend([ATTN_ApproxNode(layer=self.layer, head=h, position=self.position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args) for h in range(model_cfg.n_heads)])
		else:
			prev_nodes.append(ATTN_ApproxNode(layer=self.layer, head=None, position=self.position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args))

		# EMBED node
		prev_nodes.append(EMBED_ApproxNode(layer=0, position=self.position, **common_args))
		# Remove duplicates
		prev_nodes = list(set(prev_nodes))
		return prev_nodes

	def __repr__(self):
		return f"MLP_ApproxNode(layer={self.layer}, position={self.position})"

	def __hash__(self):
		return hash((type(self).__name__, self.layer, self.position))
	
	def calculate_gradient(self, grad_outputs=None, save=True, use_precomputed=False) -> Tensor:
		if self.gradient is not None and use_precomputed:
			if self.position is None:
				return self.gradient.detach().clone()
			gradient = self.gradient.detach().clone()
			out = torch.zeros_like(self.msg_cache[self.input_name], device=gradient.device)
			out[:, self.position, :] = gradient
			return out

		input_residual = self.msg_cache[self.input_name].detach().clone()
		if self.position is not None:
			input_residual = input_residual[:, self.position, :]
		input_residual.requires_grad_(True)

		with torch.enable_grad():
			norm_res = self.model.blocks[self.layer].ln2(input_residual)
			output = self.model.blocks[self.layer].mlp.forward(norm_res)
		
		if grad_outputs is None:
			grad_outputs = self.parent.calculate_gradient(save=True, use_precomputed=True) if self.parent is not None else torch.ones_like(input_residual)
	
		if input_residual.shape != grad_outputs.shape:
			grad_outputs = grad_outputs[:, self.position, :]
		gradient = torch.autograd.grad(
			output,
			input_residual,
			grad_outputs=grad_outputs,
		)[0]
		if save:
			self.gradient = gradient.detach().clone()
		if self.position is not None:
			out = torch.zeros_like(self.msg_cache[self.input_name], device=gradient.device)
			out[:, self.position, :] = gradient.detach().clone()
			return out
		return gradient.detach().clone()
	

class ATTN_ApproxNode(ApproxNode):
	"""Represents an Attention node (potentially a specific head) in the transformer."""
	def __init__(self, model: HookedTransformer, layer: int, head: int = None, position: int = None, keyvalue_position: int = None, parent: ApproxNode = None, children = set(), msg_cache = {}, cf_cache = {}, gradient = None, patch_query: bool = True, patch_key: bool = True, patch_value: bool = True, plot_patterns: bool = False, patch_type = 'zero'):
		super().__init__(model=model, layer=layer, position=position, parent=parent, children=children, msg_cache=msg_cache, cf_cache=cf_cache, gradient=gradient, patch_type=patch_type, input_name=f"blocks.{layer}.hook_resid_pre", output_name="")
		self.head = head
		self.keyvalue_position = keyvalue_position
		self.patch_key = patch_key
		self.patch_value = patch_value
		self.patch_query = patch_query
		output_name = f"blocks.{layer}.head.{head}" if head is not None else f"blocks.{layer}"
		output_name += ".hook_attn_out" if keyvalue_position is None else f".kv.{keyvalue_position}.hook_attn_out"
		self.output_name = output_name
		self.attn_scores = f"blocks.{layer}.attn.hook_attn_scores"
		self.plot_patterns = plot_patterns

		if self.position is not None and self.keyvalue_position is not None:
			assert self.position >= self.keyvalue_position, "query position must be greater than or equal to keyvalue position"
		if msg_cache.get(self.output_name, None) is not None:
			assert msg_cache[self.input_name].shape == msg_cache[self.output_name].shape, "Input and output shapes must match"

	def forward(self, message: Tensor) -> Tensor:
		length = self.position+1 if self.position is not None else self.msg_cache[self.input_name].shape[1]
		value_residual = self.msg_cache[self.input_name].detach().clone()
		if message is None:
			if self.patch_type == 'zero':
				if self.output_name in self.msg_cache:
					if self.position is None:
						return self.msg_cache[self.output_name].detach().clone()
					else:
						out = torch.zeros_like(self.msg_cache[self.input_name])
						out[:, self.position, :] = self.msg_cache[self.output_name][:, self.position, :].detach().clone()
						return out
				else:
					if self.position is None:
						query_residual = self.msg_cache[self.input_name].detach().clone()
					else:
						query_residual = self.msg_cache[self.input_name][:, self.position, :].detach().clone().unsqueeze(1)
					if self.keyvalue_position is None:
						key_residual = self.msg_cache[self.input_name][:, :length].detach().clone()
					else:
						key_residual = self.msg_cache[self.input_name][:, self.keyvalue_position, :].detach().clone().unsqueeze(1)
			if self.patch_type == 'counterfactual':
				value_residual = self.cf_cache[self.input_name].detach().clone()
				if self.output_name in self.cf_cache and self.output_name in self.msg_cache:
					if self.position is None:
						return self.msg_cache[self.output_name].detach().clone() - self.cf_cache[self.output_name].detach().clone()
					else:
						out = torch.zeros_like(self.msg_cache[self.input_name])
						out[:, self.position, :] = self.msg_cache[self.output_name][:, self.position, :].detach().clone() - self.cf_cache[self.output_name][:, self.position, :].detach().clone()
						return out
				else:
					if self.position is None:
						query_residual = self.cf_cache[self.input_name].detach().clone()
					else:
						query_residual = self.cf_cache[self.input_name][:, self.position, :].detach().clone().unsqueeze(1)
					if self.keyvalue_position is None:
						key_residual = self.cf_cache[self.input_name][:, :length].detach().clone()
					else:
						key_residual = self.cf_cache[self.input_name][:, self.keyvalue_position, :].detach().clone().unsqueeze(1)
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
			
			if self.patch_key:
				if self.keyvalue_position is None:
					key_residual = self.msg_cache[self.input_name][:,:length].detach().clone() - message[:,:length]
				else:
					key_residual = self.msg_cache[self.input_name][:, self.keyvalue_position, :].detach().clone() - message[:, self.keyvalue_position, :]
					key_residual = key_residual.unsqueeze(1)
			else:
				if self.keyvalue_position is None:
					key_residual = self.msg_cache[self.input_name][:,:length].detach().clone()
				else:
					key_residual = self.msg_cache[self.input_name][:, self.keyvalue_position, :].detach().clone()
					key_residual = key_residual.unsqueeze(1)
			if self.patch_value:
				if self.keyvalue_position is None:
					value_residual = value_residual - message
				else:
					value_residual[:, self.keyvalue_position, :] = value_residual[:, self.keyvalue_position, :] - message[:, self.keyvalue_position, :]

		key_residual = self.model.blocks[self.layer].ln1(key_residual)
		value_residual = self.model.blocks[self.layer].ln1(value_residual)
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
			if self.keyvalue_position is not None:
				v = torch.einsum('bd,ndh->bnh', value_residual[:, self.keyvalue_position, :], W_V) + b_V[None, None, :, :]
				value = torch.zeros(v.shape[0], value_residual.shape[1], v.shape[2], v.shape[3], device=v.device)
				value[:, self.keyvalue_position, :] = v
			else:
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
			if self.keyvalue_position is not None:
				v = torch.einsum('bd,ndh->bnh', value_residual[:, self.keyvalue_position, :], W_V) + b_V[None, None, :, :]
				value = torch.zeros(v.shape[0], value_residual.shape[1], v.shape[2], v.shape[3], device=v.device)
				value[:, self.keyvalue_position, :] = v
			else:
				value = torch.einsum('bsd,ndh->bsnh', value_residual, W_V) + b_V[None, None, :, :]
		out = custom_attention_forward(
			attention_module=self.model.blocks[self.layer].attn,
			head=self.head,
			q=query,
			k=key,
			v=value,
			precomputed_attention_scores=self.msg_cache.get(self.attn_scores, None).detach().clone(),
			query_position=self.position,
			keyvalue_position=self.keyvalue_position,
			plot_patterns=self.plot_patterns
		)
		

		if self.msg_cache.get(self.output_name, None) is None:
			if self.position is None and message is None:
				self.msg_cache[self.output_name] = out.detach().clone()
			else:
				ATTN_ApproxNode(self.model, layer=self.layer, head=self.head, msg_cache=self.msg_cache, cf_cache=self.cf_cache, keyvalue_position=self.keyvalue_position, patch_type='zero').forward(message=None)
		if self.patch_type == 'counterfactual' and self.cf_cache.get(self.output_name, None) is None:
			if self.position is None and message is None:
				self.cf_cache[self.output_name] = out.detach().clone()
			else:
				ATTN_ApproxNode(self.model, layer=self.layer, head=self.head, msg_cache=self.cf_cache, cf_cache=self.cf_cache, keyvalue_position=self.keyvalue_position, patch_type='counterfactual').forward(message=None)
		if message is None:
			if self.patch_type == 'zero':
				if self.position is not None:
					resized_out = torch.zeros_like(self.msg_cache[self.input_name], device=out.device)
					resized_out[:, self.position, :] = out
					return resized_out
				self.msg_cache[self.output_name] = out.detach().clone()
				return out
			elif self.patch_type == 'counterfactual':
				if self.position is not None:
					resized_out = torch.zeros_like(self.msg_cache[self.input_name], device=out.device)
					resized_out[:, self.position, :] = self.msg_cache[self.output_name][:, self.position, :].detach().clone() - out
					return resized_out
				self.cf_cache[self.output_name] = out.detach().clone()
				return self.msg_cache[self.output_name].detach().clone() - self.cf_cache[self.output_name].detach().clone()
			else:
				raise ValueError(f"Unknown patch type: {self.patch_type}")
		
		if self.position is not None:
			resized_out = torch.zeros_like(self.msg_cache[self.input_name], device=out.device)
			resized_out[:, self.position, :] = self.msg_cache[self.output_name][:, self.position, :].detach().clone() - out
			return resized_out
		return self.msg_cache[self.output_name].detach().clone() - out
	
	def calculate_gradient(self, grad_outputs=None, save=True, use_precomputed=False) -> Tensor:
		if self.gradient is not None and use_precomputed:
			if self.position is None or self.keyvalue_position is None:
				return self.gradient.detach().clone()
			gradient = self.gradient.detach().clone()
			out = torch.zeros_like(gradient, device=gradient.device)
			if self.patch_query or self.patch_key:
				out[:, self.keyvalue_position, :] = gradient[:, self.keyvalue_position, :]
			if self.patch_query:
				out[:, self.position, :] = gradient[:, self.position, :]
			return out
		input_residual = self.msg_cache[self.input_name].detach().clone()
		input_residual.requires_grad_(True)

		with torch.enable_grad():
			length = self.position+1 if self.position is not None else self.msg_cache[self.input_name].shape[1]
			
			if self.position is None:
				query_residual = input_residual
			else:
				query_residual = input_residual[:, self.position, :].unsqueeze(1)
			if self.keyvalue_position is None:
				key_residual = input_residual[:, :length]
			else:
				key_residual = input_residual[:, self.keyvalue_position, :].unsqueeze(1)
			value_residual = input_residual
			if not self.patch_query:
				query_residual = query_residual.detach() # detach from gradient computation
			if not self.patch_key:
				key_residual = key_residual.detach() # detach from gradient computation
			if not self.patch_value:
				value_residual = value_residual.detach() # detach from gradient computation
			key_residual = self.model.blocks[self.layer].ln1(key_residual)
			value_residual = self.model.blocks[self.layer].ln1(value_residual)
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
				if self.keyvalue_position is not None:
					v = torch.einsum('bd,ndh->bnh', value_residual[:, self.keyvalue_position, :], W_V) + b_V[None, None, :, :]
					value = torch.zeros(v.shape[0], value_residual.shape[1], v.shape[2], v.shape[3], device=v.device)
					value[:, self.keyvalue_position, :] = v
				else:
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
				if self.keyvalue_position is not None:
					v = torch.einsum('bd,ndh->bnh', value_residual[:, self.keyvalue_position, :], W_V) + b_V[None, None, :, :]
					value = torch.zeros(v.shape[0], value_residual.shape[1], v.shape[2], v.shape[3], device=v.device)
					value[:, self.keyvalue_position, :] = v
				else:
					value = torch.einsum('bsd,ndh->bsnh', value_residual, W_V) + b_V[None, None, :, :]
			out = custom_attention_forward(
				attention_module=self.model.blocks[self.layer].attn,
				head=self.head,
				q=query,
				k=key,
				v=value,
				precomputed_attention_scores=self.msg_cache.get(self.attn_scores, None).detach().clone(),
				query_position=self.position,
				keyvalue_position=self.keyvalue_position,
				plot_patterns=self.plot_patterns
			)
			if self.position is not None:
				resized_out = torch.zeros_like(self.msg_cache[self.input_name], device=out.device)
				resized_out[:, self.position, :] = out
			else:
				resized_out = out

		if grad_outputs is None:
			grad_outputs = self.parent.calculate_gradient(save=True, use_precomputed=True) if self.parent is not None else torch.ones_like(input_residual)
		gradient = torch.autograd.grad(
			resized_out,
			input_residual,
			grad_outputs=grad_outputs,
			allow_unused=True,
		)[0]
		if save:
			self.gradient = gradient
		return gradient


	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, include_head: bool = False, separate_kv: bool = False) -> list[ApproxNode]:
		"""Returns a list of potential previous nodes that contribute to this ATTN node.
		Previous nodes are:
			- MLP, EMBED and ATTN nodes in self.position from previous layers if patch_query=True.
			- MLP, EMBED and ATTN nodes in all previous positions from previous layers if patch_key=True and patch_value=True.
		Args:
			model_cfg: The configuration of the transformer model.
			include_head: Whether to consider specific head nodes for ATTN.
		Returns:
			A list of potential previous nodes."""
		prev_nodes = []
		common_args = {"model": self.model, "msg_cache": self.msg_cache, "parent": self, "patch_type": self.patch_type, "cf_cache": self.cf_cache}

		# MLPs
		for l in range(self.layer):
			if self.patch_query:
				prev_nodes.append(MLP_ApproxNode(layer=l, position=self.position, **common_args))
			if (self.patch_key or self.patch_value) and (not self.patch_query or self.position != self.keyvalue_position):
				prev_nodes.append(MLP_ApproxNode(layer=l, position=self.keyvalue_position, **common_args))

		# EMBED node
		if self.patch_query:
			prev_nodes.append(EMBED_ApproxNode(layer=0, position=self.position, **common_args))
		if (self.patch_key or self.patch_value) and (not self.patch_query or self.position != self.keyvalue_position):
			prev_nodes.append(EMBED_ApproxNode(layer=0, position=self.keyvalue_position, **common_args))

		# ATTN nodes patching current query position
		if self.patch_query:
			for l in range(self.layer):
				# prev ATTN query positions
				if include_head:
					prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args))

				# prev ATTN key-value positions
				if self.position is not None:
					for keyvalue_position in range(self.position + 1):
						if include_head:
							if separate_kv:
								prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
								prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
							else:
								prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
						else:
							if separate_kv:
								prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=False, patch_query=False, **common_args))
								prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_key=False, patch_value=True, patch_query=False, **common_args))
							else:
								prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=True, patch_query=False, **common_args))
				else:
					if include_head:
						if separate_kv:
							prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=None, keyvalue_position=None, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
							prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=None, keyvalue_position=None, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
						else:
							prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=None, keyvalue_position=None, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
					else:
						if separate_kv:
							prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=None, keyvalue_position=None, patch_key=True, patch_value=False, patch_query=False, **common_args))
							prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=None, keyvalue_position=None, patch_key=False, patch_value=True, patch_query=False, **common_args))
						else:
							prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=None, keyvalue_position=None, patch_key=True, patch_value=True, patch_query=False, **common_args))

		# ATTN nodes patching current key-value position
		if self.patch_key or self.patch_value:
			keyvalue_positions = range(self.keyvalue_position + 1) if self.keyvalue_position is not None else [None]
			for l in range(self.layer):
				# prev ATTN query positions
				if include_head:
					prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.keyvalue_position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.keyvalue_position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args))

				# prev ATTN key-value positions
				for prev_keyvalue_position in keyvalue_positions:
					if include_head:
						if separate_kv:
							prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.keyvalue_position, keyvalue_position=prev_keyvalue_position, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
							prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.keyvalue_position, keyvalue_position=prev_keyvalue_position, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
						else:
							prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.keyvalue_position, keyvalue_position=prev_keyvalue_position, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
					else:
						if separate_kv:
							prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.keyvalue_position, keyvalue_position=prev_keyvalue_position, patch_key=True, patch_value=False, patch_query=False, **common_args))
							prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.keyvalue_position, keyvalue_position=prev_keyvalue_position, patch_key=False, patch_value=True, patch_query=False, **common_args))
						else:
							prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.keyvalue_position, keyvalue_position=prev_keyvalue_position, patch_key=True, patch_value=True, patch_query=False, **common_args))

		# Remove duplicates
		prev_nodes = list(set(prev_nodes))
		return prev_nodes

	def __repr__(self):
		return f"ATTN_ApproxNode(layer={self.layer}, head={self.head}, position={self.position}, keyvalue_position={self.keyvalue_position}, patch_query={self.patch_query}, patch_key={self.patch_key}, patch_value={self.patch_value})"

	def __hash__(self):
		return hash((type(self).__name__, self.layer, self.head, self.position, self.keyvalue_position, self.patch_query, self.patch_key, self.patch_value))


class EMBED_ApproxNode(ApproxNode):
	"""Represents the embedding node in the transformer."""
	def __init__(self, model: HookedTransformer, layer: int = 0, position: int = None, parent: ApproxNode = None, children = set(), msg_cache = {}, cf_cache = {}, gradient = None, patch_type = 'zero'):
		super().__init__(model=model, layer=layer, position=position, parent=parent, children=children, msg_cache=msg_cache, cf_cache=cf_cache, gradient=gradient, input_name="hook_embed", output_name="hook_embed", patch_type=patch_type)

	def forward(self, message: Tensor = None) -> Tensor:
		if message is None:
			if self.patch_type == 'zero':
				embedding = self.msg_cache["hook_embed"].detach().clone()
			elif self.patch_type == 'counterfactual':
				embedding = self.msg_cache["hook_embed"].detach().clone() - self.cf_cache["hook_embed"].detach().clone()
			else:
				raise ValueError(f"Unknown patch type: {self.patch_type}")
		else:
			embedding = self.msg_cache["hook_embed"].detach().clone() - message
		if self.position is not None:
			embedding[:, :self.position, :] = torch.zeros_like(embedding[:, :self.position, :], device=embedding.device)
			embedding[:, self.position + 1:, :] = torch.zeros_like(embedding[:, self.position + 1:, :], device=embedding.device)
		return embedding

	def calculate_gradient(self, grad_outputs=None, save=True, use_precomputed=False):
		if self.gradient is not None and use_precomputed:
			if self.position is None:
				return self.gradient.detach().clone()
			gradient = self.gradient.detach().clone()
			out = torch.zeros_like(self.msg_cache[self.input_name], device=gradient.device)
			out[:, self.position, :] = gradient[:, self.position, :]
			return out
		if grad_outputs is None:
			gradient = self.parent.calculate_gradient(grad_outputs, save=True, use_precomputed=True) if self.parent is not None else torch.ones_like(self.msg_cache[self.input_name])
		else:
			gradient = self.parent.calculate_gradient(grad_outputs, save=True, use_precomputed=False) if self.parent is not None else torch.ones_like(self.msg_cache[self.input_name])
		gradient[:, :self.position, :] = torch.zeros_like(gradient[:, :self.position, :], device=gradient.device)
		gradient[:, self.position + 1:, :] = torch.zeros_like(gradient[:, self.position + 1:, :], device=gradient.device)
		if save:
			if self.position is None:
				self.gradient = gradient.detach().clone()
			else:
				self.gradient = gradient[:, self.position, :].detach().clone()
		return gradient.detach().clone()


	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, sequence_length: int, include_head: bool = False, separate_kv: bool = False) -> list[ApproxNode]:
		return []

	def __repr__(self):
		return f"EMBED_ApproxNode(layer={self.layer}, position={self.position})"

	def __hash__(self):
		return hash((type(self).__name__, self.layer, self.position))


class FINAL_ApproxNode(ApproxNode):
	"""Represents the final node in the transformer (This is a dummy node)."""
	def __init__(self, model: HookedTransformer, layer: int, metric: callable = None, position: Optional[int] = None, parent: ApproxNode = None, children = set(), msg_cache = {}, cf_cache = {}, gradient = None, patch_type = 'zero'):
		super().__init__(model=model, layer=layer, position=position, parent=parent, children=children, msg_cache=msg_cache, cf_cache=cf_cache, gradient=gradient, input_name=f"blocks.{layer}.hook_resid_post", output_name=f"blocks.{layer}.hook_resid_post", patch_type=patch_type)
		self.metric = metric
	def forward(self, message: Tensor = None) -> Tensor:
		if message is None:
			if self.patch_type == 'zero':
				res = self.msg_cache[self.input_name].detach().clone()
			elif self.patch_type == 'counterfactual':
				res = self.msg_cache[self.input_name].detach().clone() - self.cf_cache[self.input_name].detach().clone()
			else:
				raise ValueError(f"Unknown patch type: {self.patch_type}")
		else:
			res = message.detach().clone()
		if self.position is not None:
			res_zeroed = torch.zeros_like(res, device=res.device)
			res_zeroed[:, self.position, :] = res[:, self.position, :]
			return res_zeroed
		return res

	def calculate_gradient(self, grad_outputs=None, save=True, use_precomputed=False, metric=None) -> Tensor:
		if self.gradient is not None and use_precomputed:
			return self.gradient.detach().clone()
		if metric is None:
			metric = self.metric
		if metric is None:
			raise NotImplementedError("FINAL_ApproxNode.calculate_gradient() requires to provide a metric either at initialization or as a parameter")
		input_residual = self.msg_cache[self.output_name].detach().clone()
		input_residual.requires_grad_(True)
		with torch.enable_grad():
			output = metric(corrupted_resid=input_residual)

		gradient = torch.autograd.grad(
			output,
			input_residual,
			allow_unused=True
		)[0]
		
		if save:
			self.gradient = -gradient
		return -gradient

	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, include_head: bool = True, separate_kv: bool = False) -> list[ApproxNode]:
		"""
		Returns a list of potential previous nodes that contribute to this FINAL node.
		Previous nodes are:
			- MLP, EMBED and ATTN nodes in self.position from all layers.
		Args:
			model_cfg: The configuration of the transformer model.
			include_head: Whether to consider specific head nodes for ATTN.
		Returns:
			A list of potential previous nodes.
		"""
		prev_nodes = []
		common_args = {"model": self.model, "msg_cache": self.msg_cache, "parent": self, "patch_type": self.patch_type, "cf_cache": self.cf_cache}

		for l in range(model_cfg.n_layers):
			# MLPs
			prev_nodes.append(MLP_ApproxNode(layer=l, position=self.position, **common_args))

			# ATTN query positions
			if include_head:
				prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, keyvalue_position=None, position=self.position,  patch_key=False, patch_value=False, patch_query=True, **common_args) for h in range(model_cfg.n_heads)])
			else:
				prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, keyvalue_position=None, position=self.position,  patch_key=False, patch_value=False, patch_query=True, **common_args))

			# ATTN key-value positions
			if self.position is not None:
				for keyvalue_position in range(self.position + 1):
					if include_head:
						if separate_kv:
							prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
							prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
						else:
							prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
					else:
						if separate_kv:
							prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=False, patch_query=False, **common_args))
							prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_key=False, patch_value=True, patch_query=False, **common_args))
						else:
							prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=True, patch_query=False, **common_args))
			else:
				if include_head:
					if separate_kv:
						prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=None, keyvalue_position=None, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
						prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=None, keyvalue_position=None, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
					else:
						prev_nodes.extend([ATTN_ApproxNode(layer=l, head=h, position=None, keyvalue_position=None, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
				else:
					if separate_kv:
						prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=None, keyvalue_position=None, patch_key=True, patch_value=False, patch_query=False, **common_args))
						prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=None, keyvalue_position=None, patch_key=False, patch_value=True, patch_query=False, **common_args))
					else:
						prev_nodes.append(ATTN_ApproxNode(layer=l, head=None, position=self.position, keyvalue_position=None, patch_key=True, patch_value=True, patch_query=False, **common_args))

		prev_nodes.append(EMBED_ApproxNode(layer=0, position=self.position, **common_args))
		# Remove duplicates
		prev_nodes = list(set(prev_nodes))
		return prev_nodes

	def __repr__(self):
		pos_str = f", position={self.position}" if self.position is not None else ""
		return f"FINAL_ApproxNode(layer={self.layer}{pos_str})"

	def __hash__(self):
		return hash((type(self).__name__, self.layer, self.position)) 