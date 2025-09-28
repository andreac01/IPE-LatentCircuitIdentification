import abc
import torch
import torch
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from ipe.attention import custom_attention_forward
from functools import total_ordering
from typing import Optional


@total_ordering
class Node(abc.ABC):
	"""Abstract base class for representing computational nodes in a transformer model.
	
	Implements functionality aimed at providing an unified interface for calculating:
	- The effect of an input message on the output of the node (forward method)
	- The list of predecessor nodes in the computational graph (get_expansion_candidates method)
	- The gradient of the final output of a the path with respect to the input of this node (calculate_gradient method)

	Attributes:
		model (HookedTransformer): 
			The transformer model instance. It is assumed to be a HookedTransformer from transformer_lens library. Any other implementation which provide the same interface should work as well.
		layer (int): 
			Layer index in the transformer. Embedding layer is assumed to be layer 0.
		position (int, default=None): 
			Token position if position-specific, else None. None is equivalent to all positions.
		parent (Node, default=None): 
			Parent node in the next node in the path. The parent is a successor in the computational graph.
		children (set, default=set()): 
			Set of child nodes. A child is a predecessor in the computational graph.
		msg_cache (dict): 
			Clean activation cache. Can be obtained by running the model with hooks using the clean prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads). 
		cf_cache (dict, default={}): 
			Counterfactual activation cache. Can be obtained by running the model with hooks using the corrupted prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads).
		gradient (torch.Tensor, default=None): 
			Node cached gradient. Usually is used to represent the gradient of the final output with respect to the input of this node, passing trough the path from final node to the current one.
		input_name (str): 
			Input activation name. This is the name associated to the cache entry corresponding to the input of this node.
		output_name (str): 
			Output activation name. This is the name associated to the cache entry corresponding to the output of this node.
		patch_type (str, default='zero'): 
			Type of intervention ('zero' or 'counterfactual'). Zero patching corresponds to removing the message from the first node in the path to the input of the next node, while counterfactual patching corresponds to replacing the message with the counterfactual activation. In both cases the effect of the path is then calculated by propagating the message through the whole path.

	Notes:
		This is an abstract base class that should not be instantiated directly. Concrete implementations should inherit from this class and implement the required abstract methods.
	"""
	def __init__(self, model: HookedTransformer, layer: int, msg_cache: dict, input_name: str, output_name: str, position: int = None, cf_cache: dict = {}, parent = None, children: set = set(), gradient: torch.Tensor = None, patch_type: str = 'zero'):
		"""	 Initializes an Node instance.

		Args:
			model (HookedTransformer): 
				The transformer model instance. It is assumed to be a HookedTransformer from transformer_lens library. Any other implementation which provide the same interface should work as well.
			layer (int): 
				Layer index in the transformer. Embedding layer is assumed to be layer 0.
			position (int, default=None): 
				Token position if position-specific, else None. None is equivalent to all positions.
			parent (Node, default=None): 
				Parent node in the next node in the path. The parent is a successor in the computational graph.
			children (set, default=set()): 
				Set of child nodes. A child is a predecessor in the computational graph.
			msg_cache (dict): 
				Clean activation cache. Can be obtained by running the model with hooks using the clean prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads). 
			cf_cache (dict, default={}): 
				Counterfactual activation cache. Can be obtained by running the model with hooks using the corrupted prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads).
			gradient (torch.Tensor, default=None): 
				Node cached gradient. Usually is used to represent the gradient of the final output with respect to the input of this node, passing trough the path from final node to the current one.
			input_name (str): 
				Input activation name. This is the name associated to the cache entry corresponding to the input of this node.
			output_name (str): 
				Output activation name. This is the name associated to the cache entry corresponding to the output of this node.
			patch_type (str, default='zero'): 
				Type of intervention ('zero' or 'counterfactual'). Zero patching corresponds to removing the message from the first node in the path to the input of the next node, while counterfactual patching corresponds to replacing the message with the counterfactual activation. In both cases the effect of the path is then calculated by propagating the message through the whole path.
		Returns:
			Node:
				An instance of Node.
	"""
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

	def add_child(self, child: 'Node') -> None:
		"""Adds a node as a child of self and sets self as its parent. A child can be interpreted as a predecessor in the computational graph.
		
		Args:
			child (Node)
				The Node to be added as a child of self.
		
		Returns:
			None

		"""
		self.children.add(child)
		child.parent.add(self)

	def add_parent(self, parent: 'Node') -> None:
		"""Adds a node as a parent of self and update the list of children of the parent node. A parent can be interpreted as a successor in the computational graph.
		
		Args:
			parent (Node):
				The Node to be added as a parent of self.
		
		Returns:
			None
		
		"""
		self.parent.add(parent)
		parent.children.add(self)

	@abc.abstractmethod
	
	def forward(self, message: torch.Tensor = None) -> torch.Tensor:
		"""
		Calculate the effect of the message on the output of the node. 
		
		The effect is calculated indirectly as the difference between the normal output of the component and the 
		one obtained when the message is removed from the input of the node.
		On the other hand, if message is None the behavior depends on the patch_type:
		- 'zero': returns the normal output of the component
		- 'counterfactual': returns the difference between the normal output and the counterfactual output of the component

		Args:

			message (torch.Tensor of shape (batch_size, seq_len, d_model), default=None):
				The message whose effect on the node need to be evaluated. If None, returns the normal 
				output or the difference between normal and counterfactual output depending on patch_type.
			
		Returns:
			Tensor:
				A tensor representing the effect of the message on the output of the node.
				In simpler terms, it represents the message caused by passing the input message through this node.

		Notes:
		- If a position is specified the output will be zero for all other positions.
		- The method assumes that the msg_cache and cf_cache contain the necessary activations.
		- When message is None, the method will cache the output in msg_cache or cf_cache if not already present.
		"""
		pass

	@abc.abstractmethod
	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, include_head: bool = False, separate_kv: bool = False) -> list['Node']:
		"""
		Returns the list of predecessors nodes in the computational graph whose outputs influence the
		output of this node. 

		Args:

			model_cfg (HookedTransformerConfig):
				The configuration of the transformer model. It is used to determine the number of heads and other model parameters.

			include_head (bool, default=False):
				Whether to consider specific head nodes for ATTN.

			separate_kv (bool, default=False):
				Whether to consider key and value positions separately for ATTN nodes 
		
		Returns:
			list of Node:
				The list of all predecessor nodes infuencing the input of this node.
		"""
		pass

	@abc.abstractmethod
	
	def calculate_gradient(self, grad_outputs=None, save=True, use_precomputed=False) -> torch.Tensor:
		"""
		Calculates the gradient of the node's input with respect to the final output.
		By default the gradient is calculated propagating backwards from the parent node if present,
		or assuming a gradient of ones if self has no parent. When 'grad_outputs' is specified, it is used instead of the parent's gradient.

		Args:
			grad_outputs : torch.Tensor, optional (default=None)
				Gradient to propagate backwards. If None, uses the gradient from the parent node or ones.
			
			save : bool, optional (default=True)
				Whether to save the computed gradient in self.gradient. The gradient can be reused
				later by setting use_precomputed to True.
			
			use_precomputed : bool, optional (default=False)
				Whether to use the precomputed gradient if available. The precoputed gradient is stored whenever
				save is True.
		
		Returns:
			gradient : torch.Tensor
				A tensor representing the gradient of the output with respect to the input
				of this node, passing trough the path from final node to the current one.
		"""


	@abc.abstractmethod
	def __repr__(self) -> str:
		"""
		Returns a string representation of the node. 

		Returns:
			str:
				A string representation of the node.
		"""
		pass

	def _get_sort_key(self) -> tuple:
		"""Helper method to return a tuple for sorting.
		Sorting order:
		1. Layer (ascending)
		2. Position (ascending, None last)
		3. Node type (EMBED < ATTN < MLP < FINAL)
		4. Key/Value position (ascending, None last)
		5. Head (ascending, None last)

		Returns:
			tuple: 
				A tuple representing the sort key.
		"""
		# Define an order for node types
		type_order = {EMBED_Node: 0, ATTN_Node: 1, MLP_Node: 2, FINAL_Node: 3}
		node_type = type(self)
		layer = self.layer if self.layer is not None else -1
		pos = self.position if self.position is not None else -1
		keyvalue_position = getattr(self, 'keyvalue_position', -1)
		keyvalue_position = keyvalue_position if keyvalue_position is not None else -1
		head = getattr(self, 'head', None)
		head = head if head is not None else -1

		return (
			layer,
			pos,
			type_order.get(node_type, 99),
			keyvalue_position,
			head
		)

	def __lt__(self, other) -> bool:
		"""Defines a total ordering for Node instances based on layer, position, type, key/value position, and head.
		Args:
			other (Node): The other Node instance to compare with.
		Returns:
			bool: True if self is less than other, False otherwise.
		"""
		if not isinstance(other, Node):
			return NotImplemented
		return self._get_sort_key() < other._get_sort_key()

	def __eq__(self, other) -> bool:
		"""Checks equality between two Node instances based on layer, position, type, key/value position, and head.
		Args:
			other (Node): The other Node instance to compare with.
		Returns:
			bool: True if self is equal to other, False otherwise.
		"""
		if not isinstance(other, Node):
			return NotImplemented
		if (self.layer != other.layer or self.position != other.position or type(self) is not type(other)):
			return False
		if isinstance(self, ATTN_Node) and isinstance(other, ATTN_Node):
			return self.head == other.head and self.position == other.position and self.keyvalue_position == other.keyvalue_position and self.patch_key == other.patch_key and self.patch_query == other.patch_query and self.patch_value == other.patch_value
		return True
		

	def __hash__(self) -> int:
		"""Generates a hash based on layer, position and type.
		Returns:
			int: The hash value of the Node instance.
		Notes:
			This methos is overridden in child classes to include additional attributes.
		"""
		return hash((type(self).__name__, self.layer, self.position))

class MLP_Node(Node):
	"""Represents an Multi Layer Perceptron (also referred as Feed-Forward Networl) node in the transformer. 
	This node operates on the residual stream within a specific layer and position.
	Note that an MLP output in a specific position is independent from the outputs in other positions, allowing for easier caching and patching.
	
	Attributes:
		model (HookedTransformer): 
			The transformer model instance. It is assumed to be a HookedTransformer from transformer_lens library. Any other implementation which provide the same interface should work as well.
		layer (int): 
			Layer index in the transformer. Embedding layer is assumed to be layer 0.
		position (int, default=None): 
			Token position if position-specific, else None. None is equivalent to all positions.
		parent (Node, default=None): 
			Parent node in the next node in the path. The parent is a successor in the computational graph.
		children (set, default=set()): 
			Set of child nodes. A child is a predecessor in the computational graph.
		msg_cache (dict): 
			Clean activation cache. Can be obtained by running the model with hooks using the clean prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads). 
		cf_cache (dict, default={}): 
			Counterfactual activation cache. Can be obtained by running the model with hooks using the corrupted prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads).
		gradient (torch.Tensor, default=None): 
			Node cached gradient. Usually is used to represent the gradient of the final output with respect to the input of this node, passing trough the path from final node to the current one.
		input_name (str): 
			Input activation name. This is the name associated to the cache entry corresponding to the input of this node.
		output_name (str): 
			Output activation name. This is the name associated to the cache entry corresponding to the output of this node.
		patch_type (str, default='zero'): 
			Type of intervention ('zero' or 'counterfactual'). Zero patching corresponds to removing the message from the first node in the path to the input of the next node, while counterfactual patching corresponds to replacing the message with the counterfactual activation. In both cases the effect of the path is then calculated by propagating the message through the whole path.
	"""
	def __init__(self, model: HookedTransformer, layer: int, position: int = None, parent: Node = None, children = set(), msg_cache = {}, cf_cache = {}, gradient = None, patch_type = 'zero'):
		"""Initializes an MLP_Node instance.
		Args:
			model (HookedTransformer): 
				The transformer model instance. It is assumed to be a HookedTransformer from transformer_lens library. Any other implementation which provide the same interface should work as well.
			layer (int): 
				Layer index in the transformer. Embedding layer is assumed to be layer 0.
			position (int, default=None): 
				Token position if position-specific, else None. None is equivalent to all positions.
			parent (Node, default=None): 
				Parent node in the next node in the path. The parent is a successor in the computational graph.
			children (set, default=set()): 
				Set of child nodes. A child is a predecessor in the computational graph.
			msg_cache (dict): 
				Clean activation cache. Can be obtained by running the model with hooks using the clean prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads). 
			cf_cache (dict, default={}): 
				Counterfactual activation cache. Can be obtained by running the model with hooks using the corrupted prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads).
			gradient (torch.Tensor, default=None): 
				Node cached gradient. Usually is used to represent the gradient of the final output with respect to the input of this node, passing trough the path from final node to the current one.
			patch_type (str, default='zero'): 
				Type of intervention ('zero' or 'counterfactual'). Zero patching corresponds to removing the message from the first node in the path to the input of the next node, while counterfactual patching corresponds to replacing the message with the counterfactual activation. In both cases the effect of the path is then calculated by propagating the message through the whole path.
		Returns:
			self (MLP_Node):
				The initialized MLP_Node instance.
		"""
		super().__init__(model=model, layer=layer, position=position, parent=parent, children=children, msg_cache=msg_cache, cf_cache=cf_cache, gradient=gradient, input_name=f"blocks.{layer}.hook_resid_mid", output_name=f"blocks.{layer}.hook_mlp_out", patch_type=patch_type)
	

	
	def forward(self, message: torch.Tensor) -> torch.Tensor:
		"""
		Calculate the effect of the message on the output of the node. 
		
		The effect is calculated indirectly as the difference between the normal output of the MLP and the 
		one obtained when the message is removed from the input of the node.
		On the other hand, if message is None the behavior depends on the patch_type:
		- 'zero': returns the normal output of the MLP
		- 'counterfactual': returns the difference between the normal output and the counterfactual output

		Args:

			message (torch.Tensor of shape (batch_size, seq_len, d_model), default=None):
				The message whose effect on the node need to be evaluated. If None, returns the normal 
				output or the difference between normal and counterfactual output depending on patch_type.
			
		Returns:
			torch.Tensor:
				A tensor representing the effect of the message on the output of the node.

		Notes:
		- If a position is specified the output will be zero for all other positions.
		- The method assumes that the msg_cache and cf_cache contain the necessary activations.
		- When message is None, the method will cache the output in msg_cache or cf_cache if not already present.
		"""
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
					out[:, self.position, :] = self.msg_cache[self.output_name][:, self.position, :].detach().clone() - self.cf_cache[self.output_name][:, self.position, :].detach().clone()
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


	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, include_head: bool = False, separate_kv: bool = False) -> list[Node]:
		"""
		Returns the list of predecessors nodes in the computational graph whose outputs influence the
		output of this node. 
		Previous nodes of an MLP are:
			- MLP, EMBED and ATTN nodes in self.position from previous layers.
			- ATTN nodes in self.position from current layers.
		ATTN nodes are always patched both in query and key-value positions separately.
		Args:

			model_cfg (HookedTransformerConfig):
				The configuration of the transformer model. It is used to determine the number of heads and other model parameters.

			include_head (bool, default=False):
				Whether to consider specific head nodes for ATTN.

			separate_kv (bool, default=False):
				Whether to consider key and value positions separately for ATTN nodes 
		
		Returns:
			list of Node:
				The list of all predecessor nodes infuencing the input of this node.
		
		Notes:
			- If self.position is None, only non-position-specific previous nodes are considered.
		"""
		prev_nodes = []
		common_args = {"model": self.model, "msg_cache": self.msg_cache, "cf_cache": self.cf_cache, "parent": self, "patch_type": self.patch_type}
		if self.position is not None:
			positions_to_iterate = range(self.position + 1)
		else:
			positions_to_iterate = [None]

		# MLP and ATTN nodes from previous layers
		for l in range(self.layer):
			prev_nodes.append(MLP_Node(layer=l, position=self.position, **common_args))
			for p in positions_to_iterate:
				if include_head:
					if separate_kv:
						prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=p, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
						prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=p, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
					else:
						prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=p, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
				else:
					if separate_kv:
						prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=p, patch_key=True, patch_value=False, patch_query=False, **common_args))
						prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=p, patch_key=False, patch_value=True, patch_query=False, **common_args))
					else:
						prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=p, patch_key=True, patch_value=True, patch_query=False, **common_args))
			if include_head:
				prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args) for h in range(model_cfg.n_heads)])
			else:
				prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args))

		# ATTN nodes from current layer
		for p in positions_to_iterate:
			if include_head:
				if separate_kv:
					prev_nodes.extend([ATTN_Node(layer=self.layer, head=h, position=self.position, keyvalue_position=p, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
					prev_nodes.extend([ATTN_Node(layer=self.layer, head=h, position=self.position, keyvalue_position=p, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.extend([ATTN_Node(layer=self.layer, head=h, position=self.position, keyvalue_position=p, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
			else:
				if separate_kv:
					prev_nodes.append(ATTN_Node(layer=self.layer, head=None, position=self.position, keyvalue_position=p, patch_key=True, patch_value=False, patch_query=False, **common_args))
					prev_nodes.append(ATTN_Node(layer=self.layer, head=None, position=self.position, keyvalue_position=p, patch_key=False, patch_value=True, patch_query=False, **common_args))
				else:
					prev_nodes.append(ATTN_Node(layer=self.layer, head=None, position=self.position, keyvalue_position=p, patch_key=True, patch_value=True, patch_query=False, **common_args))
		if include_head:
			prev_nodes.extend([ATTN_Node(layer=self.layer, head=h, position=self.position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args) for h in range(model_cfg.n_heads)])
		else:
			prev_nodes.append(ATTN_Node(layer=self.layer, head=None, position=self.position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args))

		# EMBED node
		prev_nodes.append(EMBED_Node(layer=0, position=self.position, **common_args))
		# Remove duplicates
		prev_nodes = list(set(prev_nodes))
		return prev_nodes

	def __repr__(self) -> str:
		"""Returns a string representation of the MLP node.
		Returns:
			str:
				A string representation of the MLP node.
		"""
		return f"MLP_Node(layer={self.layer}, position={self.position})"

	def __hash__(self) -> int:
		"""Generates a hash based on layer, position and type.
		Returns:
			int: The hash value of the MLP_Node instance.
		"""
		return hash((type(self).__name__, self.layer, self.position))
	
	
	def calculate_gradient(self, grad_outputs=None, save=True, use_precomputed=False) -> torch.Tensor:
		"""
		Calculates the gradient of the node's input with respect to the final output.
		By default the gradient is calculated propagating backwards from the parent node if present,
		or assuming a gradient of ones if self has no parent. When 'grad_outputs' is specified, it is used instead of the parent's gradient.

		Args:
			grad_outputs : torch.Tensor, optional (default=None)
				Gradient to propagate backwards. If None, uses the gradient from the parent node or ones.
			
			save : bool, optional (default=True)
				Whether to save the computed gradient in self.gradient. The gradient can be reused
				later by setting use_precomputed to True.
			
			use_precomputed : bool, optional (default=False)
				Whether to use the precomputed gradient if available. The precoputed gradient is stored whenever
				save is True.
		
		Returns:
			gradient : torch.Tensor
				A tensor representing the gradient of the output with respect to the input
				of this node, passing trough the path from final node to the current one.
		"""
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
	

class ATTN_Node(Node):
	"""Represents an Attention node (potentially a specific head) in the transformer model.

	Attributes:
		model (HookedTransformer): 
			The transformer model instance. It is assumed to be a HookedTransformer from transformer_lens library. Any other implementation which provide the same interface should work as well.
		layer (int): 
			Layer index in the transformer. Embedding layer is assumed to be layer 0.
		head (int, default=None):
			Attention head index if head-specific, else None. None is equivalent to all heads. When an head is specified the contribution of the is considered, particularly the bias term which is not head-specific is not included. Therefore the output of an ATTN node is equal to the output of all the heads plus the bias term. If head is None the whole attention output is considered.
		position (int, default=None): 
			Token position if position-specific, else None. None is equivalent to all positions.
		keyvalue_position (int, default=None):
			Key/Value token position if position-specific, else None. None is equivalent to all positions.
			If keyvalue_position is specified, the node represents the contribution of the attention head when the value residual strams of all other positions are zeroed out. This is equivalent to attending only to a single position, but scaling the output by the attention score of that position.
		patch_key (bool, default=True):
			Whether to patch the key projection of the attention head. If False, the key projection is not patched and the message is only removed from the query and/or value projections.
		patch_value (bool, default=True):
			Whether to patch the value projection of the attention head. If False, the value projection is not patched and the message is only removed from the query and/or key projections.
		patch_query (bool, default=True):
			Whether to patch the query projection of the attention head. If False, the query projection is not patched and the message is only removed from the key and/or value projections.
		parent (Node, default=None): 
			Parent node in the next node in the path. The parent is a successor in the computational graph.
		children (set, default=set()): 
			Set of child nodes. A child is a predecessor in the computational graph.
		msg_cache (dict): 
			Clean activation cache. Can be obtained by running the model with hooks using the clean prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads). 
		cf_cache (dict, default={}): 
			Counterfactual activation cache. Can be obtained by running the model with hooks using the corrupted prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads).
		gradient (torch.Tensor, default=None): 
			Node cached gradient. Usually is used to represent the gradient of the final output with respect to the input of this node, passing trough the path from final node to the current one.
		attn_scores (str):
			Attention scores activation name. This is the name associated to the cache entry corresponding to the attention scores of this attention block. It is used to only recompute the attention scores of relevant positions when patching, drastically reducing computation.
		input_name (str): 
			Input activation name. This is the name associated to the cache entry corresponding to the input of this node.
		output_name (str): 
			Output activation name. This is the name associated to the cache entry corresponding to the output of this node.
		patch_type (str, default='zero'): 
			Type of intervention ('zero' or 'counterfactual'). Zero patching corresponds to removing the message from the first node in the path to the input of the next node, while counterfactual patching corresponds to replacing the message with the counterfactual activation. In both cases the effect of the path is then calculated by propagating the message through the whole path.
		plot_patterns (bool, default=False):
			Whether to plot the attention patterns when calculating the forward pass. This is useful for debugging purposes but also to visualize the changes in the attention patterns when patching specific positions.
	"""
	def __init__(self, model: HookedTransformer, layer: int, head: int = None, position: int = None, keyvalue_position: int = None, parent: Node = None, children = set(), msg_cache = {}, cf_cache = {}, gradient = None, patch_query: bool = True, patch_key: bool = True, patch_value: bool = True, plot_patterns: bool = False, patch_type = 'zero'):
		"""Initializes an ATTN_Node instance.

		Args:
			model (HookedTransformer): 
				The transformer model instance. It is assumed to be a HookedTransformer from transformer_lens library. Any other implementation which provide the same interface should work as well.
			layer (int): 
				Layer index in the transformer. Embedding layer is assumed to be layer 0.
			head (int, default=None):
				Attention head index if head-specific, else None. None is equivalent to all heads. When an head is specified the contribution of the is considered, particularly the bias term which is not head-specific is not included. Therefore the output of an ATTN node is equal to the output of all the heads plus the bias term. If head is None the whole attention output is considered.
			position (int, default=None): 
				Token position if position-specific, else None. None is equivalent to all positions.
			keyvalue_position (int, default=None):
				Key/Value token position if position-specific, else None. None is equivalent to all positions.
				If keyvalue_position is specified, the node represents the contribution of the attention head when the value residual strams of all other positions are zeroed out. This is equivalent to attending only to a single position, but scaling the output by the attention score of that position.
			patch_key (bool, default=True):
				Whether to patch the key projection of the attention head. If False, the key projection is not patched and the message is only removed from the query and/or value projections.
			patch_value (bool, default=True):
				Whether to patch the value projection of the attention head. If False, the value projection is not patched and the message is only removed from the query and/or key projections.
			patch_query (bool, default=True):
				Whether to patch the query projection of the attention head. If False, the query projection is not patched and the message is only removed from the key and/or value projections.
			parent (Node, default=None): 
				Parent node in the next node in the path. The parent is a successor in the computational graph.
			children (set, default=set()): 
				Set of child nodes. A child is a predecessor in the computational graph.
			msg_cache (dict): 
				Clean activation cache. Can be obtained by running the model with hooks using the clean prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads). 
			cf_cache (dict, default={}): 
				Counterfactual activation cache. Can be obtained by running the model with hooks using the corrupted prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads).
			gradient (torch.Tensor, default=None): 
				Node cached gradient. Usually is used to represent the gradient of the final output with respect to the input of this node, passing trough the path from final node to the current one.
			patch_type (str, default='zero'): 
				Type of intervention ('zero' or 'counterfactual'). Zero patching corresponds to removing the message from the first node in the path to the input of the next node, while counterfactual patching corresponds to replacing the message with the counterfactual activation. In both cases the effect of the path is then calculated by propagating the message through the whole path.
			plot_patterns (bool, default=False):
				Whether to plot the attention patterns when calculating the forward pass. This is useful for debugging purposes but also to visualize the changes in the attention patterns when patching specific positions.
		
		Returns:
			self (ATTN_Node):
				The initialized ATTN_Node instance.
		"""
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

	
	def forward(self, message: torch.Tensor) -> torch.Tensor:
		"""
		Calculate the effect of the message on the output of the node. 
		
		The effect is calculated indirectly as the difference between the normal output of the component and the 
		one obtained when the message is removed from the input of the node.
		On the other hand, if message is None the behavior depends on the patch_type:
		- 'zero': returns the normal output of the component
		- 'counterfactual': returns the difference between the normal output and the counterfactual output of the component

		Args:

			message (torch.Tensor of shape (batch_size, seq_len, d_model), default=None):
				The message whose effect on the node need to be evaluated. If None, returns the normal 
				output or the difference between normal and counterfactual output depending on patch_type.
			
		Returns:
			torch.Tensor:
				A tensor representing the effect of the message on the output of the node.
				In simpler terms, it represents the message caused by passing the input message through this node.

		Notes:
		- If a position is specified the output will be zero for all other positions.
		- The method assumes that the msg_cache and cf_cache contain the necessary activations.
		- When message is None, the method will cache the output in msg_cache or cf_cache if not already present.
		- The method automatically adds entries to the msg_cache and cf_cache, correspoding to the output of single attention heads, if they are not already present.
		- This method uses precomputed attention scores if possible, this may introduce small numerical differences compared to a full recomputation.
		"""
		length = self.position+1 if self.position is not None else self.msg_cache[self.input_name].shape[1]
		if message is None:
			value_residual = self.msg_cache[self.input_name]
			if self.patch_type == 'zero':
				if self.output_name in self.msg_cache:
					if self.position is None:
						return self.msg_cache[self.output_name]
					else:
						out = torch.zeros_like(self.msg_cache[self.input_name])
						out[:, self.position, :] = self.msg_cache[self.output_name][:, self.position, :].detach()
						return out
				else:
					if self.position is None:
						query_residual = self.msg_cache[self.input_name]
					else:
						query_residual = self.msg_cache[self.input_name][:, self.position, :].unsqueeze(1)
					if self.keyvalue_position is None:
						key_residual = self.msg_cache[self.input_name][:, :length]
					else:
						key_residual = self.msg_cache[self.input_name][:, self.keyvalue_position, :].unsqueeze(1)
			if self.patch_type == 'counterfactual':
				value_residual = self.cf_cache[self.input_name]
				if self.output_name in self.cf_cache and self.output_name in self.msg_cache:
					if self.position is None:
						return self.msg_cache[self.output_name] - self.cf_cache[self.output_name]
					else:
						out = torch.zeros_like(self.msg_cache[self.input_name])
						out[:, self.position, :] = self.msg_cache[self.output_name][:, self.position, :] - self.cf_cache[self.output_name][:, self.position, :]
						return out
				else:
					out = ATTN_Node(self.model, layer=self.layer, head=self.head, msg_cache=self.msg_cache, cf_cache={}, keyvalue_position=self.keyvalue_position, patch_type='zero').forward(message=None) - ATTN_Node(self.model, layer=self.layer, head=self.head, msg_cache=self.cf_cache, cf_cache={}, keyvalue_position=self.keyvalue_position, patch_type='zero').forward(message=None)
					if self.position is None:
						return out
					out_pos = torch.zeros_like(self.msg_cache[self.input_name])
					out_pos[:, self.position, :] = out[:, self.position, :]
					return out_pos
						
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
					value_residual = self.msg_cache[self.input_name].detach().clone() - message
				else:
					value_residual = self.msg_cache[self.input_name].detach().clone()
					value_residual[:, self.keyvalue_position, :] = value_residual[:, self.keyvalue_position, :].detach().clone() - message[:, self.keyvalue_position, :]
			else:
				value_residual = self.msg_cache[self.input_name]

		key_residual = self.model.blocks[self.layer].ln1(key_residual)
		value_residual = self.model.blocks[self.layer].ln1(value_residual)
		query_residual = self.model.blocks[self.layer].ln1(query_residual)
		if self.head is not None:
			add_bias = False
			W_Q = self.model.blocks[self.layer].attn.W_Q[self.head].unsqueeze(0)
			W_K = self.model.blocks[self.layer].attn.W_K[self.head].unsqueeze(0)
			W_V = self.model.blocks[self.layer].attn.W_V[self.head].unsqueeze(0)
			b_Q = self.model.blocks[self.layer].attn.b_Q[self.head].unsqueeze(0)
			b_K = self.model.blocks[self.layer].attn.b_K[self.head].unsqueeze(0)
			b_V = self.model.blocks[self.layer].attn.b_V[self.head].unsqueeze(0)
			query = torch.einsum('bsd,ndh->bsnh', query_residual, W_Q) + b_Q[None, None, :, :]
			key = torch.einsum('bsd,ndh->bsnh', key_residual, W_K) + b_K[None, None, :, :]
			if self.keyvalue_position is not None:
				v = torch.einsum('bd,ndh->bnh', value_residual[:, self.keyvalue_position, :], W_V) + b_V[None, :, :]
				value = torch.zeros(v.shape[0], value_residual.shape[1], W_V.shape[0], v.shape[2], device=v.device)
				value[:, self.keyvalue_position, :] = v
			else:
				value = torch.einsum('bsd,ndh->bsnh', value_residual, W_V) + b_V[None, None, :, :]
		else:
			add_bias = True
			if self.model.cfg.positional_embedding_type == 'rotary':
				step = self.model.cfg.n_heads // self.model.cfg.n_key_value_heads
			else:
				step = 1
			W_Q = self.model.blocks[self.layer].attn.W_Q
			W_K = self.model.blocks[self.layer].attn.W_K[::step]
			W_V = self.model.blocks[self.layer].attn.W_V[::step]
			b_Q = self.model.blocks[self.layer].attn.b_Q
			b_K = self.model.blocks[self.layer].attn.b_K[::step]
			b_V = self.model.blocks[self.layer].attn.b_V[::step]
			query = torch.einsum('bsd,ndh->bsnh', query_residual, W_Q) + b_Q[None, None, :, :]
			key = torch.einsum('bsd,ndh->bsnh', key_residual, W_K) + b_K[None, None, :, :]
			if self.keyvalue_position is not None:
				v = torch.einsum('bd,ndh->bnh', value_residual[:, self.keyvalue_position, :], W_V) + b_V[None, :, :]
				value = torch.zeros(v.shape[0], value_residual.shape[1], W_V.shape[0], v.shape[2], device=v.device)
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
			plot_patterns=self.plot_patterns,
			add_bias=add_bias
		)
		

		if self.patch_type == 'zero' and self.msg_cache.get(self.output_name, None) is None:
			if self.position is None and message is None:
				self.msg_cache[self.output_name] = out.detach().clone()
			else:
				ATTN_Node(self.model, layer=self.layer, head=self.head, msg_cache=self.msg_cache, cf_cache=self.cf_cache, keyvalue_position=self.keyvalue_position, patch_type='zero').forward(message=None)
		if message is None:
			if self.patch_type == 'zero':
				if self.position is not None:
					resized_out = torch.zeros_like(self.msg_cache[self.input_name], device=out.device)
					resized_out[:, self.position, :] = out
					return resized_out
				self.msg_cache[self.output_name] = out.detach().clone()
				return out
			else:
				raise ValueError(f"Invalid patch type: {self.patch_type}")
		
		if self.position is not None:
			resized_out = torch.zeros_like(self.msg_cache[self.input_name], device=out.device)
			resized_out[:, self.position, :] = self.msg_cache[self.output_name][:, self.position, :].detach().clone() - out
			return resized_out
		return self.msg_cache[self.output_name].detach().clone() - out
	
	
	def calculate_gradient(self, grad_outputs=None, save=True, use_precomputed=False) -> torch.Tensor:
		"""
		Calculates the gradient of the node's input with respect to the final output.
		By default the gradient is calculated propagating backwards from the parent node if present,
		or assuming a gradient of ones if self has no parent. When 'grad_outputs' is specified, it is used instead of the parent's gradient.

		Args:
			grad_outputs : torch.Tensor, optional (default=None)
				Gradient to propagate backwards. If None, uses the gradient from the parent node or ones.
			
			save : bool, optional (default=True)
				Whether to save the computed gradient in self.gradient. The gradient can be reused
				later by setting use_precomputed to True.
			
			use_precomputed : bool, optional (default=False)
				Whether to use the precomputed gradient if available. The precoputed gradient is stored whenever
				save is True.
		
		Returns:
			gradient : torch.Tensor
				A tensor representing the gradient of the output with respect to the input
				of this node, passing trough the path from final node to the current one.
		"""
		if self.gradient is not None and use_precomputed:
			if self.position is None or self.keyvalue_position is None:
				return self.gradient.detach().clone()
			gradient = self.gradient.detach().clone()
			out = torch.zeros_like(gradient, device=gradient.device)
			if self.patch_value or self.patch_key:
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
				add_bias = False
				W_Q = self.model.blocks[self.layer].attn.W_Q[self.head].unsqueeze(0)
				W_K = self.model.blocks[self.layer].attn.W_K[self.head].unsqueeze(0)
				W_V = self.model.blocks[self.layer].attn.W_V[self.head].unsqueeze(0)
				b_Q = self.model.blocks[self.layer].attn.b_Q[self.head].unsqueeze(0)
				b_K = self.model.blocks[self.layer].attn.b_K[self.head].unsqueeze(0)
				b_V = self.model.blocks[self.layer].attn.b_V[self.head].unsqueeze(0)
				query = torch.einsum('bsd,ndh->bsnh', query_residual, W_Q) + b_Q[None, None, :, :]
				key = torch.einsum('bsd,ndh->bsnh', key_residual, W_K) + b_K[None, None, :, :]
				if self.keyvalue_position is not None:
					v = torch.einsum('bd,ndh->bnh', value_residual[:, self.keyvalue_position, :], W_V) + b_V[None, :, :]
					value = torch.zeros(v.shape[0], value_residual.shape[1], W_V.shape[0], v.shape[2], device=v.device)
					value[:, self.keyvalue_position, :] = v
				else:
					value = torch.einsum('bsd,ndh->bsnh', value_residual, W_V) + b_V[None, None, :, :]
			else:
				add_bias = True
				if self.model.cfg.positional_embedding_type == 'rotary':
					step = self.model.cfg.n_heads // self.model.cfg.n_key_value_heads
				else:
					step = 1
				W_Q = self.model.blocks[self.layer].attn.W_Q
				W_K = self.model.blocks[self.layer].attn.W_K[::step]
				W_V = self.model.blocks[self.layer].attn.W_V[::step]
				b_Q = self.model.blocks[self.layer].attn.b_Q
				b_K = self.model.blocks[self.layer].attn.b_K[::step]
				b_V = self.model.blocks[self.layer].attn.b_V[::step]
				query = torch.einsum('bsd,ndh->bsnh', query_residual, W_Q) + b_Q[None, None, :, :]
				key = torch.einsum('bsd,ndh->bsnh', key_residual, W_K) + b_K[None, None, :, :]
				if self.keyvalue_position is not None:
					v = torch.einsum('bd,ndh->bnh', value_residual[:, self.keyvalue_position, :], W_V) + b_V[None, :, :]
					value = torch.zeros(v.shape[0], value_residual.shape[1], W_V.shape[0], v.shape[2], device=v.device)
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
				plot_patterns=self.plot_patterns,
				add_bias=add_bias
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


	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, include_head: bool = False, separate_kv: bool = False) -> list[Node]:
		"""
		Returns the list of predecessors nodes in the computational graph whose outputs influence the
		output of this node. 
		Previous nodes are:
			- MLP, EMBED and ATTN nodes in self.position from previous layers if patch_query=True.
			- MLP, EMBED and ATTN nodes in all previous positions from previous layers if patch_key=True or patch_value=True.
		Args:

			model_cfg (HookedTransformerConfig):
				The configuration of the transformer model. It is used to determine the number of heads and other model parameters.

			include_head (bool, default=False):
				Whether to consider specific head nodes for ATTN.

			separate_kv (bool, default=False):
				Whether to consider key and value positions separately for ATTN nodes 
		
		Returns:
			list of Node:
				The list of all predecessor nodes infuencing the input of this node.

		Notes:
			- If self.position is None, only non-position-specific previous nodes are considered.
		"""	
		prev_nodes = []
		common_args = {"model": self.model, "msg_cache": self.msg_cache, "parent": self, "patch_type": self.patch_type, "cf_cache": self.cf_cache}
		# MLPs
		for l in range(self.layer):
			if self.patch_query:
				prev_nodes.append(MLP_Node(layer=l, position=self.position, **common_args))
			if (self.patch_key or self.patch_value) and (not self.patch_query or self.position != self.keyvalue_position):
				if self.keyvalue_position is not None:
					prev_nodes.append(MLP_Node(layer=l, position=self.keyvalue_position, **common_args))
				elif self.position is not None:
					for pos in range(self.position + 1):
						prev_nodes.append(MLP_Node(layer=l, position=pos, **common_args))
				else:
					prev_nodes.append(MLP_Node(layer=l, position=None, **common_args))

		# EMBED node
		if self.patch_query:
			prev_nodes.append(EMBED_Node(layer=0, position=self.position, **common_args))
		if (self.patch_key or self.patch_value) and (not self.patch_query or self.position != self.keyvalue_position):
			if self.keyvalue_position is not None:
				prev_nodes.append(EMBED_Node(layer=0, position=self.keyvalue_position, **common_args))
			elif self.position is not None:
				for pos in range(self.position + 1):
					prev_nodes.append(EMBED_Node(layer=0, position=pos, **common_args))
			else:
				prev_nodes.append(EMBED_Node(layer=0, position=None, **common_args))
		# ATTN nodes patching current query position
		if self.patch_query:
			for l in range(self.layer):
				# prev ATTN query positions
				if include_head:
					prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args) for h in range(model_cfg.n_heads)])
				else:
					prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args))

				# prev ATTN key-value positions
				if self.position is not None:
					for keyvalue_position in range(self.position + 1):
						if include_head:
							if separate_kv:
								prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
								prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
							else:
								prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
						else:
							if separate_kv:
								prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=False, patch_query=False, **common_args))
								prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_key=False, patch_value=True, patch_query=False, **common_args))
							else:
								prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=True, patch_query=False, **common_args))
				else:
					if include_head:
						if separate_kv:
							prev_nodes.extend([ATTN_Node(layer=l, head=h, position=None, keyvalue_position=None, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
							prev_nodes.extend([ATTN_Node(layer=l, head=h, position=None, keyvalue_position=None, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
						else:
							prev_nodes.extend([ATTN_Node(layer=l, head=h, position=None, keyvalue_position=None, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
					else:
						if separate_kv:
							prev_nodes.append(ATTN_Node(layer=l, head=None, position=None, keyvalue_position=None, patch_key=True, patch_value=False, patch_query=False, **common_args))
							prev_nodes.append(ATTN_Node(layer=l, head=None, position=None, keyvalue_position=None, patch_key=False, patch_value=True, patch_query=False, **common_args))
						else:
							prev_nodes.append(ATTN_Node(layer=l, head=None, position=None, keyvalue_position=None, patch_key=True, patch_value=True, patch_query=False, **common_args))

		# ATTN nodes patching current key-value position
		if (self.patch_key or self.patch_value) and (not self.patch_query or self.position != self.keyvalue_position):
			if self.keyvalue_position is not None:
				keyvalue_positions = [self.keyvalue_position]
			else:
				if self.position is None:
					keyvalue_positions = [None]
				else:
					keyvalue_positions = range(self.position + 1)
			for l in range(self.layer):
				# prev ATTN key-value positions
				for resid_position in keyvalue_positions:
					# prev ATTN query positions
					if include_head:
						prev_nodes.extend([ATTN_Node(layer=l, head=h, position=resid_position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args) for h in range(model_cfg.n_heads)])
					else:
						prev_nodes.append(ATTN_Node(layer=l, head=None, position=resid_position, keyvalue_position=None,  patch_key=False, patch_value=False, patch_query=True, **common_args))
					for prev_keyvalue_position in ( range(resid_position+1) if resid_position is not None else [None] ):
						if include_head:
							if separate_kv:
								prev_nodes.extend([ATTN_Node(layer=l, head=h, position=resid_position, keyvalue_position=prev_keyvalue_position, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
								prev_nodes.extend([ATTN_Node(layer=l, head=h, position=resid_position, keyvalue_position=prev_keyvalue_position, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
							else:
								prev_nodes.extend([ATTN_Node(layer=l, head=h, position=resid_position, keyvalue_position=prev_keyvalue_position, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
						else:
							if separate_kv:
								prev_nodes.append(ATTN_Node(layer=l, head=None, position=resid_position, keyvalue_position=prev_keyvalue_position, patch_key=True, patch_value=False, patch_query=False, **common_args))
								prev_nodes.append(ATTN_Node(layer=l, head=None, position=resid_position, keyvalue_position=prev_keyvalue_position, patch_key=False, patch_value=True, patch_query=False, **common_args))
							else:
								prev_nodes.append(ATTN_Node(layer=l, head=None, position=resid_position, keyvalue_position=prev_keyvalue_position, patch_key=True, patch_value=True, patch_query=False, **common_args))

		# Remove duplicates
		if self.position == 0 and self.keyvalue_position == None: # Avoid the case of overlapping duplicates 
			for node in prev_nodes:
				if node.position is None:
					node.position = 0
		prev_nodes = list(set(prev_nodes))
		return prev_nodes

	def __repr__(self) -> str:
		"""String representation of the ATTN_Node instance.
		Includes layer, head, position, keyvalue_position, and patching options.
		
		Returns:
			str:
				A string representation of the ATTN_Node instance.
		"""
		return f"ATTN_Node(layer={self.layer}, head={self.head}, position={self.position}, keyvalue_position={self.keyvalue_position}, patch_query={self.patch_query}, patch_key={self.patch_key}, patch_value={self.patch_value})"

	def __hash__(self) -> int:
		"""Hash function for the ATTN_Node instance.
		
		Returns:
			int:
				A hash value based on the layer, head, position, keyvalue_position, and patching options.
		"""
		return hash((type(self).__name__, self.layer, self.head, self.position, self.keyvalue_position, self.patch_query, self.patch_key, self.patch_value))


class EMBED_Node(Node):
	"""Represents the embedding node in the transformer. This is almost a dummy node, as it only serves as the starting point for paths that begin at the input embeddings. This classes uses cached activations from the model to provide an interface consistent with other node types.	

	Attributes:
		model (HookedTransformer): 
			The transformer model instance. It is assumed to be a HookedTransformer from transformer_lens library. Any other implementation which provide the same interface should work as well.
		layer (int): 
			Layer index in the transformer. Embedding layer is assumed to be layer 0.
		position (int, default=None): 
			Token position if position-specific, else None. None is equivalent to all positions.
		parent (Node, default=None): 
			Parent node in the next node in the path. The parent is a successor in the computational graph.
		children (set, default=set()): 
			Set of child nodes. A child is a predecessor in the computational graph.
		msg_cache (dict): 
			Clean activation cache. Can be obtained by running the model with hooks using the clean prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads). 
		cf_cache (dict, default={}): 
			Counterfactual activation cache. Can be obtained by running the model with hooks using the corrupted prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads).
		gradient (torch.Tensor, default=None): 
			Node cached gradient. Usually is used to represent the gradient of the final output with respect to the input of this node, passing trough the path from final node to the current one.
		input_name (str): 
			Input activation name. This is the name associated to the cache entry corresponding to the input of this node.
		output_name (str): 
			Output activation name. This is the name associated to the cache entry corresponding to the output of this node.
		patch_type (str, default='zero'): 
			Type of intervention ('zero' or 'counterfactual'). Zero patching corresponds to removing the message from the first node in the path to the input of the next node, while counterfactual patching corresponds to replacing the message with the counterfactual activation. In both cases the effect of the path is then calculated by propagating the message through the whole path.
	"""
	def __init__(self, model: HookedTransformer, layer: int = 0, position: int = None, parent: Node = None, children = set(), msg_cache = {}, cf_cache = {}, gradient = None, patch_type = 'zero'):
		"""Initializes the EMBED_Node instance.
		Args:
			model (HookedTransformer): 
				The transformer model instance. It is assumed to be a HookedTransformer from transformer_lens library. Any other implementation which provide the same interface should work as well.
			layer (int): 
				Layer index in the transformer. Embedding layer is assumed to be layer 0.
			position (int, default=None): 
				Token position if position-specific, else None. None is equivalent to all positions.
			parent (Node, default=None): 
				Parent node in the next node in the path. The parent is a successor in the computational graph.
			children (set, default=set()): 
				Set of child nodes. A child is a predecessor in the computational graph.
			msg_cache (dict): 
				Clean activation cache. Can be obtained by running the model with hooks using the clean prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads). 
			cf_cache (dict, default={}): 
				Counterfactual activation cache. Can be obtained by running the model with hooks using the corrupted prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads).
			gradient (torch.Tensor, default=None): 
				Node cached gradient. Usually is used to represent the gradient of the final output with respect to the input of this node, passing trough the path from final node to the current one.
			patch_type (str, default='zero'): 
				Type of intervention ('zero' or 'counterfactual'). Zero patching corresponds to removing the message from the first node in the path to the input of the next node, while counterfactual patching corresponds to replacing the message with the counterfactual activation. In both cases the effect of the path is then calculated by propagating the message through the whole path.
		Returns:
			self (EMBED_Node):
				An instance of the EMBED_Node class.
		"""		
		super().__init__(model=model, layer=layer, position=position, parent=parent, children=children, msg_cache=msg_cache, cf_cache=cf_cache, gradient=gradient, input_name="hook_embed", output_name="hook_embed", patch_type=patch_type)

	
	def forward(self, message: torch.Tensor = None) -> torch.Tensor:
		"""
		Calculate the effect of the message on the output of the node. 
		
		The effect is calculated indirectly as the difference between the normal output of the component and the 
		one obtained when the message is removed from the input of the node.
		On the other hand, if message is None the behavior depends on the patch_type:
		- 'zero': returns the normal output of the component
		- 'counterfactual': returns the difference between the normal output and the counterfactual output of the component

		Args:

			message (torch.Tensor of shape (batch_size, seq_len, d_model), default=None):
				The message whose effect on the node need to be evaluated. If None, returns the normal 
				output or the difference between normal and counterfactual output depending on patch_type.
			
		Returns:
			torch.Tensor:
				A tensor representing the effect of the message on the output of the node.
				In simpler terms, it represents the message caused by passing the input message through this node.

		Notes:
			- If a position is specified the output will be zero for all other positions.
			- The method assumes that the msg_cache and cf_cache contain the necessary activations.
			- When message is None, the method will cache the output in msg_cache or cf_cache if not already present.
		"""
		if message is None:
			if self.patch_type == 'zero':
				embedding = self.msg_cache["hook_embed"].detach().clone()
			elif self.patch_type == 'counterfactual':
				embedding = self.msg_cache["hook_embed"].detach().clone() - self.cf_cache["hook_embed"].detach().clone()
			else:
				raise ValueError(f"Unknown patch type: {self.patch_type}")
		else:
			embedding = message
		if self.position is not None:
			embedding[:, :self.position, :] = torch.zeros_like(embedding[:, :self.position, :], device=embedding.device)
			embedding[:, self.position + 1:, :] = torch.zeros_like(embedding[:, self.position + 1:, :], device=embedding.device)
		return embedding

	
	def calculate_gradient(self, grad_outputs=None, save=True, use_precomputed=False) -> torch.Tensor:
		"""
		Calculates the gradient of the node's input with respect to the final output.
		By default the gradient is calculated propagating backwards from the parent node if present,
		or assuming a gradient of ones if self has no parent. When 'grad_outputs' is specified this is used.

		Args:
			grad_outputs : torch.Tensor, optional (default=None)
				Usually the gradient to propagate backwards in this particular case it is never used.
			
			save : bool, optional (default=True)
				Whether to save the computed gradient in self.gradient. The gradient can be reused
				later by setting use_precomputed to True.
			
			use_precomputed : bool, optional (default=False)
				Whether to use the precomputed gradient if available. The precoputed gradient is stored whenever
				save is True.
		
		Returns:
			gradient : torch.Tensor
				A tensor representing the gradient of the output with respect to the input
				of this node, passing trough the path from final node to the current one.
		
		Notes:
			- Given that the EMBED node is a dummy node, the gradient is simply the one provided or the one from the parent node. The only modification is to zero out the gradient for positions not equal to self.position if specified.
		"""
		if self.gradient is not None and use_precomputed:
			if self.position is None:
				return self.gradient.detach().clone()
			gradient = self.gradient.detach().clone()
			out = torch.zeros_like(self.msg_cache[self.input_name], device=gradient.device)
			out[:, self.position, :] = gradient[:, self.position, :]
			return out
		if grad_outputs is None:
			gradient = self.parent.calculate_gradient(grad_outputs=None, save=True, use_precomputed=True) if self.parent is not None else torch.ones_like(self.msg_cache[self.input_name])
		else:
			gradient = grad_outputs
		if self.position is not None:
			gradient[:, :self.position, :] = torch.zeros_like(gradient[:, :self.position, :], device=gradient.device)
			gradient[:, self.position + 1:, :] = torch.zeros_like(gradient[:, self.position + 1:, :], device=gradient.device)
		if save:
			if self.position is None:
				self.gradient = gradient.detach().clone()
			else:
				self.gradient = gradient[:, self.position, :].detach().clone()
		return gradient.detach().clone()


	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, include_head: bool = False, separate_kv: bool = False) -> list[Node]:
		"""
		Returns the list of predecessors nodes in the computational graph whose outputs influence the
		output of this node. 
		Given that this is an EMBED node, there are no predecessors, so the method returns an empty list.
		Args:

			model_cfg (HookedTransformerConfig):
				The configuration of the transformer model. It is used to determine the number of heads and other model parameters.

			include_head (bool, default=False):
				Whether to consider specific head nodes for ATTN.

			separate_kv (bool, default=False):
				Whether to consider key and value positions separately for ATTN nodes 
		
		Returns:
			list of Node:
				The list of all predecessor, which is always empty for EMBED nodes.
		"""	
		return []

	def __repr__(self) -> str:
		"""String representation of the EMBED_Node instance.
		Includes layer and position.

		Returns:
			str:
				A string representation of the EMBED_Node instance.
		"""
		return f"EMBED_Node(layer={self.layer}, position={self.position})"

	def __hash__(self) -> int:
		"""Hash function for the EMBED_Node instance.
		Returns:
			int:
				A hash value based on the layer and position.
		"""
		return hash((type(self).__name__, self.layer, self.position))


class FINAL_Node(Node):
	"""Represents the final node in the transformer. This is almost a dummy node, as it only serves as the final point for paths that begin at the input embeddings. This classes uses cached activations from the model to provide an interface consistent with other node types.

	Attributes:
		model (HookedTransformer): 
			The transformer model instance. It is assumed to be a HookedTransformer from transformer_lens library. Any other implementation which provide the same interface should work as well.
		layer (int): 
			Layer index in the transformer. Embedding layer is assumed to be layer 0.
		position (int, default=None): 
			Token position if position-specific, else None. None is equivalent to all positions.
		parent (Node, default=None): 
			Parent node in the next node in the path. The parent is a successor in the computational graph.
		children (set, default=set()): 
			Set of child nodes. A child is a predecessor in the computational graph.
		msg_cache (dict): 
			Clean activation cache. Can be obtained by running the model with hooks using the clean prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads). 
		cf_cache (dict, default={}): 
			Counterfactual activation cache. Can be obtained by running the model with hooks using the corrupted prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads).
		gradient (torch.Tensor, default=None): 
			Node cached gradient. Usually is used to represent the gradient of the final output with respect to the input of this node, passing trough the path from final node to the current one.
		input_name (str): 
			Input activation name. This is the name associated to the cache entry corresponding to the input of this node.
		output_name (str): 
			Output activation name. This is the name associated to the cache entry corresponding to the output of this node.
		patch_type (str, default='zero'): 
			Type of intervention ('zero' or 'counterfactual'). Zero patching corresponds to removing the message from the first node in the path to the input of the next node, while counterfactual patching corresponds to replacing the message with the counterfactual activation. In both cases the effect of the path is then calculated by propagating the message through the whole path.
	"""
	def __init__(self, model: HookedTransformer, layer: int, metric: callable = None, position: Optional[int] = None, parent: Node = None, children = set(), msg_cache = {}, cf_cache = {}, gradient = None, patch_type = 'zero'):
		"""Initializes the FINAL_Node instance.
		
		Args:
			model (HookedTransformer): 
				The transformer model instance. It is assumed to be a HookedTransformer from transformer_lens library. Any other implementation which provide the same interface should work as well.
			layer (int): 
				Layer index in the transformer. Embedding layer is assumed to be layer 0.
			metric (callable):
				A callable that takes as input a tensor of shape (batch_size, seq_len, d_model) and returns a scalar tensor.
				It is used to compute the gradient of the output with respect to the input of this node.
			position (int, default=None): 
				Token position if position-specific, else None. None is equivalent to all positions.
			parent (Node, default=None): 
				Parent node in the next node in the path. The parent is a successor in the computational graph.
			children (set, default=set()): 
				Set of child nodes. A child is a predecessor in the computational graph.
			msg_cache (dict): 
				Clean activation cache. Can be obtained by running the model with hooks using the clean prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads). 
			cf_cache (dict, default={}): 
				Counterfactual activation cache. Can be obtained by running the model with hooks using the corrupted prompt and converting the result to a dictionary. It must be a dictionary because it might be modified by adding new cached entries, corresponding to the outputs of subcomponents (e.g. single attention heads).
			gradient (torch.Tensor, default=None): 
				Node cached gradient. Usually is used to represent the gradient of the final output with respect to the input of this node, passing trough the path from final node to the current one.
			patch_type (str, default='zero'): 
				Type of intervention ('zero' or 'counterfactual'). Zero patching corresponds to removing the message from the first node in the path to the input of the next node, while counterfactual patching corresponds to replacing the message with the counterfactual activation. In both cases the effect of the path is then calculated by propagating the message through the whole path.
		
		Returns:
			self (FINAL_Node):
				An instance of the FINAL_Node class.
		"""
		if layer != model.cfg.n_layers - 1:
			print(f"WARNING: FINAL_Node should be initialized with the last layer index ({model.cfg.n_layers - 1}), got {layer}")
		super().__init__(model=model, layer=layer, position=position, parent=parent, children=children, msg_cache=msg_cache, cf_cache=cf_cache, gradient=gradient, input_name=f"blocks.{layer}.hook_resid_post", output_name=f"blocks.{layer}.hook_resid_post", patch_type=patch_type)
		self.metric = metric
	
	def forward(self, message: torch.Tensor = None) -> torch.Tensor:
		"""
		Calculate the effect of the message on the output of the node. 
		
		The effect is calculated indirectly as the difference between the normal output of the component and the 
		one obtained when the message is removed from the input of the node.
		On the other hand, if message is None the behavior depends on the patch_type:
		- 'zero': returns the normal output of the component
		- 'counterfactual': returns the difference between the normal output and the counterfactual output of the component

		Args:

			message (torch.Tensor of shape (batch_size, seq_len, d_model), default=None):
				The message whose effect on the node need to be evaluated. If None, returns the normal 
				output or the difference between normal and counterfactual output depending on patch_type.
			
		Returns:
			torch.Tensor:
				A tensor representing the effect of the message on the output of the node.
				In simpler terms, it represents the message caused by passing the input message through this node.

		Notes:
			- If a position is specified the output will be zero for all other positions.
			- The method assumes that the msg_cache and cf_cache contain the necessary activations.
			- When message is None, the method will cache the output in msg_cache or cf_cache if not already present.
		"""
		if message is None:
			if self.patch_type == 'zero' or self.patch_type == 'counterfactual':
				res = self.msg_cache[self.input_name].detach().clone()
			else:
				raise ValueError(f"Unknown patch type: {self.patch_type}")
		else:
			res = message.detach().clone()
		if self.position is not None:
			res_zeroed = torch.zeros_like(res, device=res.device)
			res_zeroed[:, self.position, :] = res[:, self.position, :]
			return res_zeroed
		return res

	
	def calculate_gradient(self, grad_outputs=None, save=True, use_precomputed=False, metric=None) -> torch.Tensor:
		"""
		Calculates the gradient of the node's input with respect to the final output.
		By default the gradient is calculated propagating backwards from the parent node if present,
		or assuming a gradient of ones if self has no parent. When 'grad_outputs' is specified, it is used instead of the parent's gradient.

		Args:
			grad_outputs : torch.Tensor, optional (default=None)
				Usually the gradient to propagate backwards in this particular case it is never used.
			
			save : bool, optional (default=True)
				Whether to save the computed gradient in self.gradient. The gradient can be reused
				later by setting use_precomputed to True.
			
			use_precomputed : bool, optional (default=False)
				Whether to use the precomputed gradient if available. The precoputed gradient is stored whenever
				save is True.
			
			metric : callable, optional (default=None)
				A callable that takes as input a tensor of shape (batch_size, seq_len, d_model) and returns a scalar tensor.
				It is used to compute the gradient of the output with respect to the input of this node.
				If None, uses the metric provided at initialization. If neither is provided, raises an error.
		
		Returns:
			gradient : torch.Tensor
				A tensor representing the gradient of the output with respect to the input
				of this node, passing trough the path from final node to the current one.
		"""
		if self.gradient is not None and use_precomputed:
			return self.gradient.detach().clone()
		if metric is None:
			metric = self.metric
		if metric is None:
			raise NotImplementedError("FINAL_Node.calculate_gradient() requires to provide a metric either at initialization or as a parameter")
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

	def get_expansion_candidates(self, model_cfg: HookedTransformerConfig, include_head: bool = True, separate_kv: bool = False) -> list[Node]:
		"""
		Returns the list of predecessors nodes in the computational graph whose outputs influence the output of this node. 
		For the FINAL node, these are all MLP, EMBED and ATTN nodes from all layers.

		Args:

			model_cfg (HookedTransformerConfig):
				The configuration of the transformer model. It is used to determine the number of heads and other model parameters.

			include_head (bool, default=False):
				Whether to consider specific head nodes for ATTN.

			separate_kv (bool, default=False):
				Whether to consider key and value positions separately for ATTN nodes 
		
		Returns:
			list of Node:
				The list of all nodes.
		
		Notes:
			- If self.position is None, only non-position-specific previous nodes are considered.
		"""
		prev_nodes = []
		common_args = {"model": self.model, "msg_cache": self.msg_cache, "parent": self, "patch_type": self.patch_type, "cf_cache": self.cf_cache}

		for l in range(model_cfg.n_layers):
			# MLPs
			prev_nodes.append(MLP_Node(layer=l, position=self.position, **common_args))

			# ATTN query positions
			if include_head:
				prev_nodes.extend([ATTN_Node(layer=l, head=h, keyvalue_position=None, position=self.position,  patch_key=False, patch_value=False, patch_query=True, **common_args) for h in range(model_cfg.n_heads)])
			else:
				prev_nodes.append(ATTN_Node(layer=l, head=None, keyvalue_position=None, position=self.position,  patch_key=False, patch_value=False, patch_query=True, **common_args))

			# ATTN key-value positions
			if self.position is not None:
				for keyvalue_position in range(self.position + 1):
					if include_head:
						if separate_kv:
							prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
							prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
						else:
							prev_nodes.extend([ATTN_Node(layer=l, head=h, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
					else:
						if separate_kv:
							prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=False, patch_query=False, **common_args))
							prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_key=False, patch_value=True, patch_query=False, **common_args))
						else:
							prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=keyvalue_position, patch_key=True, patch_value=True, patch_query=False, **common_args))
			else:
				if include_head:
					if separate_kv:
						prev_nodes.extend([ATTN_Node(layer=l, head=h, position=None, keyvalue_position=None, patch_key=True, patch_value=False, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
						prev_nodes.extend([ATTN_Node(layer=l, head=h, position=None, keyvalue_position=None, patch_key=False, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
					else:
						prev_nodes.extend([ATTN_Node(layer=l, head=h, position=None, keyvalue_position=None, patch_key=True, patch_value=True, patch_query=False, **common_args) for h in range(model_cfg.n_heads)])
				else:
					if separate_kv:
						prev_nodes.append(ATTN_Node(layer=l, head=None, position=None, keyvalue_position=None, patch_key=True, patch_value=False, patch_query=False, **common_args))
						prev_nodes.append(ATTN_Node(layer=l, head=None, position=None, keyvalue_position=None, patch_key=False, patch_value=True, patch_query=False, **common_args))
					else:
						prev_nodes.append(ATTN_Node(layer=l, head=None, position=self.position, keyvalue_position=None, patch_key=True, patch_value=True, patch_query=False, **common_args))

		prev_nodes.append(EMBED_Node(layer=0, position=self.position, **common_args))
		# Remove duplicates
		prev_nodes = list(set(prev_nodes))
		return prev_nodes

	def __repr__(self) -> str:
		"""Returns a string representation of the FINAL_Node instance.
		Includes layer and position if specified.
		Returns:
			str:
				A string representation of the FINAL_Node instance.
		"""
		pos_str = f", position={self.position}" if self.position is not None else ""
		return f"FINAL_Node(layer={self.layer}{pos_str})"

	def __hash__(self) -> int:
		"""Hash function for the FINAL_Node instance.
		Returns:
			int:
				A hash value based on the layer and position.
		"""
		return hash((type(self).__name__, self.layer, self.position))
