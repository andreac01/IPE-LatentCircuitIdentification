from itertools import islice
from transformer_lens import HookedTransformer
from torch import Tensor
import torch
import inspect
from typing import Iterable

def get_function_params(func: callable, which: str='all') -> dict[str, inspect.Parameter]:
	"""
	Returns a dictionary of all parameters for a given function.
	Keys are parameter names, values are inspect.Parameter objects.

	Args:
		func: The function to inspect.
		which: 'all' to get all parameters, 'required' to get only required parameters, 'default' to get only parameters with default values.
	Returns:
		dict: A dictionary of parameter names and their default values.
	"""
	sig = inspect.signature(func)
	if which == 'required':
		return {k: v for k, v in sig.parameters.items() if v.default == inspect.Parameter.empty}
	elif which == 'default':
		return {k: v.default for k, v in sig.parameters.items() if v.default != inspect.Parameter.empty}
	elif which == 'all':
		return {k: v for k, v in sig.parameters.items()}
	else:
		raise ValueError("Parameter 'which' must be one of 'all', 'required', or 'default'.")

def batch_iterable(iterable: Iterable, batch_size: int):
	"""Batch an iterable into chunks of a specified size.
	Args:
		iterable (iterable): 
			The input iterable to be batched.
		batch_size (int): 
			The size of each batch.
	Yields:
		list: A batch of elements from the iterable.
	"""
	it = iter(iterable)
	while True:
		chunk = list(islice(it, batch_size))
		if not chunk:
			break
		yield chunk

def get_topk(model: HookedTransformer, residual: Tensor, topk=5) -> dict[list]:
	"""Get the top-k token predictions from the model's output logits.
	Args:
		model (HookedTransformer): 
			The transformer model.
		residual (Tensor): 
			The residual stream tensor of shape (d_model,).
		topk (int): 
			The number of top predictions to return.
	Returns:
		dict: A dictionary containing top-k indices, logits, probabilities, and string tokens.
			- 'topk_indices': The top-k token indices.
			- 'topk_logits': The top-k logits.
			- 'topk_probs': The top-k probabilities.
			- 'topk_strtokens': The top-k string representations of the tokens.
	"""
	assert residual.dim() == 1, "Residual must be a 1D tensor of shape (d_model,)"
	resid_norm = model.ln_final(residual)
	logits = model.unembed(resid_norm)
	probabilities = torch.softmax(logits, dim=-1)

	topk_indices = torch.topk(logits, topk, dim=-1).indices

	# Get topk values and indices
	topk_values, topk_indices = torch.topk(logits, topk, dim=-1)
	topk_logits = topk_values
	topk_probs = torch.gather(probabilities, 0, topk_indices)
	topk_strtokens = [model.tokenizer.decode([int(idx)]).replace(' ', '_') for idx in topk_indices] 

	return {
		"topk_indices": topk_indices.detach().cpu().numpy().tolist(),
		"topk_logits": topk_logits.detach().cpu().numpy().tolist(),
		"topk_probs": topk_probs.detach().cpu().numpy().tolist(),
		"topk_strtokens": topk_strtokens
	}