from itertools import islice
from transformer_lens import HookedTransformer
from torch import Tensor
import torch

def batch_iterable(iterable, batch_size):
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