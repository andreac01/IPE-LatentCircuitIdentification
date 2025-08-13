import torch
from torch import Tensor
from typing import Any

def attention_score_patch_hook(residual: Tensor, hook: Any, new_scores: Tensor, head: int, in_pos: int, out_pos: int) -> Tensor:
	"residual has shape [bs, nheads, seq, seq]"
	residual[:, head, out_pos, in_pos] = new_scores

def zero_patch_head_hook(residual: Tensor, hook: Any, head: int) -> Tensor:
	"""
	zero_patch_head_hook is a hook function that sets the values of a specific head in the residual tensor to zero.

	Given a residual tensor, this function will set the values of the specified head to zero, effectively removing its contribution from the final output.
	The hook takes the residual at a specific hook point from TransformerLens library and modifies it in place.

	Args:
		residual (Tensor): The residual tensor that is being modified. It has the shape (batch_size, sequence_length, num_heads, head_dim).
		hook (Any): The hook object that is passed to the function. It is not used in this function, but it is required by the hook signature.
		head (int): The index of the head that is being zeroed out. It should be an integer between 0 and num_heads - 1.

	Returns:
		Tensor: The modified residual tensor with the specified head set to zero.
	"""
	residual[:, :, head, :] = torch.zeros_like(residual[:, :, head, :])
	return residual

def remove_patch_head_hook(residual: Tensor, hook: Any, head: int, patch_to_remove: Tensor) -> Tensor:
	"""
	remove_patch_head_hook is a hook function that removes a specific patch from the residual tensor for a given head.

	Given a residual tensor and a patch to remove, this function will subtract the patch from the specified head in the residual tensor.
	The hook takes the residual at a specific hook point from TransformerLens library and modifies it in place.
	Args:
		residual (Tensor): The residual tensor that is being modified. It has the shape (batch_size, sequence_length, num_heads, head_dim).
		hook (Any): The hook object that is passed to the function. It is not used in this function, but it is required by the hook signature.
		head (int): The index of the head that is being modified. It should be an integer between 0 and num_heads - 1.
		patch_to_remove (Tensor): The patch tensor that is being removed from the residual. It should have the same shape as the residual for the specified head.
		It is expected to have the shape (batch_size, sequence_length, head_dim).

	Returns:
		Tensor: The modified residual tensor with the specified patch removed from the specified head.
	"""
	residual[:, :, head, :] = residual[:, :, head, :] - patch_to_remove
	return residual

def zero_patch_except_head_hook(residual: Tensor, hook: Any, head: int) -> Tensor:
	"""
	zero_patch_except_head_hook is a hook function that sets the values of all heads except a specific head in the residual tensor to zero.

	Given a residual tensor, this function will set the values of all heads except the specified head to zero, effectively removing their contribution from the final output.
	The hook takes the residual at a specific hook point from TransformerLens library and modifies it in place. It is used to analyze the contribution of a specific head in the attention mechanism of a transformer model.

	Args:
		residual (Tensor): The residual tensor that is being modified. It has the shape (batch_size, sequence_length, num_heads, head_dim).
		hook (Any): The hook object that is passed to the function. It is not used in this function, but it is required by the hook signature.
		head (int): The index of the head that is being kept. It should be an integer between 0 and num_heads - 1.

	Returns:
		Tensor: The modified residual tensor with all heads except the specified head set to zero.
	"""
	residual[:, :, :head, :] = torch.zeros_like(residual[:, :, :head, :])
	residual[:, :, head + 1:, :] = torch.zeros_like(residual[:, :, head + 1:, :])
	return residual

def zero_patch_except_pos_hook(residual: Tensor, hook: Any, pos: int) -> Tensor:
	"""
	zero_patch_except_pos_hook is a hook function that sets the values of all positions except a specific position in the residual tensor to zero.

	Given a residual tensor, this function will set the values of all positions except the specified position to zero, effectively removing their contribution from the final output.
	The hook takes the residual at a specific hook point from TransformerLens library and modifies it in place. It is used to analyze the contribution of a specific position in the attention mechanism of a transformer model.

	Args:
		residual (Tensor): The residual tensor that is being modified. It has the shape (batch_size, sequence_length, num_heads, head_dim).
		hook (Any): The hook object that is passed to the function. It is not used in this function, but it is required by the hook signature.
		pos (int): The index of the position that is being kept. It should be an integer between 0 and sequence_length - 1.

	Returns:
		Tensor: The modified residual tensor with all positions except the specified position set to zero.
	"""
	residual[:, :pos, :, :] = torch.zeros_like(residual[:, :pos, :, :])
	residual[:, pos + 1:, :, :] = torch.zeros_like(residual[:, pos + 1:, :, :])
	return residual

def zero_patch_except_head_batched_hook(residual: Tensor, hook: Any) -> Tensor:
	"""
	Generate a batch of size batch_size * num_heads, where each head is zeroed out except for the head in the current position//batch_size. 

	This function is similar to zero_patch_except_head_hook but can be used to calculate the contribution of all heads in a batched manner.
	Given a residual tensor, this function will generate the output of zero_patch_except_head_hook for all heads, stacked over the first dimension.

	Args:
		residual (Tensor): The residual tensor that is being modified. It has the shape (batch_size, sequence_length, num_heads, head_dim).
		hook (Any): The hook object that is passed to the function. It is not used in this function, but it is required by the hook signature.

	Returns:
		Tensor: The batch modified residual tensor with all heads except the current head set to zero in a batched manner. It has the shape (batch_size * num_heads, sequence_length, num_heads, head_dim).
	"""
	batch_size, sequence_length, num_heads, head_dim = residual.shape
	residual = torch.cat([residual] * num_heads, dim=0)
	mask = torch.zeros_like(residual)
	for head in range(num_heads):
		mask[head*batch_size:(head+1)*batch_size, :, head, :] = 1
	residual = residual * mask
	return residual

def zero_patch_except_pos_batched_hook(residual: Tensor, hook: Any) -> Tensor:
	""" 
	Generate a batch of size batch_size * sequence_length, where each position is zeroed out except for the position in the current position//batch_size.
	
	This function is similar to zero_patch_except_pos_hook but can be used to calculate the contribution of all positions in a batched manner.
	Given a residual tensor, this function will generate the output of zero_patch_except_pos_hook for all positions, stacked over the first dimension.
	
	Args:
		residual (Tensor): The residual tensor that is being modified. It has the shape (batch_size, sequence_length, num_heads, head_dim).
		hook (Any): The hook object that is passed to the function. It is not used in this function, but it is required by the hook signature.
	Returns:
		Tensor: The batch modified residual tensor with all positions except the current position set to zero in a batched manner. It has the shape (batch_size * sequence_length, sequence_length, num_heads, head_dim).
	"""
	batch_size, sequence_length, num_heads, head_dim = residual.shape
	residual = torch.cat([residual] * sequence_length, dim=0)
	mask = torch.zeros_like(residual)
	for pos in range(sequence_length):
		mask[pos*batch_size:(pos+1)*batch_size, pos, :, :] = 1
	residual = residual * mask
	return residual

def zero_patch_except_pos_and_head_batched_hook(residual: Tensor, hook: Any) -> Tensor:
	"""
	Generate a batch of size batch_size * sequence_length * num_heads, where each position and head is zeroed out except for the position and head in the current position//batch_size.

	This function is similar to zero_patch_except_pos_hook but can be used to calculate the contribution of all positions and heads in a batched manner.
	Given a residual tensor, this function will generate the output of zero_patch_except_pos_hook for all positions and heads, stacked over the first dimension.

	Args:
		residual (Tensor): The residual tensor that is being modified. It has the shape (batch_size, sequence_length, num_heads, head_dim).
		hook (Any): The hook object that is passed to the function. It is not used in this function, but it is required by the hook signature.

	Returns:
		Tensor: The batch modified residual tensor with all positions and heads except the current position and head set to zero in a batched manner. It has the shape (batch_size * sequence_length * num_heads, sequence_length, num_heads, head_dim).
	"""
	batch_size, sequence_length, num_heads, head_dim = residual.shape
	residual = torch.cat([residual] * (sequence_length * num_heads), dim=0)
	mask = torch.zeros_like(residual)
	for pos in range(sequence_length):
		for head in range(num_heads):
			mask[pos*batch_size:(pos+1)*batch_size, pos, head, :] = 1
	residual = residual * mask
	return residual

def stack_hook(residual: Tensor, hook: Any, n, dim=0) -> Tensor:
	"""
	stack_hook is a hook function that stacks the residual tensor along a specified dimension.

	Given a residual tensor, this function will stack the tensor n times along the specified dimension. This is useful for analyzing the contribution of multiple heads or positions in the attention mechanism of a transformer model.

	Args:
		residual (Tensor): The residual tensor that is being modified. It has the shape (batch_size, sequence_length, num_heads, head_dim).
		hook (Any): The hook object that is passed to the function. It is not used in this function, but it is required by the hook signature.
		n (int): The number of times to stack the residual tensor.
		dim (int): The dimension along which to stack the tensor. Defaults to 0.

	Returns:
		Tensor: The stacked residual tensor.
	"""
	return torch.cat([residual] * n, dim=dim)