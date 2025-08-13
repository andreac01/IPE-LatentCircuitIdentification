import einops
import torch
import torch.nn.functional as F
from transformer_lens.components.abstract_attention import AbstractAttention


def custom_attention_forward(
	attention_module: AbstractAttention,
	head: int,
	q,
	k,
	v,
	precomputed_attention_scores: torch.Tensor = None,
	query_position: int = None,
	keyvalue_position: int = None,
) -> torch.Tensor:

	if attention_module.cfg.positional_embedding_type == "rotary":
		raise NotImplementedError(
			"Rotary positional embeddings are not supported in this function. Use the full attention module instead."
		)
	# print(f"q shape: {q.shape}, dtype: {q.dtype}")
	# print(f"k shape: {k.shape}, dtype: {k.dtype}")
	# print(f"v shape: {v.shape}, dtype: {v.dtype}")
	if attention_module.cfg.dtype not in [torch.float32, torch.float64]:
		# If using 16 bits, increase the precision to avoid numerical instabilities
		q = q.to(torch.float32)
		k = k.to(torch.float32)
		v = v.to(torch.float32)
	
	if precomputed_attention_scores is not None:
		attn_scores_new = attention_module.calculate_attention_scores(q, k)
		# print(f"attn_scores_new shape: {attn_scores_new.shape}, dtype: {attn_scores_new.dtype}")
		if head is None:
			attn_scores = precomputed_attention_scores
		else:
			attn_scores = precomputed_attention_scores[:, head:head+1, :, :]
		# print(f"attn_scores shape: {attn_scores.shape}, dtype: {attn_scores.dtype}")
		if query_position is not None:
			if keyvalue_position is not None:
				# print(f"query_position: {query_position}, keyvalue_position: {keyvalue_position}")
				# print(f"Changing attn_score from {attn_scores[:, 0, query_position, keyvalue_position]} to {attn_scores_new[:, 0, 0, 0]}")
				if head is None:
					attn_scores[:, :, query_position, keyvalue_position] = attn_scores_new[:, :, 0, 0]
				else:
					attn_scores[:, 0, query_position, keyvalue_position] = attn_scores_new[:, 0, 0, 0]
			else:
				if head is None:
					attn_scores[:, :, query_position, :k.shape[1]] = attn_scores_new[:, :, 0, :]
				else:
					attn_scores[:, 0, query_position, :k.shape[1]] = attn_scores_new[:, 0, 0, :]
			attn_scores = attn_scores[:, :, query_position].unsqueeze(2)

		elif keyvalue_position is not None:
			if head is None:
				attn_scores[:, :, :, keyvalue_position] = attn_scores_new[:, :, :, 0]
			else:
				attn_scores[:, 0, :, keyvalue_position] = attn_scores_new[:, 0, 0, 0]
		else:
			if head is None:
				attn_scores[:, :, :, :] = attn_scores_new[:, :, 0, :]
			else:
				attn_scores[:, 0, :, :] = attn_scores_new[:, 0, :, :]
	else:
		attn_scores = attention_module.calculate_attention_scores(q, k)  # [batch, 1, query_pos, key_pos]

	# print(f"attn_scores shape: {attn_scores.shape}, dtype: {attn_scores.dtype}")
	# print(f"attn_scores_new shape: {attn_scores_new.shape}, dtype: {attn_scores_new.dtype}" if 'attn_scores_new' in locals() else "")
	# print(f"q shape: {q.shape}, dtype: {q.dtype}")
	# print(f"k shape: {k.shape}, dtype: {k.dtype}")
	# print(f"v shape: {v.shape}, dtype: {v.dtype}")

	if attention_module.cfg.positional_embedding_type == "alibi":
		raise NotImplementedError(
			"ALiBi positional embeddings are not supported in this function. Use the full attention module instead."
		)
	elif attention_module.cfg.positional_embedding_type == "relative_positional_bias":
		raise NotImplementedError(
			"Relative positional bias is not supported in this function. Use the full attention module instead."
		)

	if attn_scores.shape[-1] == attn_scores.shape[-2]:
		attn_scores = attention_module.apply_causal_mask(attn_scores)

	pattern = F.softmax(attn_scores, dim=-1)
	# print(f"pattern shape: {pattern.shape}, dtype: {pattern.dtype}, pattern: {pattern}")
	# print(pattern.shape, pattern.dtype)
	pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
	pattern = pattern.to(v.device)
	# print(f"pattern shape: {pattern.shape}, dtype: {pattern.dtype}, pattern: {pattern}")
	# print(v.shape, pattern.shape, v.dtype, pattern.dtype)
	z_head = attention_module.calculate_z_scores(v, pattern)  # [batch, pos, 1, d_head]
	# print(f"z_head shape: {z_head.shape}, dtype: {z_head.dtype}")


	z = torch.zeros((z_head.shape[0], z_head.shape[1], attention_module.cfg.n_heads, attention_module.cfg.d_head), device=z_head.device, dtype=z_head.dtype)
	# print(f"z shape: {z.shape}, dtype: {z.dtype}")
	if head is None:
		z = z_head
	else:
		z[:, :, head, :] = z_head[:, :, 0, :]
	# print(z[:,:,head], z.shape, z.dtype)
	w = einops.rearrange(
		attention_module.W_O, "head_index d_head d_model -> d_model (head_index d_head)"
	).to(torch.float32)
	out = F.linear(
		z.reshape(z.shape[0], z.shape[1], attention_module.cfg.d_head * attention_module.cfg.n_heads),
		w,
	)
	out = out.to(attention_module.cfg.dtype)

	if query_position is not None:
		out = out[:, 0]
		# print(f"out shape after query_position: {out.shape}, dtype: {out.dtype}")
	return out