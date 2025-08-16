import einops
import torch
import torch.nn.functional as F
from transformer_lens.components.abstract_attention import AbstractAttention
import matplotlib.pyplot as plt 

def custom_attention_forward(
	attention_module: AbstractAttention,
	head: int,
	q,
	k,
	v,
	precomputed_attention_scores: torch.Tensor = None,
	query_position: int = None,
	keyvalue_position: int = None,
	plot_patterns: bool = False,
	add_bias: bool = True,
) -> torch.Tensor:
	if attention_module.cfg.positional_embedding_type == "rotary":
		raise NotImplementedError(
			"Rotary positional embeddings are not supported in this function. Use the full attention module instead."
		)
	if attention_module.cfg.dtype not in [torch.float32, torch.float64]:
		# If using 16 bits, increase the precision to avoid numerical instabilities
		q = q.to(torch.float32)
		k = k.to(torch.float32)
		v = v.to(torch.float32)
	
	if precomputed_attention_scores is not None:
		attn_scores_new = attention_module.calculate_attention_scores(q, k)
		if head is None:
			attn_scores = precomputed_attention_scores
		else:
			attn_scores = precomputed_attention_scores[:, head:head+1, :, :]
		if query_position is not None:
			if keyvalue_position is not None:
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
				attn_scores[:, :, :, :] = attn_scores_new[:, :, :, :]
			else:
				attn_scores[:, 0, :, :] = attn_scores_new[:, 0, :, :]
	else:
		attn_scores = attention_module.calculate_attention_scores(q, k)  # [batch, 1, query_pos, key_pos]

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
	pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
	pattern = pattern.to(v.device)
	z_head = attention_module.calculate_z_scores(v, pattern)  # [batch, pos, 1, d_head]

	if plot_patterns:
		for h in range(pattern.shape[1]):
			plt.figure(figsize=(10, 5))
			plt.imshow(pattern[0][h].detach().numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=1)
			plt.colorbar()
			title = f'Head {h} Pattern' if head is None else f'Head {head} Pattern'
			plt.title(title)
			kv_pos = 'Key Position' if keyvalue_position is None else f'Key Position {keyvalue_position}'
			q_pos = 'Query Position' if query_position is None else f'Query Position {query_position}'
			plt.xlabel(kv_pos)
			plt.ylabel(q_pos)
			plt.show()

	z = torch.zeros((z_head.shape[0], z_head.shape[1], attention_module.cfg.n_heads, attention_module.cfg.d_head), device=z_head.device, dtype=z_head.dtype)
	if head is None:
		z = z_head
	else:
		z[:, :, head, :] = z_head[:, :, 0, :]
	w = einops.rearrange(
		attention_module.W_O, "head_index d_head d_model -> d_model (head_index d_head)"
	).to(torch.float32)
	out = F.linear(
		z.reshape(z.shape[0], z.shape[1], attention_module.cfg.d_head * attention_module.cfg.n_heads),
		w,
		# attention_module.b_O.to(torch.float32) if add_bias else None
	)
	out = out.to(attention_module.cfg.dtype)
	if query_position is not None:
		out = out[:, 0]
	return out