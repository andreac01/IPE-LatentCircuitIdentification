import torch
from transformer_lens import HookedTransformer
from typing import List, Optional
from torch import Tensor
import torch.nn.functional as F

def target_probability_percentage(clean_final_resid: Tensor,
								corrupted_resid: Tensor,
								model: HookedTransformer,
								target_tokens: List[int]) -> Tensor:
	"""
	Compute the difference in the probability of associated with the target token
	when decoding the clean and corrupted residuals. 
	This probability is returned as a percentage of the clean model's probability.

	Args:
		clean_final_resid (torch.Tensor): 
			The final residual stream of the clean model.
			Shape: (batch, seq_len, d_model).
		corrupted_resid (torch.Tensor): 
			The final residual stream of the corrupted model.
			Shape: (batch, seq_len, d_model).
		model (HookedTransformer): 
			The hooked transformer model.
		target_tokens (List[int]): 
			The indexes of the target tokens.

	Returns:
		float: 
			The difference in probability of predicting the target token.
	"""
	# Get logits for the last token
	clean_final_resid = model.ln_final(clean_final_resid[:, -1, :])
	corrupted_resid = model.ln_final(corrupted_resid[:, -1, :])
	clean_logits = model.unembed(clean_final_resid)
	corrupted_logits = model.unembed(corrupted_resid)

	# Get the probability of the target token
	prob_clean = F.softmax(clean_logits, dim=-1)[..., target_tokens]
	prob_corrupted = F.softmax(corrupted_logits, dim=-1)[..., target_tokens]

	return torch.mean(100*(prob_clean - prob_corrupted)/prob_clean)

def target_logit_percentage(clean_final_resid: Tensor,
						corrupted_resid: Tensor,
						model: HookedTransformer,
						target_tokens: List[int]) -> Tensor:
	"""
	Compute the difference in logits for the target token as a percentage
	between the logit obtained from the decoding of the clean and corrupted residual.
	This implementation is optimized for transformerlens HookedTransformer.

	Args:
		clean_final_resid (torch.Tensor): 
			The final residual stream of the clean model.
			Shape: (batch, seq_len, d_model).
		corrupted_resid (torch.Tensor): 
			The final residual stream of the corrupted model.
			Shape: (batch, seq_len, d_model).
		model (HookedTransformer): 
			The hooked transformer model.
		target_tokens (List[int]): 
			The indexes of the target tokens.

	Returns:
		torch.Tensor: 
			The percentage difference in logits for the target token.
	"""
	
	# Get the unembedding weights and bias
	W_U = model.W_U
	b_U = model.b_U

	# Get the final residual stream for the last token
	clean_final_resid = clean_final_resid[:, -1, :]
	corrupted_final_resid = corrupted_resid[:, -1, :]
	
	# Apply the layer norm to the final residuals
	clean_final_resid = model.ln_final(clean_final_resid)
	corrupted_final_resid = model.ln_final(corrupted_final_resid)
	
	# Get the logits associated with the residuals
	clean_logits = torch.einsum('b d, d b-> b', clean_final_resid, W_U[:, target_tokens]) + b_U[target_tokens]
	corrupted_logits = torch.einsum('b d, d b-> b', corrupted_final_resid, W_U[:, target_tokens]) + b_U[target_tokens]
	# Calculate the percentage difference
	#print(f"Clean logits: {clean_logits.mean().item()}, Corrupted logits: {corrupted_logits.mean().item()}")
	percentage_diffs = 100 * (clean_logits - corrupted_logits) / (torch.abs(clean_logits))
	return torch.mean(percentage_diffs)

def kl_divergence(clean_final_resid: Tensor,
				corrupted_resid: Tensor,
				model: HookedTransformer) -> Tensor:
	"""
	Compute the KL divergence between the output distributions of the clean and corrupted residuals.
	The implementation is particularly useful when the target token is not known in advance.
	This implementation is optimized for transformerlens HookedTransformer.

	Args:
		clean_final_resid (torch.Tensor): 
			The final residual stream of the clean model.
			Shape: (batch, seq_len, d_model).
		corrupted_resid (torch.Tensor): 
			The final residual stream of the corrupted model.
			Shape: (batch, seq_len, d_model).
		model (HookedTransformer): 
			The hooked transformer model.
	Returns:
		torch.Tensor: 
			The KL divergence between the output distributions.
	"""
	clean_final_resid = clean_final_resid[:, -1, :]
	corrupted_final_resid = corrupted_resid[:, -1, :]

	clean_normed = model.ln_final(clean_final_resid)
	corrupted_normed = model.ln_final(corrupted_final_resid)

	clean_logits = model.unembed(clean_normed)
	corrupted_logits = model.unembed(corrupted_normed)

	clean_log_probs = F.softmax(clean_logits, dim=-1)
	corrupted_probs = F.softmax(corrupted_logits, dim=-1)

	kl_divs = F.kl_div(clean_log_probs.log(), corrupted_probs, reduction='batchmean')
	return kl_divs

def indirect_effect(clean_final_resid: Tensor,
					corrupted_resid: Tensor,
					model: HookedTransformer,
					target_tokens: List[int],
					cf_target_tokens: List[int],
					verbose = False) -> Tensor:
	"""
	Compute the Indirect Effect (IE) score.
	IE(z) = 0.5 * [ (P*z(r) - P(r)) / P(r) + (P(r') - P*z(r')) / P*z(r') ]
	This measures how much a component's activation (z) from a corrupted run
	influences the output probabilities on a clean run.

	Args:
		clean_final_resid (torch.Tensor): 
			The final residual stream of the clean model run (prompt p2).
			Shape: (batch, seq_len, d_model).
		corrupted_resid (torch.Tensor): 
			The final residual stream of the corrupted model run (prompt p2 with intervention z from p1).
			Shape: (batch, seq_len, d_model).
		model (HookedTransformer): The hooked transformer model.
		target_tokens (List[int]): 
			The indexes of the target tokens for the clean prompt (r').
		cf_target_tokens (List[int]): 
			The indexes of the target tokens from the corrupted prompt (r).
		verbose (bool, optional): If True, prints intermediate values for debugging. Default is False.

	Returns:
		torch.Tensor: The Indirect Effect score.
	"""

	# Get the final residual stream for the last token
	clean_final_resid = clean_final_resid[:, -1, :]
	corrupted_final_resid = corrupted_resid[:, -1, :]
	
	# Apply the layer norm to the final residuals
	clean_final_resid = model.ln_final(clean_final_resid)
	corrupted_final_resid = model.ln_final(corrupted_final_resid)
	
	# Get the logits for both runs
	clean_logits = model.unembed(clean_final_resid)
	corrupted_logits = model.unembed(corrupted_final_resid)

	# Apply softmax to get probabilities
	clean_probs = F.softmax(clean_logits, dim=-1)
	corrupted_probs = F.softmax(corrupted_logits, dim=-1)

	batch_indices = torch.arange(len(target_tokens))

	# P(r'): Probability of the clean target (r') on a clean run.
	P_r_prime = clean_probs[batch_indices, target_tokens]

	# P(r): Probability of the corrupt target (r) on a clean run.
	P_r = clean_probs[batch_indices, cf_target_tokens]

	# P*z(r'): Probability of the clean target (r') on a corrupted run.
	P_z_star_r_prime = corrupted_probs[batch_indices, target_tokens]
	# P*z(r): Probability of the corrupt target (r) on a corrupted run.
	P_z_star_r = corrupted_probs[batch_indices, cf_target_tokens]

	# Term 1: (P*z(r) - P(r)) / P(r)
	# Relative increase in probability for the new answer (r)
	term1 = (P_z_star_r - P_r) / (P_r + 1e-8)

	# Term 2: (P(r') - P*z(r')) / P*z(r')
	# Change in probability for the original answer (r')
	term2 = (P_r_prime - P_z_star_r_prime) / (P_z_star_r_prime + 1e-8)

	indirect_effects = 0.5 * (term1 + term2)

	if verbose:
		print(f"First prompt top 3 tokens: {torch.topk(clean_probs, 3).indices, torch.topk(clean_probs, 3).values}")
		print(f"Target tokens (r'): {target_tokens}")
		print(f"Counterfactual tokens (r): {cf_target_tokens}")
		print(f"P(r): {P_r.mean().item()}, P*z(r): {P_z_star_r.mean().item()}")
		print(f"P(r'): {P_r_prime.mean().item()}, P*z(r'): {P_z_star_r_prime.mean().item()}")
		print(f"Indirect effect: {indirect_effects.mean().item()}")

	return torch.mean(indirect_effects)

def logit_difference(corrupted_resid: Tensor, 
					model: HookedTransformer,
					target_tokens: List[int],
					cf_target_tokens: List[int],
					baseline_value: float = 0.) -> Tensor:
	"""
	Compute logit difference: y' - y between the logit associated with the target token of the
	clean prompt y and y', the answer associated with the counterfactual prompt.
	This function supports both explicit counterfactual comparison and ablation-based comparison.

	Args:
		corrupted_resid (torch.Tensor): 
			The final residual stream of the counterfactual model (or ablated model).
			Shape: (batch, seq_len, d_model).
		model (HookedTransformer): 
			The hooked transformer model.
		target_tokens (List[int]): 
			The indexes of the target tokens for the clean model.
		cf_target_tokens (Optional[List[int]]): 
			The indexes of the target tokens for the counterfactual model.
			Required when use_ablation_mode is False.
		use_ablation_mode (bool): 
			If True, uses ablation-based comparison (counterfactual_resid is the ablated version).
			If False, uses explicit counterfactual comparison.

	Returns:
		float: 
			The logit difference: y' - y.
	"""
	# Get the unembedding weights and bias
	W_U = model.W_U
	b_U = model.b_U

	# Get the final residual stream for the last token
	corrupted_final_resid = corrupted_resid[:, -1, :]
	
	# Apply the layer norm to the final residuals
	corrupted_final_resid = model.ln_final(corrupted_final_resid)
	
	target_logits = torch.einsum('b d, d b-> b', corrupted_final_resid, W_U[:, target_tokens]) + b_U[target_tokens]
	counterfactual_logits = torch.einsum('b d, d b-> b', corrupted_final_resid, W_U[:, cf_target_tokens]) + b_U[cf_target_tokens]
	
	logit_diffs = counterfactual_logits - target_logits
	return torch.mean(logit_diffs) - baseline_value
