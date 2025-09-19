import torch
from transformer_lens import HookedTransformer
from typing import List, Optional
from torch import Tensor
import torch.nn.functional as F

def compare_token_probability(clean_resid: Tensor,
								corrupted_resid: Tensor,
								model: HookedTransformer,
								target_tokens: List[int]) -> Tensor:
	"""
	Compute the difference in the probability of predicting the target token
	between the clean and corrupted model based on the final residuals. 
	This probability is returned as a percentage of the clean model's probability.

	Args:
		clean_resid (torch.Tensor): 
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
	clean_resid = model.ln_final(clean_resid)
	corrupted_resid = model.ln_final(corrupted_resid)
	clean_logits = model.unembed(clean_resid)[:, -1, :]
	corrupted_logits = model.unembed(corrupted_resid)[:, -1, :]

	# Get the probability of the target token
	prob_clean = torch.Tensor([clean_logits[i].softmax(dim=-1)[target_tokens[i]] for i in range(len(target_tokens))])
	prob_corrupted = torch.Tensor([corrupted_logits[i].softmax(dim=-1)[target_tokens[i]] for i in range(len(target_tokens))])

	return torch.mean(100*(prob_clean - prob_corrupted)/prob_clean)

def compare_token_logit(clean_resid: Tensor,
						corrupted_resid: Tensor,
						model: HookedTransformer,
						target_tokens: List[int]) -> Tensor:
	"""
	Compute the difference in logits for the target token as a percentage
	between the clean and corrupted model based on the final residuals.
	This implementation is optimized for transformerlens HookedTransformer.

	Args:
		clean_resid (torch.Tensor): 
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
	clean_final_resid = clean_resid[:, -1, :]
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

def indirect_effect(clean_resid: Tensor,
					corrupted_resid: Tensor,
					model: HookedTransformer,
					clean_targets: List[int],
					corrupt_targets: List[int],
					verbose = False,
					set_baseline = False) -> Tensor:
	"""
	Compute the Indirect Effect (IE) score.
	IE(z) = 0.5 * [ (P*z(r) - P(r)) / P(r) + (P(r') - P*z(r')) / P*z(r') ]
	This measures how much a component's activation (z) from a corrupted run
	influences the output probabilities on a clean run.

	Args:
		clean_resid (torch.Tensor): 
			The final residual stream of the clean model run (prompt p2).
			Shape: (batch, seq_len, d_model).
		corrupted_resid (torch.Tensor): 
			The final residual stream of the corrupted model run (prompt p2 with intervention z from p1).
			Shape: (batch, seq_len, d_model).
		model (HookedTransformer): The hooked transformer model.
		clean_targets (List[int]): 
			The indexes of the target tokens for the clean prompt (r').
		corrupt_targets (List[int]): 
			The indexes of the target tokens from the corrupted prompt (r).
		verbose (bool, optional): If True, prints intermediate values for debugging. Default is False.
		set_baseline (bool, optional): If True, sets the baseline score for centering the metric. Default is False.

	Returns:
		torch.Tensor: The Indirect Effect score.
	"""

	# Get the final residual stream for the last token
	clean_final_resid = clean_resid[:, -1, :]
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

	batch_indices = torch.arange(len(clean_targets))

	# P(r'): Probability of the clean target (r') on a clean run.
	P_r_prime = clean_probs[batch_indices, clean_targets]
	# P(r): Probability of the corrupt target (r) on a clean run.
	P_r = clean_probs[batch_indices, corrupt_targets]

	# P*z(r'): Probability of the clean target (r') on a corrupted run.
	P_z_star_r_prime = corrupted_probs[batch_indices, clean_targets]
	# P*z(r): Probability of the corrupt target (r) on a corrupted run.
	P_z_star_r = corrupted_probs[batch_indices, corrupt_targets]

	# Term 1: (P*z(r) - P(r)) / P(r)
	# Relative increase in probability for the new answer (r)
	term1 = (P_z_star_r - P_r) / (P_r + 1e-8)

	# Term 2: (P(r') - P*z(r')) / P*z(r')
	# Change in probability for the original answer (r')
	term2 = (P_r_prime - P_z_star_r_prime) / (P_z_star_r_prime + 1e-8)

	indirect_effects = 0.5 * (term1 + term2)

	if verbose:
		print(f"P(r): {P_r.mean().item()}, P*z(r): {P_z_star_r.mean().item()}")
		print(f"P(r'): {P_r_prime.mean().item()}, P*z(r'): {P_z_star_r_prime.mean().item()}")
		print(f"Indirect effect: {indirect_effects.mean().item()}")

	return torch.mean(indirect_effects)

def logit_difference_counterfactual(clean_resid: Tensor,
                                   counterfactual_resid: Tensor, 
                                   model: HookedTransformer,
                                   target_tokens: List[int],
                                   counterfactual_tokens: Optional[List[int]] = None,
                                   use_ablation_mode: bool = False) -> Tensor:
	"""
	Compute logit difference: y' - y between the correct answer y (clean) and y' (counterfactual).
	This function supports both explicit counterfactual comparison and ablation-based comparison.

	Args:
		clean_resid (torch.Tensor): 
			The final residual stream of the clean model.
			Shape: (batch, seq_len, d_model).
		counterfactual_resid (torch.Tensor): 
			The final residual stream of the counterfactual model (or ablated model).
			Shape: (batch, seq_len, d_model).
		model (HookedTransformer): 
			The hooked transformer model.
		target_tokens (List[int]): 
			The indexes of the target tokens for the clean model.
		counterfactual_tokens (Optional[List[int]]): 
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
	clean_final_resid = clean_resid[:, -1, :]
	counterfactual_final_resid = counterfactual_resid[:, -1, :]
	
	# Apply the layer norm to the final residuals
	clean_final_resid = model.ln_final(clean_final_resid)
	counterfactual_final_resid = model.ln_final(counterfactual_final_resid)
	
	if use_ablation_mode:
		# Ablation mode: counterfactual_resid is the ablated version
		# Use same target tokens for both clean and ablated
		clean_logits = torch.einsum('b d, d b-> b', clean_final_resid, W_U[:, target_tokens]) + b_U[target_tokens]
		counterfactual_logits = torch.einsum('b d, d b-> b', counterfactual_final_resid, W_U[:, target_tokens]) + b_U[target_tokens]
	else:
		# Explicit counterfactual mode: need separate target tokens
		if counterfactual_tokens is None:
			raise ValueError("counterfactual_tokens must be provided when use_ablation_mode=False")
		
		clean_logits = torch.einsum('b d, d b-> b', clean_final_resid, W_U[:, target_tokens]) + b_U[target_tokens]
		counterfactual_logits = torch.einsum('b d, d b-> b', counterfactual_final_resid, W_U[:, counterfactual_tokens]) + b_U[counterfactual_tokens]
	
	# Calculate the logit difference: y' - y   ## y - y*
	logit_diffs = counterfactual_logits - clean_logits
	return torch.mean(logit_diffs).item()
