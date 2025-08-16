import torch
from transformer_lens import HookedTransformer
from typing import List, Optional
from torch import Tensor
import torch.nn.functional as F

baseline_score = 0.0

def compare_token_probability(clean_resid: Tensor,
								corrupted_resid: Tensor,
								model: HookedTransformer,
								target_tokens: List[int]) -> Tensor:
	""" Compute the difference of predicting the target token in probability
		between the clean and corrupted model based on the final residuals.
		args:
			clean_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the clean model.
			corrupted_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the corrupted model.
			model: HookedTransformer
				The hooked transformer model.
			target_tokens: list of int
				The indexes of the target tokens.
		returns:
			float
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
	""" Compute the difference of logits for the target token as a percentage
		between the clean and corrupted model based on the final residuals.
		This implementation is optimized for transformerlens HookedTransformer.
		args:
			clean_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the clean model.
			corrupted_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the corrupted model.
			model: HookedTransformer
				The hooked transformer model.
			target_tokens: list of int
				The indexes of the target tokens.
		returns:
			float
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
					use_ablation_mode = True,
    				set_baseline = False) -> Tensor:
	""" Compute the difference of logits for the target token as a percentage
		between the clean and corrupted model based on the final residuals.
		This implementation is optimized for transformerlens HookedTransformer.
		args:
			clean_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the clean model.
			corrupted_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the corrupted model.
			model: HookedTransformer
				The hooked transformer model.
			target_tokens: list of int
				The indexes of the target tokens.
		returns:
			float
				The percentage difference in logits for the target token.
	"""

	# Get the final residual stream for the last token
	clean_final_resid = clean_resid[:, -1, :]
	corrupted_final_resid = corrupted_resid[:, -1, :]
	
	# Apply the layer norm to the final residuals
	clean_final_resid = model.ln_final(clean_final_resid)
	corrupted_final_resid = model.ln_final(corrupted_final_resid)
	
	# Get the logits associated with the clean target in clean run and corrupted counterfactual run
	P_r = torch.tensor([torch.softmax(model.unembed(clean_final_resid[i]), dim=-1)[clean_targets[i]] for i in range(len(clean_targets))])
	P_r_star = torch.tensor([torch.softmax(model.unembed(corrupted_final_resid[i]), dim=-1)[clean_targets[i]] for i in range(len(clean_targets))])
	
	# Get the logits associated with the corrupted target in clean run and corrupted counterfactual run
	P_r_prime = torch.tensor([torch.softmax(model.unembed(clean_final_resid[i]), dim=-1)[corrupt_targets[i]] for i in range(len(corrupt_targets))])
	P_r_star_prime = torch.tensor([torch.softmax(model.unembed(corrupted_final_resid[i]), dim=-1)[corrupt_targets[i]] for i in range(len(corrupt_targets))])

	indirect_effects = 0.5 * ( (P_r_star - P_r)/(P_r + 1e-8) + (P_r_prime - P_r_star_prime)/(P_r_star_prime + 1e-8) )
	if verbose:
		print(f"P_r: {P_r.mean().item()}, P_r_star: {P_r_star.mean().item()}")
		print(f"P_r_prime: {P_r_prime.mean().item()}, P_r_star_prime: {P_r_star_prime.mean().item()}")
		print(f"Indirect effect: {indirect_effects.mean().item()}")

	if set_baseline: # Workaround for the fact that the score may be negative and is easier to have the metric centered
		global baseline_score
		baseline_score = indirect_effects.mean()
	return torch.mean(indirect_effects) - baseline_score

def logit_difference_counterfactual(clean_resid: Tensor,
                                   counterfactual_resid: Tensor, 
                                   model: HookedTransformer,
                                   target_tokens: List[int],
                                   counterfactual_tokens: Optional[List[int]] = None,
                                   use_ablation_mode: bool = False) -> Tensor:
	""" Compute logit difference: y' - y between correct answer y (clean) and y' (counterfactual)
		This function can handle both explicit counterfactual comparison and ablation-based comparison.
		
		args:
			clean_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the clean model.
			counterfactual_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the counterfactual model (or ablated model).
			model: HookedTransformer
				The hooked transformer model.
			target_tokens: list of int
				The indexes of the target tokens for the clean model.
			counterfactual_tokens: Optional[list of int]
				The indexes of the target tokens for the counterfactual model.
				Required when use_ablation_mode=False.
			use_ablation_mode: bool
				If True, uses ablation-based comparison (counterfactual_resid is ablated version).
				If False, uses explicit counterfactual comparison.
		returns:
			float
				The logit difference: y' - y
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

# TODO: check if this is correct
def kl_divergence(clean_resid: Tensor,
					corrupted_resid: Tensor,
					model: HookedTransformer,
					target_token: int) -> Tensor:
	""" Compute the Kullback-Leibler divergence between the probability distributions
		of the clean and corrupted model based on the final residuals.
		args:
			clean_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the clean model.
			corrupted_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the corrupted model.
			model: HookedTransformer
				The hooked transformer model.
		returns:
			float
				The Kullback-Leibler divergence.
	"""
	# Get logits for the last token	
	clean_resid = model.ln_final(clean_resid)
	corrupted_resid = model.ln_final(corrupted_resid)
	clean_logits = model.unembed(clean_resid)[:, -1, :]
	corrupted_logits = model.unembed(corrupted_resid)[:, -1, :]

	# Convert logits to probability distributions using softmax
	prob_clean = F.softmax(clean_logits, dim=-1)
	prob_corrupted = F.softmax(corrupted_logits, dim=-1)

	# Compute KL divergence
	kl_div = F.kl_div(torch.log(prob_corrupted), prob_clean, reduction='batchmean')
	return kl_div.item()