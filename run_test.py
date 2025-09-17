import torch
import transformer_lens
from backward_search_approximated.utils.nodes import ATTN_ApproxNode, EMBED_ApproxNode, FINAL_ApproxNode, MLP_ApproxNode
from backward_search_approximated.utils.graph_search import PathAttributionPatching
from backward_search_approximated.utils.metrics import compare_token_logit, indirect_effect
from functools import partial
from copy import deepcopy
import transformers
import huggingface_hub
from datasets import load_dataset
import dotenv
from transformer_lens import HookedTransformer
import os

transformers.logging.set_verbosity_error()
# torch.set_default_dtype(torch.bfloat16)

dotenv.load_dotenv()

TOKEN = os.getenv("TOKEN")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)
TASK = "ioi" # "ioi" or "mcqa"
TARGET_LENGTH = 15 # from 15 to 19 for ioi - from 32 to 37 for mcqa
BATCH_SIZE = 4 # Number of samples from the dataset to consider

huggingface_hub.login(token=TOKEN)
model = HookedTransformer.from_pretrained('Qwen/Qwen2.5-0.5B',# 'gpt2-small', # 
                                          device=DEVICE, 
                                          torch_dtype=torch.float32, 
                                          center_unembed=True,
                                          )
model.eval()



samples = []
samples_prompt = []
sample_answers = []

samples_counterfactual_prompt = []
sample_counterfactual_answers = []

train_dataset = load_dataset('mib-bench/ioi', split='train')
validation_dataset = load_dataset('mib-bench/ioi', split='validation')
test_dataset = load_dataset('mib-bench/ioi', split='test')
for sample in train_dataset:
	if model.to_tokens(sample['prompt'], prepend_bos=True).shape[1] == TARGET_LENGTH:
		samples.append(sample)
		samples_prompt.append(sample['prompt'])
		sample_answers.append(model.to_tokens(f' {sample['metadata']['indirect_object']}', prepend_bos=False).item())

		samples_counterfactual_prompt.append(sample['s2_io_flip_counterfactual']['prompt'])
		sample_counterfactual_answers.append(model.to_tokens(f' {sample['s2_io_flip_counterfactual']['choices'][sample['s2_io_flip_counterfactual']['answerKey']]}', prepend_bos=False).item())
		if len(samples) >= BATCH_SIZE:
			break

print(f"Loaded {len(samples)} samples for the task {TASK} with target length {TARGET_LENGTH}.")
print(f"Sample prompt: \n''{samples_prompt[0]}''")
print(f"Sample answer: ''{model.to_string(sample_answers[0])}''")
print(f"Probability of the answer: {torch.softmax(model(samples_prompt[0], prepend_bos=True, return_type='logits')[0, -1], dim=-1)[sample_answers[0]].item()} ~ Logit: {model(samples_prompt[0], prepend_bos=True, return_type='logits')[0, -1, sample_answers[0]].item()}")
print(f"\nSample counterfactual prompt: \n''{samples_counterfactual_prompt[0]}''")
print(f"Sample counterfactual answer: ''{model.to_string(sample_counterfactual_answers[0])}''")
print(f"Probability of the counterfactual answer: {torch.softmax(model(samples_counterfactual_prompt[0], prepend_bos=True, return_type='logits')[0, -1], dim=-1)[sample_counterfactual_answers[0]].item()} ~ Logit: {model(samples_counterfactual_prompt[0], prepend_bos=True, return_type='logits')[0, -1, sample_counterfactual_answers[0]].item()}")

logits, cache = model.run_with_cache(samples_prompt, prepend_bos=True)

cf_logits, cf_cache = model.run_with_cache(samples_counterfactual_prompt, prepend_bos=True)

correct_token_ids = [model.to_single_token(sample['metadata']['indirect_object']) for sample in samples]
cf_token_ids = [model.to_single_token(sample['s2_io_flip_counterfactual']['choices'][sample['s2_io_flip_counterfactual']['answerKey']]) for sample in samples]




msg_cache = dict(cache)
counterfactual_cache = dict(cf_cache)
# for key in msg_cache.keys():
# 	counterfactual_cache[key] = torch.zeros_like(msg_cache[key], device=msg_cache[key].device)
#metric = partial(compare_token_logit, clean_resid=cache[f'blocks.{model.cfg.n_layers-1}.hook_resid_post'], model=model, target_tokens=correct_token_ids)
#print(indirect_effect(clean_resid=cache[f'blocks.{model.cfg.n_layers-1}.hook_resid_post'], corrupted_resid=cache[f'blocks.{model.cfg.n_layers-1}.hook_resid_post'], model=model, clean_targets=correct_token_ids, corrupt_targets=cf_token_ids, verbose=True, set_baseline=True).item())
metric = partial(indirect_effect, clean_resid=cache[f'blocks.{model.cfg.n_layers-1}.hook_resid_post'], model=model, clean_targets=correct_token_ids, corrupt_targets=cf_token_ids, verbose=False, set_baseline=False)

paths = PathAttributionPatching(
	model=model,
	msg_cache=msg_cache,
	metric=metric,
	root= FINAL_ApproxNode(
		model=model,
		layer=model.cfg.n_layers - 1,
		metric=metric,
		position=None,
		parent=None,
		msg_cache=msg_cache,
		cf_cache=counterfactual_cache,
		patch_type='counterfactual',
		),
	ground_truth_tokens=correct_token_ids,
	min_contribution = 0.0035,
	include_negative=True,
	return_all=True,
)


# save circuit
import json
from datetime import datetime

# Convert the complete_paths to a serializable format
def convert_path_to_dict(path_tuple):
    score, path = path_tuple
    path_dict = {
        "score": float(score),
        "nodes": []
    }
    
    for node in path:
        node_dict = {
            "type": node.__class__.__name__,
            "layer": node.layer,
            "position": node.position
        }
        
        # Add attention-specific attributes
        if hasattr(node, 'head'):
            node_dict["head"] = node.head
        if hasattr(node, 'keyvalue_position'):
            node_dict["keyvalue_position"] = node.keyvalue_position
        if hasattr(node, 'patch_query'):
            node_dict["patch_query"] = node.patch_query
        if hasattr(node, 'patch_keyvalue'):
            node_dict["patch_keyvalue"] = node.patch_keyvalue
            
        path_dict["nodes"].append(node_dict)
    
    return path_dict

# Convert all paths
serializable_paths = [convert_path_to_dict(path) for path in paths]
# Create metadata
metadata = {
    "model": model.cfg.model_name,
    "prompt": samples_prompt[0],
    "cf_prompt": samples_counterfactual_prompt[0],
    "correct_answer": str(model.to_string(correct_token_ids[0])),
    "target_idx": correct_token_ids[0],
    "timestamp": datetime.now().isoformat(),
    "total_paths": len(paths),
    "min_treshold": 10,
    "n_layers": model.cfg.n_layers,
    "d_model": model.cfg.d_model,
    "n_heads": model.cfg.n_heads,
    "metric": "compare_token_logit",
}

# Combine data
output_data = {
    "metadata": metadata,
    "paths": serializable_paths
}

# Save to JSON file
filename = f"test_paths.json"

with open(filename, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Saved {len(paths)} paths to {filename}")
print(f"Top 3 paths by score:")
for i, path in enumerate(serializable_paths[:3]):
    print(f"  {i+1}. Score: {path['score']:.4f}, Nodes: {len(path['nodes'])}")
