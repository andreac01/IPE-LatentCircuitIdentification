import torch
import transformers
from transformer_lens import HookedTransformer
import huggingface_hub
from datasets import load_dataset
import dotenv
import os
import pickle as pkl
from backward_search.utils.metrics import compare_token_probability, kl_divergence, compare_token_logit, indirect_effect, logit_difference_counterfactual
from backward_search.utils.nodes import MLP_Node, EMBED_Node, FINAL_Node, Node, ATTN_Node
from backward_search.utils.graph_search import path_message, evaluate_path, breadth_first_search, breadth_first_search_cached, breadth_first_search_cached_no_pos, breadth_first_search_recursive, breadth_first_search_with_counterfactual_cached_no_pos
from datetime import datetime

transformers.logging.set_verbosity_error()
# torch.set_default_dtype(torch.bfloat16)

dotenv.load_dotenv()

TOKEN = os.getenv("TOKEN")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)
TASK = "ioi" # "ioi" or "mcqa"
TARGET_LENGTH = 15 # from 15 to 19 for ioi - from 32 to 37 for mcqa
BATCH_SIZE = 8 # Number of samples from the dataset to consider
DEFAULT_METRIC = indirect_effect # Note: if logit_difference_counterfactual => take_message_from_clean = False | if indirect_effect => take_message_from_clean = True
CONTRIBUTION_THRESHOLD = 0.00025 # close to 0.5 for logit_difference_counterfactual, close to 0.0001 for indirect_effect

huggingface_hub.login(token=TOKEN)
# Note: Eventually can set set fold_ln=False, center_unembed=False, center_writing_weights=False
model = HookedTransformer.from_pretrained('gpt2-small', 
                                          device=DEVICE, 
                                          torch_dtype=torch.float32, 
                                          center_unembed=True,
                                          )



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

print(f"Baseline indirect effect: \n{indirect_effect(cache['ln_final.hook_normalized'], cf_cache['ln_final.hook_normalized'], model, sample_answers, sample_counterfactual_answers, verbose=True, set_baseline=True)}")

complete_paths =  breadth_first_search_with_counterfactual_cached_no_pos(
	model = model,
	cache = cache,
	counterfactual_cache = cf_cache,
	metric = DEFAULT_METRIC,
	start_node = [FINAL_Node(model.cfg.n_layers - 1, None)],
	ground_truth_tokens = sample_answers,
	counterfactual_tokens = sample_counterfactual_answers,
	max_depth = 10,
	max_branching_factor = 2048,
	min_contribution = CONTRIBUTION_THRESHOLD,
	min_contribution_percentage = 0.0,
	inibition_task = False,
	take_message_from_clean = False,
	cached_path_length = 1,
	absolute = True
)

incomplete_paths = []
print(complete_paths)
print(f"Found {len(complete_paths)} complete paths and {len(incomplete_paths)} incomplete paths.")

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
serializable_paths = [convert_path_to_dict(path) for path in complete_paths]
serializable_paths = serializable_paths + [convert_path_to_dict(path) for path in incomplete_paths]
# Create metadata
metadata = {
    "model": "gpt2-small",
    "prompt": samples_prompt[0],
    "correct_answer": str(model.to_string(sample_answers[0])),
    "target_idx": sample_answers[0],
    "find_subject_inhibition": False,
    "timestamp": datetime.now().isoformat(),
    "total_paths": len(complete_paths + incomplete_paths),
    "min_treshold": CONTRIBUTION_THRESHOLD,
    "n_layers": model.cfg.n_layers,
    "d_model": model.cfg.d_model,
    "n_heads": model.cfg.n_heads,
    "metric": DEFAULT_METRIC.__name__
}

# Combine data
output_data = {
    "metadata": metadata,
    "paths": serializable_paths
}

# Save to JSON file
filename = f"detected_paths/detected_circuit_gpt2_{TASK}_absolute_{DEFAULT_METRIC.__name__}_{CONTRIBUTION_THRESHOLD}_bs{BATCH_SIZE}_l{TARGET_LENGTH}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(filename, 'w') as f:
    json.dump(output_data, f, indent=2)


# Save to pickle file
filename = f"detected_circuit_gpt2_{TASK}_absolute_{DEFAULT_METRIC.__name__}_{CONTRIBUTION_THRESHOLD}_bs{BATCH_SIZE}_l{TARGET_LENGTH}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
with open(filename, 'wb') as f:
    pkl.dump((complete_paths, incomplete_paths), f)
print(f"Saved {len(complete_paths + incomplete_paths)} paths to {filename}")
print(f"Top 3 paths by score:")
for i, path in enumerate(serializable_paths[:3]):
    print(f"  {i+1}. Score: {path['score']:.4f}, Nodes: {len(path['nodes'])}")
    
