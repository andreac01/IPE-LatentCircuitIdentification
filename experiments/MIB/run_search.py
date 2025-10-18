import time
import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
from ipe.experiment import ExperimentManager
import os

if __name__ == "__main__":

    METRIC = "logit_difference" # target_logit_percentage, target_probability_percentage, logit_difference, kl_divergence, indirect_effect
    METRIC_PARAMS = {}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_LENGTH = 15 # for ioi minimum is 14, normal is 15, up to 19 if multiple words are split into multiple tokens
    BATCH_SIZE = 8
    MODEL = "Qwen/Qwen2.5-0.5B" # "Qwen/Qwen2.5-0.5B", "gpt2-small"
    TASK = "ioi" # mcqa
    ALGORITHM = "PathAttributionPatching" # PathMessagePatching, PathAttributionPatching
    SEARCH_STRATEGY = "Threshold" # BestFirstSearch, MaxWidth
    CF = True
    DENOISING = True
    POSITIONAL = True
    ALGORITHM_PARAMS = {"min_contribution": 0.0025}#, "batch_heads": True} # {"top_n": 10, "max_time": 8*3600}, {"max_width": 10}
    model = HookedTransformer.from_pretrained(MODEL,
                                            device=DEVICE, 
                                            torch_dtype=torch.float32, 
    )
    model.eval()

    prompts = []
    answers = []
    counterfactual_prompts = []
    counterfactual_answers = []
    train_dataset = load_dataset(f'mib-bench/{TASK}', split='train')
    
    if TASK == "ioi":
        cf_key = 's2_io_flip_counterfactual'
    if TASK == "mcqa":
        cf_key = 'symbol_counterfactual'
    for sample in train_dataset:
        if model.to_tokens(sample['prompt'], prepend_bos=True).shape[1] == TARGET_LENGTH:
            prompts.append(sample['prompt'])
            answers.append(f' {sample['metadata']['indirect_object']}')

            counterfactual_prompts.append(sample[cf_key]['prompt'])
            counterfactual_answers.append(f' {sample[cf_key]['choices'][sample[cf_key]['answerKey']]}')
            if len(prompts) >= BATCH_SIZE:
                break

    experiment = ExperimentManager(
        model=model,
        prompts=prompts, 
        targets=answers,
        cf_prompts=counterfactual_prompts if CF else [],
        cf_targets=counterfactual_answers if CF else [],
        algorithm=ALGORITHM,
        search_strategy=SEARCH_STRATEGY,
        algorithm_params=ALGORITHM_PARAMS,
        metric=METRIC,
        metric_params=METRIC_PARAMS,
        positional_search=POSITIONAL,
        patch_type="auto",
        patch_clean_into_cf=DENOISING
    )

    paths = experiment.run()
    if not os.path.exists("./saved_paths"):
        os.makedirs("./saved_paths")
    experiment.save_paths(filepath=f"./saved_paths/paths_{MODEL.lower().replace('/','-')}_{TASK}_{ALGORITHM}_{SEARCH_STRATEGY}_{METRIC}_cf{CF}_pos{POSITIONAL}-{time.strftime('%Y%m%d_%H%M%S')}.pkl")
    print("Saved paths.")