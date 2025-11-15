import time
import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
from ipe.experiment import ExperimentManager
import os
import time

if __name__ == "__main__":
    start_time = time.time()
    METRIC = "indirect_effect" # target_logit_percentage, target_probability_percentage, logit_difference, kl_divergence, indirect_effect
    METRIC_PARAMS = {}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_LENGTH = 30 # for ioi minimum is 14, normal is 15, up to 19 if multiple words are split into multiple tokens
    BATCH_SIZE = 4
    MODEL = "Qwen/Qwen2.5-0.5B" #"gpt2-small" # "Qwen/Qwen2.5-0.5B" # "Qwen/Qwen2.5-0.5B", "gpt2-small"
    TASK = "mcqa" # mcqa
    ALGORITHM = "PathAttributionPatching" # PathMessagePatching, PathAttributionPatching
    SEARCH_STRATEGY = "Threshold" #"Threshold" # BestFirstSearch, LimitedLevelWidth
    CF = True
    DENOISING = True
    POSITIONAL = False
    ALGORITHM_PARAMS = {"min_contribution": 0.01, "confirm_relevance": True} # "batch_heads": True} # {"top_n": 10, "max_time": 8*3600}, {"max_width": 10}
    model = HookedTransformer.from_pretrained(MODEL,
                                            device=DEVICE, 
                                            torch_dtype=torch.float32,
                                            center_unembed=True,
                                            
    )
    model.eval()

    prompts = []
    answers = []
    counterfactual_prompts = []
    counterfactual_answers = []
    
    if TASK == "ioi":
        train_dataset = load_dataset(f'mib-bench/{TASK}', split='train')
        cf_key = 's2_io_flip_counterfactual'
    if TASK == "mcqa":
        train_dataset = load_dataset(f'mib-bench/copycolors_{TASK}', '4_answer_choices', split='train')
        cf_key = 'symbol_counterfactual'
    for sample in train_dataset:
        if model.to_tokens(sample['prompt'], prepend_bos=True).shape[1] == TARGET_LENGTH:
            prompts.append(sample['prompt'])
            counterfactual_prompts.append(sample[cf_key]['prompt'])
            if TASK == "ioi":
                answers.append(f' {sample['metadata']['indirect_object']}')
                counterfactual_answers.append(f' {sample[cf_key]['choices'][sample[cf_key]['answerKey']]}')
            if TASK == "mcqa":
                prompts[-1] += " "
                counterfactual_prompts[-1] += " "
                answers.append(f'{sample['choices']['label'][sample['answerKey']]}')
                counterfactual_answers.append(f'{sample[cf_key]['choices']['label'][sample[cf_key]['answerKey']]}')
            if len(prompts) >= BATCH_SIZE:
                break
    
    # Note:
    # On the IOI task the clean and counterfactual prompts complete the same task but with different names
    # Given that we start from a prompt with all counterfactuals we can actually invert clean and counterfactuals
    # This does not hold in the case of MCQA where the counterfactuals change the question itself
    if TASK == "ioi":
        experiment = ExperimentManager(
            model=model,
            prompts=counterfactual_prompts if CF else prompts, 
            targets=counterfactual_answers if CF else answers,
            cf_prompts=prompts if CF else [],
            cf_targets=answers if CF else [],
            algorithm=ALGORITHM,
            search_strategy=SEARCH_STRATEGY,
            algorithm_params=ALGORITHM_PARAMS,
            metric=METRIC,
            metric_params=METRIC_PARAMS,
            positional_search=POSITIONAL,
            patch_type="counterfactual" if CF else "zero",
            patch_clean_into_cf=DENOISING
        )
    if TASK == "mcqa":
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
            patch_type="counterfactual" if CF else "zero",
            patch_clean_into_cf=DENOISING
        )
    paths = experiment.run()
    if not os.path.exists("./saved_paths"):
        os.makedirs("./saved_paths")
    experiment.save_paths(filepath=f"./detected_paths/paths_{MODEL.lower().replace('/','-')}_{TASK}_{ALGORITHM}_{SEARCH_STRATEGY}_{METRIC}_cf{CF}_pos{POSITIONAL}-{time.strftime('%Y%m%d_%H%M%S')}.pkl")
    print("Saved paths.")
    print(f"Total time: {time.time() - start_time:.2f} seconds")