import time
import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
from ipe.experiment import ExperimentManager
import os
import gc
import multiprocessing


def run_experiment(MODEL, TASK, METRIC, ALGORITHM, SEARCH_STRATEGY, ALGORITHM_PARAMS, CF, POSITIONAL, TOPN=200):
    # This function needs to be self-contained in terms of device selection
    # as it will run in a separate process.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1 # Or pass this as an argument if it varies

    model = HookedTransformer.from_pretrained(MODEL,
                                            device=DEVICE,
                                            torch_dtype=torch.float32,
    )
    model.eval()

    prompts = []
    answers = []
    counterfactual_prompts = []
    counterfactual_answers = []
    
    if TASK == "ioi":
        cf_key = 's2_io_flip_counterfactual'
        train_dataset = load_dataset(f'mib-bench/ioi', split='train')
        TARGET_LENGTH = 14
    if TASK == "mcqa":
        train_dataset = load_dataset(f'mib-bench/copycolors_mcqa', '4_answer_choices', split='train')
        cf_key = 'symbol_counterfactual'
        TARGET_LENGTH = 31
    for sample in train_dataset:
        if model.to_tokens(sample['prompt'], prepend_bos=True).shape[1] == TARGET_LENGTH:
            prompts.append(sample['prompt'])
            if TASK == "mcqa":
                answers.append(f' {sample["choices"]["label"][sample["answerKey"]]}')
            else:
                answers.append(f' {sample["metadata"]["indirect_object"]}')

            counterfactual_prompts.append(sample[cf_key]['prompt'])
            if TASK == "mcqa":
                counterfactual_answers.append(f'{sample[cf_key]["choices"]["label"][sample[cf_key]["answerKey"]]}')
            else:
                counterfactual_answers.append(f'{sample[cf_key]['choices'][sample[cf_key]['answerKey']]}')
            if len(prompts) >= BATCH_SIZE:
                break
    print(prompts[0])
    print(answers[0])
    print(counterfactual_prompts[0])
    print(counterfactual_answers[0])

    experiment = ExperimentManager(
        model=model,
        prompts=prompts, 
        targets=answers,
        cf_prompts=counterfactual_prompts,
        cf_targets=counterfactual_answers,
        algorithm=ALGORITHM,
        search_strategy=SEARCH_STRATEGY,
        algorithm_params=ALGORITHM_PARAMS,
        metric=METRIC,
        positional_search=POSITIONAL,
        patch_clean_into_cf=CF,
        patch_type='counterfactual' if CF else 'zero'
    )
    paths = experiment.run()
    print(f"Found {len(paths)} paths.")
    experiment.paths = sorted(paths, key=lambda x: abs(x[0].item()), reverse=True)[:TOPN]
    if not os.path.exists("./saved_paths"):
        os.makedirs("./saved_paths")
    experiment.save_paths(f"./saved_paths/paths_{MODEL.lower().replace("/", "-")}_{TASK}_{ALGORITHM}_{SEARCH_STRATEGY}_{METRIC}_cf{CF}_pos{POSITIONAL}.pkl")
    print("Saved paths.")

if __name__ == "__main__":
    # It's important to set the start method for multiprocessing with CUDA
    multiprocessing.set_start_method('spawn', force=True)

    METRICS = ["target_probability_percentage", "logit_difference", "kl_divergence", "indirect_effect", "target_logit_percentage"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1
    MODELS = ["gpt2-small", "Qwen/Qwen2.5-0.5B"]
    TASK = ["mcqa", "ioi"]
    ALGORITHM = "PathAttributionPatching" #"PathAttributionPatching" # PathMessagePatching
    SEARCH_STRATEGY = "BestFirstSearch" #"BestFirstSearch"Threshold" #"BestFirstSearch"
    ALGORITHM_PARAMS = {"top_n": 300, "max_time": 1000}  #{"min_contribution": 1.}#, "batch_positions": True} #{"top_n": 300, "max_time": 3600} # {"top_n": 10, "max_time": 8*3600}, {"max_width": 10}
    CF = [True, False]
    experiment_number = len(MODELS) * len(TASK) * len(METRICS) * len(CF)
    
    for modelname in MODELS:
        for task in TASK:
            for metric in METRICS:
                for cf in CF:
                    # The subprocess will handle memory cleanup, but these don't hurt.
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    if torch.cuda.is_available():
                        allocated_memory = torch.cuda.memory_allocated(DEVICE) / (1024 ** 3)
                        reserved_memory = torch.cuda.memory_reserved(DEVICE) / (1024 ** 3)
                        print(f"Current GPU memory usage: Allocated: {allocated_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB")
                    
                    print(f"Running experiment for Model: {modelname}, Task: {task}, Metric: {metric}, Counterfactual: {cf}")
                    
                    # Create and start the subprocess
                    p = multiprocessing.Process(
                        target=run_experiment,
                        args=(
                            modelname,
                            task,
                            metric,
                            ALGORITHM,
                            SEARCH_STRATEGY,
                            ALGORITHM_PARAMS,
                            cf,
                            True,  # POSITIONAL
                            200    # TOPN
                        )
                    )
                    p.start()
                    p.join() # Wait for the subprocess to finish

                    if p.exitcode != 0:
                        print(f"Subprocess for experiment failed with exit code {p.exitcode}. Stopping.")
                        exit(1)

