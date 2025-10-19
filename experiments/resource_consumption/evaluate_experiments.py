# This script requires to install nvidia-ml-py package.
import pynvml
import time
import threading
import torch
import pandas as pd
import gc
import os
from transformer_lens import HookedTransformer
from datasets import load_dataset
from ipe.experiment import ExperimentManager
import argparse
import json
import sys
import subprocess
import importlib
import transformer_lens

# Initialize NVML
try:
    pynvml.nvmlInit()
except pynvml.NVMLError as error:
    print(f"Failed to initialize NVML: {error}. Make sure NVIDIA drivers are installed.")
    # In the controller, we might exit. In a worker, an error is fine.
    if 'run_worker' not in sys.argv:
        exit()

GPU_ID = 0
handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_ID)

def bytes_to_mb(b):
    """Utility function to convert bytes to megabytes."""
    return round(b / (1024**2), 2)

def wait_for_gpu_low_usage(util_threshold=10, mem_threshold=10):
    """Waits until GPU utilization and memory usage are below specified thresholds."""
    print(f"\nWaiting for GPU usage to be low (Util < {util_threshold}%, Memory < {mem_threshold}%)...")
    while True:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_percent = (mem_info.used / mem_info.total) * 100

            if utilization < util_threshold and mem_used_percent < mem_threshold:
                print("✅ GPU is idle. Starting experiment.")
                break
            else:
                print(f"GPU is busy. Current Util: {utilization}%, Memory: {mem_used_percent:.2f}%. Waiting...")
                time.sleep(5)
        except pynvml.NVMLError as error:
            print(f"NVML Error: {error}. Retrying...")
            time.sleep(5)

def profile_gpu(func):
    """
    A decorator that profiles the execution time and the GPU memory usage of a
    specific process. This is robust for long-running jobs on shared GPUs.
    """
    def wrapper(*args, **kwargs):
        script_pid = os.getpid()
        class MemoryMonitor(threading.Thread):
            def __init__(self):
                super().__init__()
                self.daemon = True
                self.peak_memory = 0
                self.running = True

            def run(self):
                while self.running:
                    try:
                        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        for p in procs:
                            if p.pid == script_pid:
                                self.peak_memory = max(self.peak_memory, p.usedGpuMemory)
                                break
                    except pynvml.NVMLError:
                        pass
                    time.sleep(0.5)

            def stop(self):
                self.running = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        monitor = MemoryMonitor()
        monitor.start()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        monitor.stop()
        monitor.join()

        peak_process_mem_mb = bytes_to_mb(monitor.peak_memory)
        metrics = {
            "execution_time_s": round(end_time - start_time, 4),
            "peak_process_memory_mb": peak_process_mem_mb
        }
        print(f"--- Function '{func.__name__}' Profile")
        print(f"  Execution Time: {metrics['execution_time_s']} s")
        print(f"  Peak Memory Used by this Process: {metrics['peak_process_memory_mb']} MB")
        return result, metrics
    return wrapper

@profile_gpu
def run(experiment):
    experiment.run()

@profile_gpu
def instantiate(experiment_class, *args, **kwargs):
    return experiment_class(*args, **kwargs)


# Worker Function for a Single Experiment Run
def run_single_experiment(worker_args):
    """
    This function encapsulates the logic for running exactly ONE experiment.
    It's designed to be called in a separate process with its own specific arguments.
    """
    print(f"\n--- [WORKER PID: {os.getpid()}] Running {worker_args.algorithm_type} with {worker_args.search_strategy} {worker_args.algorithm_params} (Positional: {worker_args.positional_search})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    prompts, answers = [], []
    cf_prompts, cf_answers = [], []
    train_dataset = load_dataset('mib-bench/ioi', split='train')
    temp_tokenizer = HookedTransformer.from_pretrained(worker_args.model, device='cpu')
    for sample in train_dataset:
        if temp_tokenizer.to_tokens(sample['prompt'], prepend_bos=True).shape[1] == worker_args.target_length:
            prompts.append(sample['prompt'])
            answers.append(f' {sample["metadata"]["indirect_object"]}')
            if len(prompts) >= worker_args.batch_size:
                break
    del temp_tokenizer
    time.sleep(1)  # Ensure system stability
    gc.collect()
    torch.cuda.empty_cache()

    wait_for_gpu_low_usage(util_threshold=2, mem_threshold=25)
    model = HookedTransformer.from_pretrained(
        worker_args.model,
        device=device,
        torch_dtype=torch.float32,
    )
    model.eval()

    tname = worker_args.algorithm_type.replace("Batched", "").replace("Heads", "").replace("Pos", "")
    
    # Deserialize the algorithm parameters from the JSON string
    algorithm_params_dict = json.loads(worker_args.algorithm_params)

    experiment, _ = instantiate(
        ExperimentManager, model, prompts, answers,
        algorithm=tname,
        search_strategy=worker_args.search_strategy,
        algorithm_params=algorithm_params_dict,
        metric=worker_args.metric,
        metric_params={},
        positional_search=worker_args.positional_search,
        patch_type="zero"
    )

    _, metrics_run = run(experiment)

    result = {
        "type": worker_args.algorithm_type,
        "search_strategy": worker_args.search_strategy,
        "top_n": algorithm_params_dict.get('top_n'),
        "threshold": algorithm_params_dict.get('min_contribution'),
        "max_width": algorithm_params_dict.get('max_width'),
        "paths_found": len(experiment.paths),
        "relevant_paths_found": len([p for c, p in experiment.paths if abs(c) > 0.5]),
        "execution_time_s": metrics_run['execution_time_s'],
        "peak_process_memory_mb": metrics_run['peak_process_memory_mb'],
    }

    columns = [
        "type",
        "search_strategy",
        "top_n",
        "threshold",
        "max_width",
        "paths_found",
        "relevant_paths_found",
        "execution_time_s",
        "peak_process_memory_mb"
    ]

    df = pd.DataFrame([result], columns=columns)
    if not os.path.exists(worker_args.target_file):
        df.to_csv(worker_args.target_file, index=False)
    else:
        df.to_csv(worker_args.target_file, mode='a', header=False, index=False)

    print(f"✅ [WORKER PID: {os.getpid()}] Results saved to {worker_args.target_file}")


# Main script logic
if __name__ == "__main__":
    # This helper function is used ONLY by the controller to parse the CLI arguments
    def _parse_list(arg, cast=int):
        if arg is None: return None
        if isinstance(arg, (list, tuple)): return [cast(x) for x in arg]
        if arg == "": return []
        return [cast(x) for x in str(arg).split(",")]

    parser = argparse.ArgumentParser(description="Evaluate experiments with configurable defaults")

    # User-Facing Arguments for the CONTROLLER
    parser.add_argument("--metric", type=str, default="target_logit_percentage")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--target-length", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--thresholds", type=str, default="1,0.5,0.25")
    parser.add_argument("--top-ns", type=str, default="10,100,1000")
    parser.add_argument("--max-widths", type=str, default="10,100,1000")
    parser.add_argument("--positional", type=str, default="True-False")

    # Internal Arguments for the WORKER
    # These are not meant for direct user interaction. The controller sets them.
    parser.add_argument("--run-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--target-file", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--algorithm-type", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--search-strategy", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--algorithm-params", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--positional-search", type=lambda x: (str(x).lower() == 'true'), help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Router: Decide whether to act as a worker or a controller
    if args.run_worker:
        # If --run-worker is passed, this process's ONLY job is to run one experiment.
        run_single_experiment(args)
    else:
        # This is the CONTROLLER process. It generates all experiment configurations
        # and spawns a worker for each one.

        # Parse all user-facing arguments using the helper
        METRIC = args.metric
        TARGET_LENGTH = args.target_length
        BATCH_SIZE = args.batch_size
        MODEL = args.model
        THRESHOLDS = _parse_list(args.thresholds, cast=float)
        TOP_NS = _parse_list(args.top_ns, cast=int)
        MAX_WIDTHS = _parse_list(args.max_widths, cast=int)
        positional_search_flags = [val.strip().lower() == "true" for val in args.positional.split("-")]

        if args.model not in ["gpt2", "qwen"]:
            MODELNAME = args.model
        else:
            MODELNAME = "gpt2-small" if MODEL == "gpt2" else "Qwen/Qwen2.5-0.5B" if MODEL == "qwen" else MODEL
        print(MODELNAME)
        # Generate all experiment configurations
        all_configs = []
        for positional in positional_search_flags:
            types = ['PathAttributionPatching', 'PathMessagePatching', 'PathMessagePatchingBatchedHeadsPos']
            for t in types:
                algorithms = []
                # for top_n in TOP_NS:
                #     params = {'top_n': top_n, 'max_time': 2*60*60}
                #     if "Pos" in t: params['batch_positions'] = True
                #     if "Head" in t: params['batch_heads'] = True
                #     algorithms.append({"method": 'BestFirstSearch', "params": params})
                for threshold in THRESHOLDS:
                    if t == 'PathMessagePatching':
                        continue
                    params = {'min_contribution': threshold, 'return_all':True}
                    if "Pos" in t: params['batch_positions'] = True
                    if "Head" in t: params['batch_heads'] = True
                    algorithms.append({"method": 'Threshold', "params": params})
                for max_width in MAX_WIDTHS:
                    params = {'max_width': max_width}
                    if "Pos" in t: params['batch_positions'] = True
                    if "Head" in t: params['batch_heads'] = True
                    algorithms.append({"method": 'LimitedLevelWidth', "params": params})

                target_dir = f"./results/{MODEL}"
                os.makedirs(target_dir, exist_ok=True)
                target_file = f"{target_dir}/ioi_{'positional' if positional else 'non_positional'}_bs{BATCH_SIZE}_len{TARGET_LENGTH}.csv"

                for alg_config in algorithms:
                    all_configs.append({
                        "target_file": target_file,
                        "algorithm_type": t,
                        "search_strategy": alg_config['method'],
                        "algorithm_params": json.dumps(alg_config['params']),
                        "positional_search": positional,
                    })

        # Loop through configurations and spawn worker processes
        for i, config in enumerate(all_configs):
            print(f"\n{'='*20} CONTROLLER: Starting Experiment {i+1}/{len(all_configs)} {'='*20}")
            command = [
                sys.executable, __file__,
                "--run-worker",
                "--metric", METRIC,
                "--target-length", str(TARGET_LENGTH),
                "--batch-size", str(BATCH_SIZE),
                "--model", MODELNAME,
                "--target-file", config["target_file"],
                "--algorithm-type", config["algorithm_type"],
                "--search-strategy", config["search_strategy"],
                "--algorithm-params", config["algorithm_params"],
                "--positional-search", str(config["positional_search"]),
            ]

            subprocess.run(command, check=True)
            print(f"--- CONTROLLER: Finished Experiment {i+1}/{len(all_configs)}. GPU memory is now guaranteed to be free.")
            time.sleep(2) # Brief pause for system stability

# Clean Up NVML in the main controller process
pynvml.nvmlShutdown()