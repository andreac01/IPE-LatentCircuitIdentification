from datasets import load_dataset
import argparse
from node_pruning import recursive_forward_discovery
from transformer_lens import HookedTransformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='ioi', help='Task name (default: ioi)')
    parser.add_argument('--target-length', type=int, default=15, help='Target token length (default: 15)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    args = parser.parse_args()

    prompts = []
    answers = []
    counterfactual_prompts = []
    counterfactual_answers = []

    train_dataset = load_dataset(f'mib-bench/ioi', split='train')
    model = HookedTransformer.from_pretrained('gpt2-small')

    for sample in train_dataset:
        # if a model is provided, enforce the token-length check using model.to_tokens
        if model is not None:
            if model.to_tokens(sample['prompt'], prepend_bos=True).shape[1] != args.target_length:
                continue
        # if no model is provided, skip length filtering

        prompts.append(sample['prompt'])
        answers.append(f' {sample["metadata"]["indirect_object"]}')

        counterfactual_prompts.append(sample["s2_io_flip_counterfactual"]['prompt'])
        counterfactual_answers.append(
            f' {sample["s2_io_flip_counterfactual"]["choices"][sample["s2_io_flip_counterfactual"]["answerKey"]]}'
        )
        if len(prompts) >= args.batch_size:
            break

    rfd = recursive_forward_discovery(model, prompts, answers, counterfactual_prompts, counterfactual_answers, absolute=False, base_th=0.000025, n_steps=15)
    rfd.discover()
    rfd.save_results_mib_json(f'pruning_results_ioi_tl{args.target_length}_bs{args.batch_size}.json')