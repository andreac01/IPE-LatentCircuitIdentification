from functools import partial
import transformer_lens
import pytest
from ipe.nodes import FINAL_Node
from ipe.graph_search import (
    PathAttributionPatching,
    PathAttributionPatching_BestFirstSearch,
    PathAttributionPatching_LimitedLevelWidth,
    PathMessagePatching,
    PathMessagePatching_BestFirstSearch,
    PathMessagePatching_LimitedLevelWidth,
)
import pickle as pkl
import os
from ipe.experiment import ExperimentManager
import signal
import numpy as np

############################################### Fixtures #####################################################

@pytest.fixture(scope="module")
def model():
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
    return model

@pytest.fixture(scope="module")
def prompts():
    return ["Jimmy", "Bill"]

@pytest.fixture(scope="module")
def targets():
    return [" Carter", " Clinton"]

@pytest.fixture(scope="module")
def cf_prompts():
    return ["Harry", "Hello"]

@pytest.fixture(scope="module")
def cf_targets():
    return [" Potter", " World"]

@pytest.fixture(scope="module")
def experiment_manager(model, prompts, targets, cf_prompts, cf_targets):
    return ExperimentManager(
        model=model,
        prompts=prompts,
        targets=targets,
        cf_prompts=cf_prompts,
        cf_targets=cf_targets,
        algorithm='PathAttributionPatching',
        search_strategy='BestFirstSearch',
        metric='target_logit_percentage'
    )

############################################### Tests #####################################################

def test_experiment_manager_init_invalid_prompt_lengths(model, prompts, targets):
    invalid_prompts = ["Hello", "Hi there world"]  # Different lengths
    with pytest.raises(AssertionError):
        ExperimentManager(
            model=model,
            prompts=invalid_prompts,
            targets=targets,
            positional_search=True
        )

def test_experiment_manager_init_invalid_targets(model, prompts):
    invalid_targets = ["Hello world"]  # Multi-token
    with pytest.raises(AssertionError):
        ExperimentManager(
            model=model,
            prompts=prompts,
            targets=invalid_targets
        )

def test_load_metric_unknown(model):
    with pytest.raises(ValueError, match="Unknown metric"):
        ExperimentManager(
            model=model,
            prompts=["Hello"],
            targets=["!"],
            metric='unknown_metric'
        )

def test_load_algorithm_unknown(model):
    with pytest.raises(ValueError, match="Unknown algorithm"):
        ExperimentManager(
            model=model,
            prompts=["Hello"],
            targets=["!"],
            algorithm='UnknownAlgo'
        )

def test_load_root(model, prompts, targets):
    exp = ExperimentManager(
        model=model,
        prompts=prompts,
        targets=targets
    )
    assert isinstance(exp.root, FINAL_Node)
    assert exp.root.layer == model.cfg.n_layers - 1

def test_save_paths(experiment_manager, tmp_path):
    filepath = tmp_path / "test_paths.pkl"
    experiment_manager.paths = [(0., [experiment_manager.root])]  # Dummy path
    experiment_manager.save_paths(clean=True, filepath=str(filepath))
    experiment_manager.paths = None
    assert os.path.exists(filepath)
    with open(filepath, 'rb') as f:
        loaded_paths = pkl.load(f)
    assert isinstance(loaded_paths, list)

def test_set_custom_metric(experiment_manager):
    custom_metric = partial(lambda corrupted_resid, **kwargs: 1.0)
    experiment_manager.set_custom_metric(custom_metric)
    assert experiment_manager.metric == custom_metric
    assert experiment_manager.root.metric == custom_metric

def test_plot_no_paths(experiment_manager):
    experiment_manager.paths = []
    with pytest.raises(AssertionError, match="No paths to plot"):
        experiment_manager.plot()

# New comprehensive test: ensure all algorithm x search_strategy x metric combinations can be initialized
@pytest.mark.parametrize("algorithm_name,search_strategy,expected_func", [
    ("PathAttributionPatching", "Threshold", PathAttributionPatching),
    ("PathAttributionPatching", "BestFirstSearch", PathAttributionPatching_BestFirstSearch),
    ("PathAttributionPatching", "LimitedLevelWidth", PathAttributionPatching_LimitedLevelWidth),
    ("PathMessagePatching", "Threshold", PathMessagePatching),
    ("PathMessagePatching", "BestFirstSearch", PathMessagePatching_BestFirstSearch),
    ("PathMessagePatching", "LimitedLevelWidth", PathMessagePatching_LimitedLevelWidth),
])
@pytest.mark.parametrize("metric_name", [
    "logit_difference",
    "target_logit_percentage",
    "target_probability_percentage",
    "kl_divergence",
    "indirect_effect",
])
def test_all_algorithm_searchstrategy_metric_combinations(
    model, prompts, targets, cf_prompts, cf_targets,
    algorithm_name, search_strategy, expected_func, metric_name
):
    # Adjust args depending on metric requirements
    kwargs = dict(model=model, prompts=prompts)
    # kl_divergence doesn't require targets; avoid providing targets to prevent token checks
    if metric_name == "kl_divergence":
        kwargs["targets"] = []
        kwargs["cf_targets"] = []
        kwargs["cf_prompts"] = None
    else:
        # for metrics likely needing counterfactuals, provide them
        if metric_name in ("logit_difference", "indirect_effect"):
            kwargs["targets"] = targets
            kwargs["cf_prompts"] = cf_prompts
            kwargs["cf_targets"] = cf_targets
        else:
            kwargs["targets"] = targets
            kwargs["cf_prompts"] = None
            kwargs["cf_targets"] = None

    # Provide algorithm and search strategy and metric
    kwargs["algorithm"] = algorithm_name
    kwargs["search_strategy"] = search_strategy
    kwargs["metric"] = metric_name

    # Ensure limited search space for testing
    algorithm_params = {}
    if search_strategy == "LimitedLevelWidth":
        algorithm_params["max_width"] = 1
    elif search_strategy == "Threshold":
        algorithm_params["min_contribution"] = np.inf  # Extremely high to make sure it is fast
    elif search_strategy == "BestFirstSearch":
        algorithm_params["top_n"] = 1
        algorithm_params["max_time"] = 20
    kwargs["algorithm_params"] = algorithm_params
    # This should not raise; initialization runs load_metric, load_root, load_algorithm and check_validity
    exp = ExperimentManager(**kwargs)

    # sanity checks
    assert callable(exp.metric)
    assert exp.metric.func.__name__ == metric_name
    assert callable(exp.algorithm)
    assert isinstance(exp.root, FINAL_Node)
    # check algorithm mapping
    assert exp.algorithm.func == expected_func

    # Test that the algorithm runs without errors
    def timeout_handler(signum, frame):
        raise TimeoutError("Function took too long")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)
    try:
        paths = exp.run(return_paths=True)
        signal.alarm(0)  # Cancel the alarm
    except TimeoutError:
        raise AssertionError("Test timed out after 30 seconds")
    assert isinstance(paths, list)
