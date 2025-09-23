from ipe.metrics import target_logit_percentage, target_probability_percentage, logit_difference, kl_divergence, indirect_effect
from ipe.nodes import FINAL_Node
from ipe.graph_search import (
    PathAttributionPatching,
    PathAttributionPatching_BestFirstSearch,
    PathAttributionPatching_LimitedLevelWidth,
    IsolatingPathEffect,
    IsolatingPathEffect_BestFirstSearch,
    IsolatingPathEffect_LimitedLevelWidth
)
from ipe.webutils.image_nodes import get_image_path, make_graph_from_paths
from ipe.plot.graph_plot import plot_transformer_paths
from ipe.miscellanea import get_function_params
from ipe.paths import clean_paths

from transformer_lens import HookedTransformer
from functools import partial
from copy import deepcopy
import pickle as pkl

class ExperimentManager:
    def __init__(
        self,
        model: HookedTransformer,
        prompts: list[str],
        targets: list[str],
        cf_prompts: list[str] = None,
        cf_targets: list[str] = None,
        algorithm: str = 'PathAttributionPatching',
        search_strategy: str = 'BestFirstSearch',
        algorithm_params: dict = {},
        metric: str = 'target_logit_percentage',
        metric_params: dict = {},
        positional_search: bool = True,
        patch_type: str = 'auto'
    ):
        self.model = model

        self.prompts = prompts
        self.targets = targets
        self.cf_prompts = cf_prompts
        self.cf_targets = cf_targets
        _, self.cache = self.model.run_with_cache(self.prompts, prepend_bos=True)
        self.cache = dict(self.cache)
        self.target_tokens = [self.model.to_single_token(t) for t in self.targets]
        self.clean_final_resid = self.cache[f'blocks.{self.model.cfg.n_layers - 1}.hook_resid_post']
        _, self.cf_cache = self.model.run_with_cache(self.cf_prompts, prepend_bos=True) if cf_prompts else (None, {})
        self.cf_cache = dict(self.cf_cache)
        self.cf_target_tokens = [self.model.to_single_token(t) for t in self.cf_targets] if cf_targets else []
        self.cf_final_resid = self.cf_cache[f'blocks.{self.model.cfg.n_layers - 1}.hook_resid_post'] if cf_prompts else None

        self.positional_search = positional_search
        if patch_type == 'auto':
            self.patch_type = 'counterfactual' if cf_prompts else 'zero'
        else:
            self.patch_type = patch_type

        self.load_metric(metric, metric_params)
        self.load_root()
        self.load_algorithm(algorithm, search_strategy, algorithm_params)

        self.paths = []

    
    def plot(
        self,
        cmap='tab20',
        heads_per_row: int = 4,
        save_fig: bool = False,
        save_path: str = 'transformer_paths.png',
        max_w: float = None,
        color_scheme: str = 'path_index',
        divide_heads: bool = True
    ):
        assert self.paths, "No paths to plot. Please run the experiment first."
        image_paths = [get_image_path(p, divide_heads=divide_heads) for p in self.paths]
        
        n_positions = self.cache['blocks.0.hook_resid_post'].shape[1] if self.positional_search else 1

        G = make_graph_from_paths(
            paths=image_paths,
            n_layers=self.model.cfg.n_layers,
            n_heads=self.model.cfg.n_heads,
            n_positions=n_positions,
            divide_heads=divide_heads
        )

        plot_transformer_paths(
            G=G,
            n_layers=self.model.cfg.n_layers,
            n_heads=self.model.cfg.n_heads,
            n_positions=n_positions,
            example_input=self.model.to_str_tokens(self.prompts[0], prepend_bos=True) if (self.prompts and self.positional_search)else [""]*n_positions,
            example_output=[""]*(n_positions-1) + self.model.to_str_tokens(self.targets[0], prepend_bos=False) if self.targets else [""],
            cmap_name=cmap,
            heads_per_row=heads_per_row,
            save_fig=save_fig,
            save_path=save_path,
            max_w=max_w,
            color_scheme=color_scheme,
            divide_heads=divide_heads
        )

    def run(self, return_paths=True):
        self.paths = self.algorithm()
        if return_paths:
            return self.paths
    
    def save_paths(self, clean=True, filepath='./paths.pkl'):
        if not self.paths:
            self.run()
        if clean:
            cleaned_paths = deepcopy(self.paths)
            cleaned_paths = clean_paths(cleaned_paths)
            pkl.dump(cleaned_paths, filepath)
        else:
            pkl.dump(self.paths, filepath)
        

    def set_custom_metric(
        self,
        metric: callable
    ):
        self.metric = metric
        self.root.metric = metric

    def load_root(
        self,
    ):
        self.root = FINAL_Node(
            model=self.model,
            layer=self.model.cfg.n_layers - 1,
            position=self.cache['blocks.0.hook_resid_post'].shape[1] - 1 if self.positional_search else None,
            msg_cache=self.cache,
            cf_cache=self.cf_cache,
            metric=self.metric,
            patch_type=self.patch_type
        )
    
    def load_metric(
        self,
        metric: str,
        metric_params: dict = {}
    ):
        require_baseline = False
        if metric == 'indirect_effect':
            function = indirect_effect
        elif metric == 'target_logit_percentage':
            function = target_logit_percentage
        elif metric == 'target_probability_percentage':
            function = target_probability_percentage
        elif metric == 'logit_difference':
            function = logit_difference
            require_baseline = True
        elif metric == 'kl_divergence':
            function = kl_divergence
        else:
            raise ValueError(f"Unknown metric: {metric}")

        required_params = get_function_params(function, which='required')
        if required_params.pop('corrupted_resid', 'error')=='error':
            raise ValueError(f"The metric function must have a 'corrupted_resid' parameter.")
        optional_params = get_function_params(function, which='default')

        missing_parameters = set(required_params.keys()) - set(metric_params.keys())
        self_parameters = self.__dict__.keys()
        for param in missing_parameters:
            if param in self_parameters:
                if len(str(self.__dict__[param])) > 40:
                    print(f"WARNING: [load_metric] Using ExperimentManager attribute for '{param}': (value too long to display)")
                else:
                    print(f"WARNING: [load_metric] Using ExperimentManager attribute for '{param}': {self.__dict__[param]}")
                metric_params[param] = self.__dict__[param]
        missing_parameters = set(required_params.keys()) - set(metric_params.keys())
        if missing_parameters:
            raise ValueError(f"Missing required parameters for metric '{metric}': {missing_parameters}")
        non_modified_defaults = {k: v for k, v in optional_params.items() if k not in metric_params}

        metric_params_complete = {**non_modified_defaults, **metric_params}
        self.metric = partial(function, **metric_params_complete)

        for k, v in non_modified_defaults.items():
            if k == 'baseline_value' and require_baseline:
                baseline = partial(function, **metric_params_complete)(self.cache[f'blocks.{self.model.cfg.n_layers - 1}.hook_resid_post'])
                metric_params_complete['baseline'] = baseline
                self.metric = partial(function, **metric_params_complete)
                print(f"WARNING: [load_metric] Using computed baseline for '{k}': {baseline}")
            else:
                print(f"WARNING: [load_metric] Using default parameter for '{k}': {v}")

    def load_algorithm(
        self, 
        algorithm: str,
        search_strategy: str,
        algorithm_params: dict = {}
        ):
        if algorithm == 'PathAttributionPatching':
            if search_strategy == 'Base':
                algorithm_function = PathAttributionPatching
            elif search_strategy == 'BestFirstSearch':
                algorithm_function = PathAttributionPatching_BestFirstSearch
            elif search_strategy == 'LimitedLevelWidth':
                algorithm_function = PathAttributionPatching_LimitedLevelWidth
            else:
                raise ValueError(f"Unknown search strategy: {search_strategy}, available: ['Base', 'BestFirstSearch', 'LimitedLevelWidth']")
        elif algorithm == 'IsolatingPathEffect':
            if search_strategy == 'Base':
                algorithm_function = IsolatingPathEffect
            elif search_strategy == 'BestFirstSearch':
                algorithm_function = IsolatingPathEffect_BestFirstSearch
            elif search_strategy == 'LimitedLevelWidth':
                algorithm_function = IsolatingPathEffect_LimitedLevelWidth
            else:
                raise ValueError(f"Unknown search strategy: {search_strategy}, available: ['Base', 'BestFirstSearch', 'LimitedLevelWidth']")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}, available: ['PathAttributionPatching', 'IsolatingPathEffect']")
        
        all_parameters = get_function_params(algorithm_function, which='all')
        if 'metric' in all_parameters:
            if 'metric' in algorithm_params:
                print("WARNING: [load_algorithm] Overriding provided metric with the one specified in the algorithm parameters.")
            algorithm_params['metric'] = self.metric
        if 'model' in all_parameters:
            if 'model' in algorithm_params:
                print("WARNING: [load_algorithm] Overriding provided model with the one specified in the algorithm parameters.")
            algorithm_params['model'] = self.model
        if 'root' in all_parameters:
            if 'root' in algorithm_params:
                print("WARNING: [load_algorithm] Overriding provided root with the one specified in the algorithm parameters.")
            algorithm_params['root'] = self.root
       
        required_params = get_function_params(algorithm_function, which='required')
        default_params = get_function_params(algorithm_function, which='default')

        missing_parameters = set(required_params.keys()) - set(algorithm_params.keys())
        if missing_parameters:
            raise ValueError(f"Missing required parameters for algorithm '{algorithm}': {missing_parameters}")
        
        non_modified_defaults = {k: v for k, v in default_params.items() if k not in algorithm_params}
        for k, v in non_modified_defaults.items():
            print(f"WARNING: [load_algorithm] Using default parameter for '{k}': {v}")
        
        algorithm_params_complete = {**non_modified_defaults, **algorithm_params}

        self.algorithm = partial(algorithm_function, **algorithm_params_complete)
        

    

    

