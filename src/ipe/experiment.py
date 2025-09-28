from ipe import metrics
from ipe.metrics import target_logit_percentage, target_probability_percentage, logit_difference, kl_divergence, indirect_effect
from ipe.nodes import FINAL_Node, Node
from ipe.graph_search import (
	PathAttributionPatching,
	PathAttributionPatching_BestFirstSearch,
	PathAttributionPatching_LimitedLevelWidth,
	PathMessagePatching,
	PathMessagePatching_BestFirstSearch,
	PathMessagePatching_LimitedLevelWidth
)
from ipe.webutils.image_nodes import get_image_path, make_graph_from_paths
from ipe.plot.graph_plot import plot_transformer_paths
from ipe.plot.decoding_plot import create_interactive_decoding_plot, create_interactive_path_visualization
from ipe.miscellanea import get_function_params
from ipe.paths import clean_paths

from transformer_lens import HookedTransformer
from functools import partial
from copy import deepcopy
import pickle as pkl
from torch import Tensor


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
		"""Manages the setup and execution of path-finding experiments using a easier interface, that employs default parameters and sensible checks.

		It helps in the definition of the metric, the algorithm, running the experiment, plotting the results and decoding the residuals along the paths.
		In the setup it allows you to:
			1. Choose the metric in between:
				- :func:`ipe.metrics.target_logit_percentage` (metric = 'target_logit_percentage')
				- :func:`ipe.metrics.target_probability_percentage` (metric = 'target_probability_percentage') (default)
				- :func:`ipe.metrics.logit_difference` (metric = 'logit_difference')
				- :func:`ipe.metrics.kl_divergence` (metric = 'kl_divergence')
				- :func:`ipe.metrics.indirect_effect` (metric = 'indirect_effect')
			2. Choose the seach algorithm in between:
				- :func:`ipe.graph_search.PathAttributionPatching` (alorithm = 'PathAttributionPatching', method = 'Threshold')
				- :func:`ipe.graph_search.PathMessagePatching` (alorithm = 'PathMessagePatching', method = 'Threshold')
				- :func:`ipe.graph_search.PathAttributionPatching_BestFirstSearch` (alorithm = 'PathAttributionPatching', method = 'BestFirstSearch') (default)
				- :func:`ipe.graph_search.PathMessagePatching_BestFirstSearch` (alorithm = 'PathMessagePatching', method = 'BestFirstSearch')
				- :func:`ipe.graph_search.PathAttributionPatching_LimitedLevelWidth` (alorithm = 'PathAttributionPatching', method = 'LimitedLevelWidth')
				- :func:`ipe.graph_search.PathMessagePatching_LimitedLevelWidth` (alorithm = 'PathMessagePatching', method = 'LimitedLevelWidth')
		You can also provide custom parameters to the metric and the algorithm as a dictionary.
		
		Args:
			model (HookedTransformer): 
				The transformer model to analyze.
			prompts (list[str]):
				List of input prompts for the model. If positional_search is True, all prompts must have the same length.
			targets (list[str]):
				List of target tokens corresponding to each prompt. Each target must be a single token.
			cf_prompts (list[str], optional):
				List of counterfactual prompts for the model. If provided, must have the same length as prompts and positional_search rules apply. Defaults to None.
			cf_targets (list[str], optional):
				List of counterfactual target tokens corresponding to each counterfactual prompt. Each target must be a single token. If provided, must have the same length as targets. Defaults to None
			algorithm (str, optional):
				The path-finding algorithm to use. Options are 'PathAttributionPatching' or 'PathMessagePatching'. Defaults to 'PathAttributionPatching'.
			search_strategy (str, optional):
				The search strategy to use within the chosen algorithm. Options are 'Threshold', 'BestFirstSearch', or 'LimitedLevelWidth'. Defaults to 'BestFirstSearch'.
			algorithm_params (dict, optional):
				Custom parameters for the chosen algorithm. If provided, must be a dictionary containing parameters appropriate for the algorithm/search strategy combination. Defaults to {}.
			metric (str, optional):
				The name of the metric to use. Options are 'target_logit_percentage', 'target_probability_percentage', 'logit_difference', 'kl_divergence', or 'indirect_effect'. Defaults to 'target_logit_percentage'.
				Note that 'kl_divergence is the only metric that does not require a target token.
			metric_params (dict, optional):
				Custom parameters for the chosen metric. If provided, must be a dictionary containing parameters appropriate for the metric. Defaults to {}.
			positional_search (bool, optional):
				Whether to perform positional search. If True, all prompts (and cf_prompts if provided) must have the same length. Defaults to True.
			patch_type (str, optional):
				The type of patching to use. Options are 'zero', 'counterfactual', or 'auto'. If 'auto', uses 'counterfactual' if cf_prompts are provided, otherwise uses 'zero'. Defaults to 'auto'.
		
		Raises:
			AssertionError: If inconsistencies in the parameters are found.
			ValueError: If an unknown metric, algorithm, or search strategy is provided, or if required parameters are missing.
		
		Note:
			If you wish to use a custom metric function, you can set it using the :func:`set_custom_metric` method after initializing the ExperimentManager.
		"""

		self.model = model

		self.prompts = prompts
		self.prompt_length = len(model.to_str_tokens(prompts[0], prepend_bos=True))
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
		self.check_validity()

	def check_validity(self):
		"""Performs validity checks on the provided prompts, targets, and counterfactuals to ensure they meet the requirements for the experiment.
		Raises:
			AssertionError: If any of the validity checks fail.
		"""
		if self.positional_search:
			for p in self.prompts:
				assert self.prompt_length == len(self.model.to_str_tokens(p, prepend_bos=True)), f'Prompt {p} length "{len(self.model.to_str_tokens(p, prepend_bos=True))}" does not match length of other prompts.'
			if self.cf_prompts:
				for p in self.cf_prompts:
					assert self.prompt_length == len(self.model.to_str_tokens(p, prepend_bos=True)), f'Counterfactual prompt {p} length "{len(self.model.to_str_tokens(p, prepend_bos=True))}" does not match length of other prompts.'
		if self.metric != 'kl_divergence':
			for t in self.targets:
				assert len(self.model.to_str_tokens(t, prepend_bos=False)) == 1, f"Target '{t}' is not a single token."
			if self.cf_targets:
				for t in self.cf_targets:
					assert len(self.model.to_str_tokens(t, prepend_bos=False)) == 1, f"Counterfactual target '{t}' is not a single token."
		else:
			if len(self.targets) > 0:
				print("WARNING: 'kl_divergence' metric does not require target tokens. Ignoring provided targets.")
			if len(self.cf_targets) > 0:
				print("WARNING: 'kl_divergence' metric does not require counterfactual target tokens. Ignoring provided cf_targets.")
	
	def decode_residual(
		self,
		residual: Tensor
	):
		"""Decode a given residual stream using the model's unembedding layer and display an interactive decoding plot. This decoding may be used to analyze hidden layer messages (see www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
		The output will be an interactive plot showing the top token predictions based on the provided residual stream.
		Args:
			residual (Tensor):
				A tensor representing the residual stream to decode. Should be of shape (d_model,) or (batch_size, d_model).
		"""
		if residual.shape[1:] == self.clean_final_resid.shape[1:]:
			residual = residual[:, -1, :]
		logits = self.model.unembed(self.model.ln_final(residual))
		create_interactive_decoding_plot(logits=logits, model=self.model)
		
	def decode_paths(
		self,
		counterfactual: bool = False	
	): 
		"""Decode the residual streams along the found paths and display an interactive visualization. This visualization allows for an in-depth analysis of how different paths contribute to the model's predictions.
		
		Args:
			counterfactual (bool, optional):
				If True, visualize the messages passed during the counterfactual run. Defaults to False.
		"""
		if not self.paths:
			print("WARNING: No paths to decode. Running the experiment first.")
			self.run()
		vis_paths = [p for contrib, p in self.paths]
		if counterfactual:
			if not self.cf_cache:
				raise ValueError("No counterfactual cache available. Please provide cf_prompts and cf_targets when initializing the ExperimentManager.")
			create_interactive_path_visualization(vis_paths, self.model, self.cf_cache)
		create_interactive_path_visualization(vis_paths, self.model, self.cache)

	def plot(
		self,
		cmap='coolwarm',
		heads_per_row: int = 4,
		save_fig: bool = False,
		save_path: str = 'transformer_paths.png',
		max_w: float = None,
		color_scheme: str = 'path_weight',
		divide_heads: bool = True
	):
		"""Plot the found paths using a graph visualization. The graph illustrates the connections and contributions of different components in the model to the final output.

		Args:
			cmap (str, optional):
				The colormap to use for the plot. Defaults to 'coolwarm'.
			heads_per_row (int, optional):
				Number of attention heads to display per row in the plot. Defaults to 4.
			save_fig (bool, optional):
				If True, saves the figure to a file. Defaults to False.
			save_path (str, optional):
				The file path to save the figure if save_fig is True. Defaults to 'transformer_paths.png'.
			max_w (float, optional):
				Maximum width for the edges in the graph. If None, widths are scaled automatically. Defaults to None.
			color_scheme (str, optional):
				Color scheme for the graph. Options are 'path_weight' or 'node_type'. Defaults to 'path_weight'.
			divide_heads (bool, optional):
				If True, divides attention heads into separate nodes. Defaults to True.
	
		Raises:
			AssertionError: If no paths are available to plot.
		"""
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
		"""Run the path-finding experiment using the configured algorithm and metric. This method executes the search process and stores the resulting paths.

		Args:
			return_paths (bool, optional):
				If True, returns the found paths after execution. Defaults to True.

		Returns:
			list: A list of found paths if return_paths is True.
		"""
		self.paths = self.algorithm()
		if return_paths:
			return self.paths
	
	def save_paths(self, clean=True, filepath='./paths.pkl'):
		"""Save the found paths to a file for later analysis or visualization. Optionally, clean the paths to remove redundant information before saving.
		Args:
			clean (bool, optional):
				If True, cleans the paths to remove redundant information before saving. Defaults to True.
			filepath (str, optional):
				The file path to save the paths. Defaults to './paths.pkl'.
		Note:
			Cleaning the paths drastically reduces the saved file size by removing information like the cache, model, metric gradients, etc. that can be easily reloaded or recomputed.
		"""
		if not self.paths:
			self.run()
		if clean:
			cleaned_paths = deepcopy(self.paths)
			cleaned_paths = clean_paths(cleaned_paths)
			with open(filepath, 'wb') as f:
				pkl.dump(cleaned_paths, f)
		else:
			with open(filepath, 'wb') as f:
				pkl.dump(self.paths, f)

	def set_custom_metric(
		self,
		metric: callable
	):
		"""Set a custom metric function for the experiment. This allows for flexibility in defining how the effectiveness of paths is measured.
		Args:
			metric (callable):
				A custom metric function that takes the corrupted residual stream as input and returns a scalar value representing the metric.
		Note:
			The custom metric function must have a single parameter named 'corrupted_resid' to accept the corrupted residual stream. If you need to pass additional parameters, consider using functools.partial to create a version of your function with those parameters fixed.
		"""
		self.metric = metric
		self.root.metric = metric

	def load_root(
		self,
	):
		"""Initialize the root node for the path-finding algorithm. The root node represents the starting point of the search process and is configured based on the model, metric, position and other relevant parameters."""
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
		"""Load and configure the metric function for the experiment. This method selects the appropriate metric based on the provided name and configures it with any additional parameters.
		
		Args:
			metric (str):
				The name of the metric to use. Options are 'target_logit_percentage', 'target_probability_percentage', 'logit_difference', 'kl_divergence', or 'indirect_effect'.
			metric_params (dict, optional):
				Custom parameters for the chosen metric. If provided, must be a dictionary containing parameters appropriate for the metric. Defaults to {}.
		Raises:
			ValueError: If an unknown metric name is provided or if required parameters are missing.
		Note:
			If you wish to use a custom metric function, you can set it using the :func:`set_custom_metric` method after initializing the ExperimentManager.
		"""
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
		provided_params = list(metric_params.keys())
		for param in provided_params:
			if param not in required_params and param not in optional_params:
				print(f"WARNING: '{param}' is not a valid parameter for metric '{metric}'. It will be ignored.")
				metric_params.pop(param)
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
				metric_params_complete['baseline_value'] = baseline
				self.metric = partial(function, **metric_params_complete)
				print(f"WARNING: [load_metric] Using computed baseline for '{k}': {baseline}")
			else:
				print(f"WARNING: [load_metric] Using default parameter for '{k}': {v}")
		metric_params = {}

	def load_algorithm(
		self, 
		algorithm: str,
		search_strategy: str,
		algorithm_params: dict = {}
		):
		"""Load and configure the path-finding algorithm for the experiment. This method selects the appropriate algorithm based on the provided name and search strategy, and configures it with any additional parameters.

		Args:
			algorithm (str):
				The name of the path-finding algorithm to use. Options are 'PathAttributionPatching' or 'PathMessagePatching'.
			search_strategy (str):
				The search strategy to use within the chosen algorithm. Options are 'Threshold', 'BestFirstSearch', or 'LimitedLevelWidth'.
			algorithm_params (dict, optional):
				Custom parameters for the chosen algorithm. If provided, must be a dictionary containing parameters appropriate for the algorithm/search strategy combination. Defaults to {}.
		Raises:
			ValueError: If an unknown algorithm or search strategy is provided, or if required parameters are missing.
		"""
		if algorithm == 'PathAttributionPatching':
			if search_strategy == 'Threshold':
				algorithm_function = PathAttributionPatching
			elif search_strategy == 'BestFirstSearch':
				algorithm_function = PathAttributionPatching_BestFirstSearch
			elif search_strategy == 'LimitedLevelWidth':
				algorithm_function = PathAttributionPatching_LimitedLevelWidth
			else:
				raise ValueError(f"Unknown search strategy: {search_strategy}, available: ['Threshold', 'BestFirstSearch', 'LimitedLevelWidth']")
		elif algorithm == 'PathMessagePatching':
			if search_strategy == 'Threshold':
				algorithm_function = PathMessagePatching
			elif search_strategy == 'BestFirstSearch':
				algorithm_function = PathMessagePatching_BestFirstSearch
			elif search_strategy == 'LimitedLevelWidth':
				algorithm_function = PathMessagePatching_LimitedLevelWidth
			else:
				raise ValueError(f"Unknown search strategy: {search_strategy}, available: ['Threshold', 'BestFirstSearch', 'LimitedLevelWidth']")
		else:
			raise ValueError(f"Unknown algorithm: {algorithm}, available: ['PathAttributionPatching', 'PathMessagePatching']")
		
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
		

	

	

