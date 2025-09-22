from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pickle
import os
import logging
from functools import partial
from dotenv import load_dotenv
import json
import torch

from ipe.webutils.image_nodes import get_image_path, create_graph_data
from ipe.nodes import FINAL_Node
from ipe.paths import get_path_msgs
from ipe.miscellanea import get_topk
from ipe.graph_search import PathAttributionPatching
from ipe.metrics import target_logit_percentage
from ipe.webutils.model import load_model, load_model_config, load_tokenizer

# load the evnironment
load_dotenv()
DATA_DIR = os.getenv('DATA_DIR', './data')
MODEL_DIR = os.getenv('MODEL_DIR', './app/models')

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG) 
app.logger.debug("running...")
CORS(app, resources={r"/api/*": {"origins": ["http://127.0.0.1"]}})

with open("./configuration/sample_prompts.json", 'r') as f:
	sample_prompts = json.load(f)
with open("./configuration/models_config.json", 'r') as f:
	models_config = json.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- API Endpoints ---

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/api/run_model', methods=['POST'])
def run_model():
	"""API endpoint to run the model and compute paths.
	
	Expects a JSON payload with:
	- model_name: Name of the model to use (default: 'gpt2-small')
	- prompt: The input prompt string
	- target: The target token string
	- precomputed: Boolean indicating whether to use precomputed paths (default: False)
	- task_name: If precomputed is True, the name of the task to load paths for (default: 'Indirect Object Identification')
	- mode: If precomputed is True, the search mode ('Probability' or 'Marginalization', default: 'Probability')
	- divide_heads: Boolean indicating whether to divide head contributions (default: True)
	
	Returns a JSON response with:
	- graphData: Data for visualizing the graph
	- pathDetails: Details for each path including component info and top token probabilities"""
	app.logger.debug("received")
	data = request.json
	model_name = data.get('model_name', 'gpt2-small')
	model_config = models_config[model_name]
	tokenizer = load_tokenizer(model_name, model_config)
	
	try:
		model = load_model(model_name, required_bytes=model_config['required_bytes'], device=device, cache_dir=MODEL_DIR)
	except Exception as e:
		return jsonify({'error': str(e)}), 500

	path_details_store = {} # Initialize the dictionary to store path details

	if data.get('precomputed', False):
		app.logger.debug("Using precomputed paths")
		task_name = data.get('task_name', 'Indirect Object Identification')
		task_shortname = sample_prompts[task_name]['shortname']
		search_mode = data.get('mode', 'Probability')

		prompt = sample_prompts[task_name]["prompt"]
		target = sample_prompts[task_name]["target"]
		
		base_dir = os.path.join(DATA_DIR, model_name, task_shortname)
		path_file = os.path.join(base_dir, f'paths{search_mode}.pkl')
		decoding_file = os.path.join(base_dir, f'top10{search_mode}.pkl')

		if not os.path.exists(path_file):
			return jsonify({'error': 'Precomputed paths not found'}), 404
		paths = pickle.load(open(path_file, 'rb'))
		
		if os.path.exists(decoding_file):
			app.logger.debug("Found and loading precomputed decodings.")
			with open(decoding_file, 'rb') as f:
				path_details_store = pickle.load(f)
		else:
			app.logger.debug("Precomputed decodings not found. Computing and saving them now.")
			# We need the model's cache to compute messages
			_, cache = model.run_with_cache(prompt)
			img_node_paths = [get_image_path(p, divide_heads=data.get('divide_heads', True)) for p in paths]

			# The computation logic is the same as before
			for i, path_tuple in enumerate(paths):
				messages = get_path_msgs(path=path_tuple[1], messages=[], msg_cache=dict(cache), model=model)
				image_path = img_node_paths[i][1]
				decoding_info = [get_topk(model, messages[j][0][image_path[j].position].detach().clone(), topk=10) for j in range(len(image_path))]
			
				path_data = [
					{
						'cmpt': n.cmpt, 'layer': n.layer, 'head': n.head_idx, 
						'position': n.position, 'in_type': n.in_type,
						'shape': 'square' if n.cmpt in ['EMBED', 'FINAL', 'MLP'] else 'circle',
						'probs': decoding_info[j]['topk_probs'],
						'tokens': decoding_info[j]['topk_strtokens'],
					}
					for j, n in enumerate(image_path)
				]
				path_details_store[str(i)] = path_data
			
			# Save the newly computed decodings to the file for next time
			with open(decoding_file, 'wb') as f:
				pickle.dump(path_details_store, f)
			app.logger.debug(f"Saved decodings to {decoding_file}")
			
	else: # This is the on-the-fly computation case
		app.logger.debug("Computing paths and details on the fly")
		try:
			prompt = data['prompt']
			target = data['target']
		except KeyError as e:
			return jsonify({'error': f'Missing field: {str(e)}'}), 400
		
		_, cache = model.run_with_cache(prompt)
		prompt_tokenized = model.to_tokens(prompt, prepend_bos=True)
		target_tokenized = model.to_tokens(target, prepend_bos=False)

		if len(target_tokenized[0]) > 1:
			return jsonify({'error': 'Target token must be a single token.'}), 400
		
		metric = partial(target_logit_percentage, clean_resid=cache[f'blocks.{model.cfg.n_layers - 1}.hook_resid_post'], model=model, target_tokens=target_tokenized[0])
		paths= PathAttributionPatching(
			model=model,
			metric=metric,
			root=FINAL_Node(
				model=model,
				layer=model.cfg.n_layers - 1,
				position=prompt_tokenized.shape[1] - 1,
				msg_cache=dict(cache),
				metric=metric
			),
			min_contribution=0.65,
			include_negative=True,
		)
		
		# We still compute details directly, but we don't save them in this mode
		img_node_paths_runtime = [get_image_path(p, divide_heads=data.get('divide_heads', True)) for p in paths]
		for i, path_tuple in enumerate(paths):
			messages = get_path_msgs(path=path_tuple[1], messages=[], msg_cache=dict(cache), model=model)
			image_path = img_node_paths_runtime[i][1]
			decoding_info = [get_topk(model, messages[j][0][image_path[j].position], topk=10) for j in range(len(image_path))]
			
			path_data = [
				{
					'cmpt': n.cmpt, 'layer': n.layer, 'head': n.head_idx, 
					'position': n.position, 'in_type': n.in_type,
					'shape': 'square' if n.cmpt in ['EMBED', 'FINAL', 'MLP'] else 'circle',
					'probs': decoding_info[j]['topk_probs'],
					'tokens': decoding_info[j]['topk_strtokens'],
				}
				for j, n in enumerate(image_path)
			]
			path_details_store[str(i)] = path_data

	# --- Final Response Preparation (this part is now common to both modes) ---
	img_node_paths = [get_image_path(p, divide_heads=data.get('divide_heads', True)) for p in paths]
	config = load_model_config(model_name=model_name, config=model_config)
	graph_data = create_graph_data(
		img_node_paths, config.n_layer, config.n_head, 
		paths[0][1][-1].position + 1, data.get('divide_heads', True), 
		['|<bos>|'] + tokenizer.tokenize(prompt), 
		['' for _ in tokenizer.tokenize(prompt)] + tokenizer.tokenize(target)
	)

	return jsonify({
		'graphData': graph_data,
		'pathDetails': path_details_store
	})


if __name__ == '__main__':
	app.run(debug=True)
