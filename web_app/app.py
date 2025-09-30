from flask import Flask, render_template, jsonify, request, send_from_directory
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
from ipe.experiment import ExperimentManager
import uuid

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

@app.route('/api/download_paths', methods=['POST'])
def download_paths():
	"""
	Handles downloading the precomputed .pkl path file.
	This endpoint is designed to work when the user has selected a precomputed visualization.
	It locates the corresponding .pkl file on the server and sends it for download.
	"""
	payload = request.get_json(silent=True)
	if not payload or 'params' not in payload:
		return jsonify({'error': 'Invalid or empty JSON payload'}), 400

	params = payload.get('params', {})

	try:
		if not params.get('precomputed'):
			uuid = params['uuid']
			path_file = os.path.join(DATA_DIR, 'temp', f'{uuid}.pkl')
			base_dir = os.path.join(DATA_DIR, 'temp')
			filename = f'{uuid}.pkl'
		else:
			model_name = params['model_name']
			task_name = params['task_name']
			search_mode = params['mode']
			task_shortname = sample_prompts[task_name]['shortname']
			
			# Construct the path to the .pkl file, mirroring the logic in run_model
			base_dir = os.path.join(DATA_DIR, model_name, task_shortname)
			filename = f'paths{search_mode}.pkl'
			path_file = os.path.join(base_dir, filename)


		if not os.path.exists(path_file):
			app.logger.error(f"Precomputed file not found at: {path_file}")
			return jsonify({'error': 'The requested precomputed path file was not found on the server.'}), 404

		data_dir = os.path.abspath(DATA_DIR)
		base_dir = os.path.abspath(base_dir)
		if not base_dir.startswith(data_dir):
			app.logger.error(f"Attempted access outside of DATA_DIR: {base_dir}")
			return jsonify({'error': 'Invalid file path.'}), 400

		app.logger.debug(f"Sending file for download: {path_file}")
		return send_from_directory(directory=base_dir, path=filename, as_attachment=True)

	except KeyError as e:
		app.logger.error(f"Missing parameter in download request: {e}")
		return jsonify({'error': f'Missing required parameter for download: {e}'}), 400
	except Exception as e:
		app.logger.exception("An unexpected error occurred during file download.")
		return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500


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
	app.logger.debug(f"Request data: {data}")
	model_name = data.get('model_name', 'gpt2-small')
	model_config = models_config[model_name]
	tokenizer = load_tokenizer(model_config)
	
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
		
		metric = 'target_logit_percentage' if target != "" else 'kl_divergence'
		if target == "":
			target = "out"
			target_list = []
		else:
			target_list = [target]
		experiment = ExperimentManager(
			model=model,
			prompts=[prompt],
			targets=target_list,
			positional_search=True,
			metric=metric,
			patch_type='zero',
			search_strategy='BestFirstSearch',
			algorithm='PathAttributionPatching',
			algorithm_params={"top_n": 100}
		)
		
		paths = experiment.run()
		request_random_uuid_string = str(uuid.uuid1())
		# Save the paths for potential later download
		experiment.save_paths(clean=True, filepath=os.path.join('./data', 'temp', f'{request_random_uuid_string}.pkl'))
		with open(os.path.join('./data', 'temp', f'{request_random_uuid_string}_request.json'), 'w') as f:
			json.dump({'prompt': prompt, 'target': target, 'model': model_name}, f)
		
		# We still compute details directly, but we don't save them in this mode
		img_node_paths_runtime = [get_image_path(p, divide_heads=data.get('divide_heads', True)) for p in paths]
		for i, path_tuple in enumerate(paths):
			messages = get_path_msgs(path=path_tuple[1], messages=[], msg_cache=dict(experiment.cache), model=model)
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
		experiment = None  # Free memory

	img_node_paths = [get_image_path(p, divide_heads=data.get('divide_heads', True)) for p in paths]
	config = load_model_config(config=model_config)
	n_layer = config.n_layer if hasattr(config, 'n_layer') else config.num_hidden_layers
	n_head = config.n_head if hasattr(config, 'n_head') else config.num_attention_heads
	graph_data = create_graph_data(
		img_node_paths, n_layer, n_head, 
		paths[0][1][-1].position + 1, data.get('divide_heads', True), 
		['|<bos>|'] + tokenizer.tokenize(prompt), 
		['' for _ in tokenizer.tokenize(prompt)] + tokenizer.tokenize(target)
	)
	paths = None  # Free memory
	torch.cuda.empty_cache()
	return jsonify({
		'graphData': graph_data,
		'pathDetails': path_details_store,
		'uuid': request_random_uuid_string if not data.get('precomputed', False) else None
	})


if __name__ == '__main__':
	app.run(debug=True)
