from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pickle
import os
import logging
from functools import partial
import json
import torch

from ipe.webutils.image_nodes import get_image_paths, create_graph_data
from ipe.nodes import FINAL_Node
from ipe.paths import get_path_msgs
from ipe.miscellanea import get_topk
from ipe.graph_search import PathAttributionPatching, IsolatingPathEffect_BW
from ipe.metrics import compare_token_logit
from ipe.webutils.model import load_model, load_model_config, load_tokenizer

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG) 
app.logger.debug("running...")
# Get allowed origin (can be comma-separated if needed)
# allowed_origin = os.getenv("ALLOWED_ORIGIN", "http://127.0.0.1")

# CORS setup — wrap in list if it's a single string origin
# CORS(app, resources={r"/api/*": {"origins": [allowed_origin]}})
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
	app.logger.debug("received")
	data = request.json
	model_name = data.get('model_name', 'gpt2-small')
	model_config = models_config[model_name]
	tokenizer = load_tokenizer(model_name, model_config)
	if data.get('precomputed', False):
		app.logger.debug("Using precomputed paths")
		task_name = data.get('task_name', 'Indirect Object Identification')
		task_shortname = sample_prompts[task_name]['shortname']
		search_mode = data.get('mode', 'Probability')

		prompt = sample_prompts[task_name]["prompt"]
		target = sample_prompts[task_name]["target"]

		if not os.path.exists(f'data/{model_name}/{task_shortname}/paths{search_mode}.pkl'):
			return jsonify({'error': 'Paths not found'}), 404
		paths = pickle.load(open(f'data/{model_name}/{task_shortname}/paths{search_mode}.pkl', 'rb'))
		img_node_paths = [get_image_paths(p, divide_heads=data.get('divide_heads', True)) for p in paths]

		if not os.path.exists(f'data/{model_name}/{task_shortname}/messages{search_mode}.pkl'):
			try:
				model = load_model(model_name)
			except Exception as e:
				return jsonify({'error': str(e)}), 500
			app.logger.debug(f"Running inference on {prompt}")
			_, cache = model.run_with_cache(prompt)
			app.logger.debug("Computig messages")
			global_path_store = {
				i: {'path': img_node_paths[i][1], 'messages': get_path_msgs(path=paths[i][1], msg_cache=dict(cache), model=model)}
				for i in range(len(img_node_paths))
			}
			with open(f'data/{model_name}/{task_shortname}/messages{search_mode}.pkl', 'wb') as f:
				pickle.dump(global_path_store, f)
		else:
			with open(f'data/{model_name}/{task_shortname}/messages{search_mode}.pkl', 'rb') as f:
				global_path_store = pickle.load(f)
	
	else:
		app.logger.debug("Computing paths")
		try:
			prompt = data['prompt']
			target = data['target']
		except KeyError as e:
			return jsonify({'error': f'Missing field: {str(e)}'}), 400
		try:
			model = load_model(model_name, required_bytes=model_config['required_bytes'], device=device)
		except Exception as e:
			return jsonify({'error': str(e)}), 500
		_, cache = model.run_with_cache(prompt)

		prompt_tokenized = model.to_tokens(prompt, prepend_bos=True)
		target_tokenized = model.to_tokens(target, prepend_bos=False)

		if len(target_tokenized[0]) > 1:
			return jsonify({'error': 'Target token must be a single token.'}), 400
		
		print(cache['hook_embed'].shape, prompt_tokenized.shape, target_tokenized.shape)
		metric = partial(compare_token_logit, clean_resid=cache[f'blocks.{model.cfg.n_layers - 1}.hook_resid_post'], model=model, target_tokens=target_tokenized[0])
		# paths= PathAttributionPatching(
		# 	model=model,
		# 	metric=metric,
		# 	root=FINAL_Node(
		# 		model=model,
		# 		layer=model.cfg.n_layers - 1,
		# 		position=prompt_tokenized.shape[1] - 1,
		# 		msg_cache=dict(cache),
		# 		metric=metric
		# 	),
		# 	min_contribution=0.75,
		# 	include_negative=True,
		# )
		paths = IsolatingPathEffect_BW(
			model=model,
			metric=metric,
			root=FINAL_Node(
				model=model,
				layer=model.cfg.n_layers - 1,
				position=prompt_tokenized.shape[1] - 1,
				msg_cache=dict(cache),
				metric=metric
			),
			min_contribution=1.5,
			include_negative=True,
			return_all=False,
			batch_positions=True,
			batch_heads=True
		)

		print(f"Found {len(paths)} paths")
		print(paths)
		img_node_paths = [get_image_paths(p, divide_heads=data.get('divide_heads', True)) for p in paths]
		global_path_store = {
			i: {'path': img_node_paths[i][1], 'messages': get_path_msgs(path=path[1], msg_cache=dict(cache), model=model)}
			for i, path in enumerate(paths)
		}
	print(global_path_store)
	app.logger.debug("Storing paths in global store")
	app.config['GLOBAL_PATH_STORE'] = global_path_store
	config = load_model_config(model_name=data.get('model_name', 'gpt2-small'), config=model_config)
	n_layers = config.n_layer
	n_heads = config.n_head

	return jsonify(create_graph_data(img_node_paths, n_layers, n_heads, paths[0][1][-1].position + 1, data.get('divide_heads', True), ['|<bos>|'] + tokenizer.tokenize(prompt), ['' for _ in tokenizer.tokenize(prompt)] + tokenizer.tokenize(target)))

@app.route('/api/get_path_details/<int:path_idx>', methods=['POST'])
def get_path_details(path_idx):
	data = request.json
	model_name = data.get('model_name', 'gpt2-small')
	try:
		path_store = app.config.get('GLOBAL_PATH_STORE', {})
		print(path_store)
		if path_idx not in path_store:
			return jsonify({'error': 'Path not found'}), 404
		
		path_info = path_store[path_idx]
		components = path_info['path']
		app.logger.debug("loading model...")
		model = load_model(model_name=model_name, required_bytes=models_config[model_name]['required_bytes'], device='cpu')
		app.logger.debug("model loaded")
		decoding_info = [get_topk(model, path_info['messages'][i][0][components[i].position], topk=10) for i in range(len(components))]

		path_data = [
			{
				'cmpt': n.cmpt, 
				'layer': n.layer, 
				'head': n.head_idx, 
				'position': n.position,
				'in_type': n.in_type,
				'shape': 'square' if n.cmpt in ['EMBED', 'FINAL', 'MLP'] else 
						'circle' if n.cmpt == 'ATTN' else 'circle',
				'probs': decoding_info[i]['topk_probs'],
				'tokens': decoding_info[i]['topk_strtokens'],
			}
			for i, n in enumerate(path_info['path'])
		]
		
		return jsonify({
			'path_data': path_data, 
		})
	except Exception as e:
		app.logger.error(f"Error in get_path_details: {str(e)}")
		return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
	app.run(debug=True)
