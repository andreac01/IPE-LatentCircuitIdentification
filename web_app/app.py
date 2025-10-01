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
from ipe.webutils.model import load_model, load_model_config, load_tokenizer
from ipe.experiment import ExperimentManager
import uuid
import threading

# load the evnironment
load_dotenv()
DATA_DIR = './data'
MODEL_DIR = './models'
# A directory to store job results as files
JOB_RESULTS_DIR = os.path.join(DATA_DIR, 'job_results')
os.makedirs(JOB_RESULTS_DIR, exist_ok=True)
DATA_TEMP = os.path.join(DATA_DIR, 'temp')
os.makedirs(DATA_TEMP, exist_ok=True)

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


def run_model_background(data, job_id):
	"""
	This function runs the model computation in a background thread.
	It's a refactored version of the original run_model logic.
	"""
	with app.app_context():
		try:
			app.logger.debug(f"[{job_id}] Starting background computation.")
			model_name = data.get('model_name', 'gpt2-small')
			model_config = models_config[model_name]
			tokenizer = load_tokenizer(model_config)
			try:
				model = load_model(model_name, required_bytes=model_config['required_bytes'], device=device, cache_dir=MODEL_DIR)
			except Exception as e:
				job_result = {'status': 'error', 'message': str(e)}
				with open(os.path.join(JOB_RESULTS_DIR, f'{job_id}.json'), 'w') as f:
					json.dump(job_result, f)
				return

			path_details_store = {}
			request_random_uuid_string = None
			predictions = {}

			if data.get('precomputed', False):
				app.logger.debug(f"[{job_id}] Using precomputed paths")
				task_name = data.get('task_name', 'Indirect Object Identification')
				task_shortname = sample_prompts[task_name]['shortname']
				search_mode = data.get('mode', 'Probability')
				prompt = sample_prompts[task_name]["prompt"]
				target = sample_prompts[task_name]["target"]
				
				base_dir = os.path.join(DATA_DIR, model_name, task_shortname)
				path_file = os.path.join(base_dir, f'paths{search_mode}.pkl')
				decoding_file = os.path.join(base_dir, f'top10{search_mode}.pkl')

				if not os.path.exists(path_file):
					app.logger.error(f"Precomputed paths file not found at: {path_file}")
					raise FileNotFoundError("Precomputed paths not found")
				paths = pickle.load(open(path_file, 'rb'))
				
				if os.path.exists(decoding_file):
					with open(decoding_file, 'rb') as f:
						path_details_store = pickle.load(f)
				else:
					_, cache = model.run_with_cache(prompt)
					img_node_paths = [get_image_path(p, divide_heads=data.get('divide_heads', True)) for p in paths]
					for i, path_tuple in enumerate(paths):
						messages = get_path_msgs(path=path_tuple[1], messages=[], msg_cache=dict(cache), model=model)
						image_path = img_node_paths[i][1]
						decoding_info = [get_topk(model, messages[j][0][image_path[j].position].detach().clone(), topk=10) for j in range(len(image_path))]
						path_data = [{'cmpt': n.cmpt, 'layer': n.layer, 'head': n.head_idx, 'position': n.position, 'in_type': n.in_type, 'shape': 'square' if n.cmpt in ['EMBED', 'FINAL', 'MLP'] else 'circle', 'probs': decoding_info[j]['topk_probs'], 'tokens': decoding_info[j]['topk_strtokens']} for j, n in enumerate(image_path)]
						path_details_store[str(i)] = {'path': path_data, 'weight': float(path_tuple[0])}
					path_details_store['predictions'] = get_topk(model, cache[f'blocks.{model.cfg.n_layers - 1}.hook_resid_post'][0][-1], topk=10)
					with open(decoding_file, 'wb') as f:
						pickle.dump(path_details_store, f)
				predictions = path_details_store.pop('predictions')
					
			else:
				app.logger.debug(f"[{job_id}] Computing paths and details on the fly")
				prompt = data['prompt']
				target = data['target']
				metric = 'target_logit_percentage' if target != "" else 'kl_divergence'
				target_list = [target] if target != "" else []
				
				experiment = ExperimentManager(model=model, prompts=[prompt], targets=target_list, positional_search=True, metric=metric, patch_type='zero', search_strategy='BestFirstSearch', algorithm='PathAttributionPatching', algorithm_params={"top_n": 100, "include_negative": True, "max_time": 120})
				predictions = get_topk(model, experiment.cache[f'blocks.{model.cfg.n_layers- 1}.hook_resid_post'][0][-1], topk=10)
				if target == "":
					target = predictions['topk_strtokens'][0].replace('Ġ', '_').replace(' ', '_')
				paths = experiment.run()
				request_random_uuid_string = str(uuid.uuid1())
				experiment.save_paths(clean=True, filepath=os.path.join(DATA_TEMP, f'{request_random_uuid_string}.pkl'))
				with open(os.path.join(DATA_DIR, 'temp', f'{request_random_uuid_string}_request.json'), 'w') as f:
					json.dump({'prompt': prompt, 'target': target, 'model': model_name}, f)
				
				img_node_paths_runtime = [get_image_path(p, divide_heads=data.get('divide_heads', True)) for p in paths]
				for i, path_tuple in enumerate(paths):
					messages = get_path_msgs(path=path_tuple[1], messages=[], msg_cache=dict(experiment.cache), model=model)
					image_path = img_node_paths_runtime[i][1]
					decoding_info = [get_topk(model, messages[j][0][image_path[j].position], topk=10) for j in range(len(image_path))]
					path_data = [{'cmpt': n.cmpt, 'layer': n.layer, 'head': n.head_idx, 'position': n.position, 'in_type': n.in_type, 'shape': 'square' if n.cmpt in ['EMBED', 'FINAL', 'MLP'] else 'circle', 'probs': decoding_info[j]['topk_probs'], 'tokens': decoding_info[j]['topk_strtokens']} for j, n in enumerate(image_path)]
					path_details_store[str(i)] = {'path': path_data, 'weight': float(path_tuple[0])}
				experiment = None

			img_node_paths = [get_image_path(p, divide_heads=data.get('divide_heads', True)) for p in paths]
			config = load_model_config(config=model_config)
			n_layer = config.n_layer if hasattr(config, 'n_layer') else config.num_hidden_layers
			n_head = config.n_head if hasattr(config, 'n_head') else config.num_attention_heads
			graph_data = create_graph_data(img_node_paths, n_layer, n_head, paths[0][1][-1].position + 1, data.get('divide_heads', True), ['|<bos>|'] + tokenizer.tokenize(prompt), ['' for _ in tokenizer.tokenize(prompt)] + tokenizer.tokenize(target))
			
			final_result = {
				'graphData': graph_data,
				'pathDetails': path_details_store,
				'uuid': request_random_uuid_string,
				'predicted_tokens': predictions['topk_strtokens'],
				'predicted_probabilities': predictions['topk_probs'],
			}
			job_result = {'status': 'complete', 'result': final_result}
			with open(os.path.join(JOB_RESULTS_DIR, f'{job_id}.json'), 'w') as f:
				json.dump(job_result, f)
			app.logger.debug(f"[{job_id}] Background computation finished.")

		except Exception as e:
			app.logger.exception(f"[{job_id}] Error in background job.")
			job_result = {'status': 'error', 'message': str(e)}
			with open(os.path.join(JOB_RESULTS_DIR, f'{job_id}.json'), 'w') as f:
				json.dump(job_result, f)
		finally:
			# Clean up to free memory
			model = None
			paths = None
			torch.cuda.empty_cache()

@app.route('/api/run_model', methods=['POST'])
def run_model():
	"""
	Starts a model computation job in the background.
	Immediately returns a job ID.
	"""
	job_id = str(uuid.uuid4())
	data = request.json
	app.logger.debug(f"Received job {job_id} with data: {data}")

	# Set initial status by creating a file
	job_result = {'status': 'processing'}
	with open(os.path.join(JOB_RESULTS_DIR, f'{job_id}.json'), 'w') as f:
		json.dump(job_result, f)

	# Start the background task
	thread = threading.Thread(target=run_model_background, args=(data, job_id))
	thread.daemon = True
	thread.start()

	# Return acknowledgment with job ID
	return jsonify({'status': 'processing', 'job_id': job_id}), 202

@app.route('/api/get_result/<job_id>', methods=['GET'])
def get_result(job_id):
	"""
	Poll this endpoint to get the result of a computation job.
	"""
	app.logger.debug(f"Polling for job_id: {job_id}")
	job_file = os.path.join(JOB_RESULTS_DIR, f'{job_id}.json')
	if not os.path.exists(job_file):
		return jsonify({'status': 'error', 'message': 'Job ID not found'}), 404
	
	with open(job_file, 'r') as f:
		result = json.load(f)
	
	if result['status'] == 'complete':
		# Optionally remove the file after retrieval
		os.remove(job_file)
		return jsonify(result)
	
	return jsonify(result)


if __name__ == '__main__':
	app.run(debug=True)
