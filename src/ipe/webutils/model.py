import psutil
import torch
import os
from transformer_lens import HookedTransformer
from transformers import AutoConfig, AutoTokenizer



def has_enough_memory(device: torch.device, required_bytes: int) -> bool:
	"""
	Check if at least required_bytes are available on device.
	Returns True if enough memory is present, False otherwise.

	Args:
		device (torch.device):
			The device to check memory on (e.g., 'cpu' or 'cuda').
		required_bytes (int):
			The number of bytes required.
	Returns:
		bool:
			True if enough memory is available, False otherwise.
	"""
	if device.type == "cuda":
		# torch.cuda.mem_get_info returns (free, total) in bytes
		free, _ = torch.cuda.mem_get_info(device.index)
		return free >= required_bytes
	else:
		vm = psutil.virtual_memory()
		return vm.available >= required_bytes

def download_model(model_name: str, cache_dir: str = "/app/models") -> None:
	"""Download a model to local cache directory. If the model is already present, it does nothing.
	
	Args:
		model_name (str):
			The name of the model to download (e.g., 'gpt2').
		cache_dir (str, default="/app/models"):
			The directory where the model should be cached.
	Returns:
		None
	"""
	# Create cache directory if it doesn't exist
	os.makedirs(cache_dir, exist_ok=True)
	
	# Download model using HookedTransformer
	HookedTransformer.from_pretrained(
		model_name,
		device="cpu",  # Download to CPU first
		center_unembed=False,
		center_writing_weights=False,
		torch_dtype=torch.float32,
		cache_dir=cache_dir
	)

def load_model(model_name: str, required_bytes: int = 0, device: str = 'cpu', cache_dir: str = "/app/models") -> HookedTransformer:
	"""Load (and cache) a HookedTransformer, but first check memory.
	Args:
		model_name (str):
			The name of the model to load (e.g., 'gpt2').
		required_bytes (int, default=0):
			The number of bytes required to load the model.
		device (str or torch.device, default='cpu'):
			The device to load the model onto (e.g., 'cpu' or 'cuda').
		cache_dir (str, default="/app/models"):
			The directory where the model should be cached.
	Returns:
		HookedTransformer:
			The loaded transformer model.
	"""
	
	device = torch.device(device)
	
	if not has_enough_memory(device, required_bytes):
		raise MemoryError(f"Not enough free memory on {device.type}")

	# Try to load from cache first, if fails, download and then load
	try:
		model = HookedTransformer.from_pretrained(
			model_name,
			device=device,
			center_unembed=False,
			center_writing_weights=False,
			torch_dtype=torch.float32,
			cache_dir=cache_dir
		)
	except Exception as e:
		download_model(model_name, cache_dir)
		model = HookedTransformer.from_pretrained(
			model_name,
			device=device,
			center_unembed=False,
			center_writing_weights=False,
			torch_dtype=torch.float32,
			cache_dir=cache_dir
		)
	
	return model

def load_tokenizer(str, config: dict) -> AutoTokenizer:
	"""Load the tokenizer for a given model. The tokenizer is fetched from Hugging Face using the model's configuration, to avoiding loading the entire model as required by HookedTransformer.
	
	Args:
		model_name (str):
			The name of the model whose tokenizer to load (e.g., 'gpt2').
		config (dict):
			The configuration dictionary containing 'huggingface_name'.
	Returns:
		AutoTokenizer:
			The loaded tokenizer.
	"""
	return AutoTokenizer.from_pretrained(config['huggingface_name'], trust_remote_code=True)

def load_model_config(config: dict) -> AutoConfig:
	"""Load the configuration of a model from Hugging Face.
	Args:
		model_name (str):
			The name of the model whose configuration to load (e.g., 'gpt2').
		config (dict):
			The configuration dictionary containing 'huggingface_name'.
	Returns:
		AutoConfig:
			The loaded model configuration.
	"""
	return AutoConfig.from_pretrained(config['huggingface_name'], trust_remote_code=True)
