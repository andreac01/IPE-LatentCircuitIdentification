import psutil
import torch
import os
from transformer_lens import HookedTransformer
from transformers import AutoConfig, AutoTokenizer



def has_enough_memory(device: torch.device, required_bytes: int) -> bool:
	"""
	Check if at least required_bytes are available on device.
	Returns True if OK, False otherwise.
	"""
	if device.type == "cuda":
		# torch.cuda.mem_get_info returns (free, total) in bytes
		free, _ = torch.cuda.mem_get_info(device.index)
		return free >= required_bytes
	else:
		vm = psutil.virtual_memory()
		return vm.available >= required_bytes

def download_model(model_name: str, cache_dir: str = "/app/models") -> None:
	"""Download a model to local cache directory."""
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

def load_model(model_name: str, required_bytes: int = 0, device='cpu', cache_dir = "/app/models") -> HookedTransformer:
	"""Load (and cache) a HookedTransformer, but first check memory."""
	
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

def load_tokenizer(model_name: str, config: dict):
	"""Load the tokenizer for a given model."""
	return AutoTokenizer.from_pretrained(config['huggingface_name'], trust_remote_code=True)

def load_model_config(model_name: str, config: dict) -> AutoConfig:
	"""Load the configuration of a model from Hugging Face."""
	return AutoConfig.from_pretrained(config['huggingface_name'], trust_remote_code=True)
