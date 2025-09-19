from itertools import islice

def batch_iterable(iterable, batch_size):
	"""Batch an iterable into chunks of a specified size.
	Args:
		iterable (iterable): The input iterable to be batched.
		batch_size (int): The size of each batch.
	Yields:
		list: A batch of elements from the iterable.
	"""
	it = iter(iterable)
	while True:
		chunk = list(islice(it, batch_size))
		if not chunk:
			break
		yield chunk