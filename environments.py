def generate_binary_sequence(block_size, iterations):
	"""
	Generate a sequence of zeros and ones in alternating blocks.
	
	Parameters:
	- block_size (int): Number of zeros or ones in each block.
	- iterations (int): Number of alternating blocks of zeros and ones.
	
	Returns:
	- sequence (list): Generated sequence of zeros and ones.
	"""
	sequence = []
	for _ in range(iterations):
		sequence.extend([0] * block_size)
		sequence.extend([1] * block_size)
	return sequence