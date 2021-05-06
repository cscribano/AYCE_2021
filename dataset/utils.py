# -*- coding: utf-8 -*-
# ---------------------

import torch

def pad_sequence_mult(sequences, paddable_shapes=2, padding_value=0.0):
	# type: (list[torch.tensor], int, float) -> torch.tensor

	assert paddable_shapes <= 2, "Only up to 2 dimensional padding is supported yet"

	max_size = sequences[0].size()
	trailing_dims = max_size[paddable_shapes:]
	max_lens = [max([s.size(i) for s in sequences]) for i in range(paddable_shapes)]
	out_dims = (len(sequences), *max_lens) + trailing_dims

	out_tensor = sequences[0].new_full(out_dims, padding_value)
	for i, tensor in enumerate(sequences):
		lengths = [tensor.size(i) for i in range(paddable_shapes)]
		# use index notation to prevent duplicate references to the tensor
		if paddable_shapes == 2:
			out_tensor[i, :lengths[0], :lengths[1], ...] = tensor
		else:
			out_tensor[i, :lengths[0], ...] = tensor

	return out_tensor
