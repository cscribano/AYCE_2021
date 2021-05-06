# -*- coding: utf-8 -*-
# ---------------------

import torch
import numpy as np

def batch_pairwise_squared_distances(x, y):
	'''
	Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
	Input: x is a bxNxd matrix y is an optional bxMxd matirx
	Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
	i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2
	'''
	x_norm = (x**2).sum(-1).unsqueeze(-1)
	y_t = torch.transpose(y, -1, -2).contiguous()
	y_norm = (y**2).sum(-1).unsqueeze(-2)
	dist = x_norm + y_norm - 2.0 * torch.matmul(x, y_t)
	dist[dist != dist] = 0  # replace nan values with 0
	dist = torch.sqrt(dist)

	return torch.clamp(dist, 0.0, np.inf)

class FrozenBatchNorm2d(torch.nn.Module):
	# https://github.com/facebookresearch/detr/blob/master/models/backbone.py
	"""
	BatchNorm2d where the batch statistics and the affine parameters are fixed.
	Copy-paste from torchvision.misc.ops with added eps before rqsrt,
	without which any other models than torchvision.models.resnet[18,34,50,101]
	produce nans.
	"""

	def __init__(self, n):
		super(FrozenBatchNorm2d, self).__init__()
		self.register_buffer("weight", torch.ones(n))
		self.register_buffer("bias", torch.zeros(n))
		self.register_buffer("running_mean", torch.zeros(n))
		self.register_buffer("running_var", torch.ones(n))

	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
							  missing_keys, unexpected_keys, error_msgs):
		num_batches_tracked_key = prefix + 'num_batches_tracked'
		if num_batches_tracked_key in state_dict:
			del state_dict[num_batches_tracked_key]

		super(FrozenBatchNorm2d, self)._load_from_state_dict(
			state_dict, prefix, local_metadata, strict,
			missing_keys, unexpected_keys, error_msgs)

	def forward(self, x):
		# move reshapes to the beginning
		# to make it fuser-friendly
		w = self.weight.reshape(1, -1, 1, 1)
		b = self.bias.reshape(1, -1, 1, 1)
		rv = self.running_var.reshape(1, -1, 1, 1)
		rm = self.running_mean.reshape(1, -1, 1, 1)
		eps = 1e-5
		scale = w * (rv + eps).rsqrt()
		bias = b - rm * scale
		return x * scale + bias
