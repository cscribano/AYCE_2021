# -*- coding: utf-8 -*-
# ---------------------

import torch
import torch.nn as nn
from math import sqrt

class MultiHeadAttention(nn.Module):
	def __init__(self, q_emb_dim, head_dim, n_heads, kv_emb_dim=None, dropout=0.0):
		# type: (int, int, int, int, bool) -> None
		"""
		:param emb_dim: input embeddings dim
		:param head_dim: Single head hidden-dim
		:param n_heads: number of MHSA heads
		:param dropout: 0 by default (no dropout applied)
		"""

		super().__init__()

		self.n_heads = n_heads
		self.head_dim = head_dim
		self.hid_dim = head_dim * n_heads

		# scale
		scale = torch.tensor([self.hid_dim], dtype=torch.float32).rsqrt()
		self.register_buffer('scale', scale)

		if kv_emb_dim is None:
			kv_emb_dim = q_emb_dim

		self.fc_q = nn.Linear(q_emb_dim, self.hid_dim)
		self.fc_k = nn.Linear(kv_emb_dim, self.hid_dim)
		self.fc_v = nn.Linear(kv_emb_dim, self.hid_dim)
		self.fc_o = nn.Linear(self.hid_dim, self.hid_dim)

		self.dropout = nn.Dropout(dropout)

		self._reset_parameters()

	def _reset_parameters(self):
		# type: () -> None
		"""
		https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
		:return:
		"""
		# scaled initialization
		nn.init.xavier_uniform_(self.fc_k.weight, gain=1 / sqrt(2))
		nn.init.xavier_uniform_(self.fc_v.weight, gain=1 / sqrt(2))
		nn.init.xavier_uniform_(self.fc_q.weight, gain=1 / sqrt(2))
		nn.init.xavier_uniform_(self.fc_o.weight)
		nn.init.constant_(self.fc_o.bias, 0.0)
		nn.init.constant_(self.fc_k.bias, 0.0)
		nn.init.constant_(self.fc_v.bias, 0.0)

	def forward(self, query, key, value, mask=None):
		# type: (torch.tensor, torch.tensor, torch.tensor, torch.tensor) -> torch.tensor
		"""
		:param query: (bs, q, emb_dim)
		:param key: (bs, k, emb_dim)
		:param value: (bs, k, emb_dim)
		:param mask: (bs, 1, 1, k)
		:return: (bs, q, head_dim*n_heads)
		"""

		# batch_size = query.shape[0]

		# (bs, q/k/v_len, emb_dim) -> (bs, q/k/v_len, hid_dim)
		q = self.fc_q(query)
		k = self.fc_k(key)
		v = self.fc_v(value)

		"""
		# (bs, q/v_len, hid_dim) -> (bs, h, q/v_len, head_dim)
		q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		# (1, 36, 9, 256) -> (1, 36, 9, h, h_dim)
		# (bs, k_len, head_dim) -> (bs, h, head_dim, k_len)
		k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
		"""

		q = q.view(*q.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)
		v = v.view(*v.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)
		k = k.view(*k.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3).transpose(-1, -2)

		# E = (Q^T)*K -> (bs, h, q, head_dim) * (bs, h, head_dim, k) = (bs. h, q, k)
		energy = torch.matmul(q, k) / self.scale

		# Mask to avoid attending padding or future elements (in decoder attn)
		if mask is not None:
			mask = mask.unsqueeze(-2)
			energy = energy.masked_fill(mask == 0, -1e10)

		attention = torch.softmax(energy, dim=-1)

		# E*V -> (bs. h, q, k) * (bs, h, v, head_dim) -> (bs, h, q, head_dim)
		# (note: k=v always!)
		x = torch.matmul(self.dropout(attention), v)

		# (bs, h, q, head_dim) -> (bs, q, h, head_dim) -> (bs, q, hid_dim)
		x = x.transpose(-2, -3).contiguous()
		x = x.view(*x.shape[:-2], self.hid_dim)

		# (bs, q, hid_dim) -> (bs, q, hid_dim)
		x = self.fc_o(x)

		return x


if __name__ == '__main__':

	model = MultiHeadAttention(512, 512//8, 8)

	q = torch.rand((10, 14, 512), dtype=torch.float32)
	kv = torch.rand((10, 21, 512), dtype=torch.float32)
	out = model(q, kv, kv)

	print(out.shape)
