# -*- coding: utf-8 -*-
# ---------------------

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from models.attention import TransformerEncoder, len_to_mask
from models.misc import FrozenBatchNorm2d
from models.base_models import BaseModel
from dataset import Extended_Dataset, collate_fn_padd

from torchvision.models.utils import load_state_dict_from_url
from torchvision.models import resnet18, resnet34

models = {
	'resnet18': (resnet18, {'pretrained': False, 'num_classes': 256, 'norm_layer': FrozenBatchNorm2d},
				 'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
	'resnet34': (resnet34, {'pretrained': False, 'num_classes': 256, 'norm_layer': FrozenBatchNorm2d},
				 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'),
}

class TripleCosineSimilarity(nn.Module):

	def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
		super(TripleCosineSimilarity, self).__init__()
		self.dim = dim
		self.eps = eps

	def forward(self, x1, x2):
		d = 1 - F.cosine_similarity(x1.unsqueeze(1), x2, dim=-1, eps=self.eps)
		return torch.mean(d, dim=-1)

class TripleEuclidean(nn.Module):

	def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
		super(TripleEuclidean, self).__init__()
		self.dim = dim
		self.eps = eps

	def forward(self, x1, x2):
		d = torch.linalg.norm(x1.unsqueeze(1)-x2, dim=-1)  # + self.eps
		return torch.mean(d, dim=-1)


dist_dict = {
	'cosine': TripleCosineSimilarity,
	'euclid': TripleEuclidean
}

class ExtendedLiteModel(BaseModel):

	def __init__(self, cnf):
		# type: (Conf) -> None

		super().__init__()
		self.cnf = cnf

		dist_str = cnf.model_opts.get("DISTANCE", "cosine")
		self.distance = dist_dict[dist_str]

		# Dataset
		self.train_set = (Extended_Dataset, {'mode': "train", 'cache': False}, collate_fn_padd)
		self.val_set = (Extended_Dataset, {'mode': "val", 'cache': False})

		self.compute_nl = cnf.model_opts.get("COMPUTE_NL_EMBS", True)
		self.seq_pos_encoding = cnf.model_opts.get("SEQ_POS_ENCODE", True)
		print(f"Loaded Model on device {self.cnf.device}")

		# EGO_CROP CNN
		self.use_ego_crop = self.cnf.model_opts.get("USE_EGO_CROP", False)

		if self.use_ego_crop:
			ego_backbone = self.cnf.model_opts.get('BACKBONE', 'resnet18')
			model, args, ck_url = models[ego_backbone]

			self.ego_cnn = model(**args)
			state_dict = load_state_dict_from_url(ck_url, progress=True)

			# remove fc weights
			del state_dict["fc.weight"]
			del state_dict["fc.bias"]

			self.ego_cnn.load_state_dict(state_dict, strict=False)

		# Loss
		# self.criterion = nn.TripletMarginLoss(margin=1.0)
		margin = self.cnf.train_opts.get("MARGIN", 1.0)  # type: float
		self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=self.distance(), margin=margin)

		self.bb_linear = nn.Linear(261, 256)

		# Object sequence BB encoder
		self.spatial_encoder = TransformerEncoder(256, 6, dropout=0.1, pe_dropout=0.0)

		# Frame sequence transformer Encoder
		self.temporal_encoder = TransformerEncoder(256, 6, dropout=0.1, pe_dropout=0.0)

		# NLP
		if self.compute_nl:
			self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
			bert_ck = os.path.join(os.path.dirname(__file__), '../nlp/bert_ft_experimental')
			self.bert_model = BertModel.from_pretrained(bert_ck)

		self.lang_fc = torch.nn.Linear(768, cnf.model_opts['OUT_DIM'])

	def forward(self, bbs, seq, frames_len, bbs_len, obj_len,
				bb_ind=None, nl_embeddings=None, ego_crops=None, **kwargs):
		"""
		:param bbs: Padded sequences of all the M BBs: (BS, M, O, 5) where O==1
			if only the tracking BB is used
		:param seq: Sequence of 3 NL descriptions (if compute_nl == True)
		:param frames_len: Pre computed embeddings of the NL description (if compute_nl == False)
		:param bbs_len: (BS,) number of non-padded elements for each sequence in batch
		:param obj_len: (BS, N<80) Number of non-padding object-embeddings for each timestep
		:param bb_ind: (Bs, 80) Indices of the N sampled timesteps for each batch elements
		:param nl_embeddings: (VS, 2, 3, 768) pre-computed Bert embeddings
		:param ego_crops: (BS, N, 3, W, H) cropped and resized images of the tracked vehicle
		:param kwargs:
		:return:
		"""

		# Frame embeddings: (BS, N,  3, H, W) ->  (BS, N,  256)
		cnn_embs = pack_padded_sequence(ego_crops, bbs_len, batch_first=True, enforce_sorted=False)
		cnn_embs = PackedSequence(self.ego_cnn(cnn_embs.data), cnn_embs.batch_sizes,
								  cnn_embs.sorted_indices, cnn_embs.unsorted_indices)
		cnn_embs, _ = pad_packed_sequence(cnn_embs, True) # (BS, N, 256)

		bbs[:, :, 0, 5:] = cnn_embs  # Replace the fake ego embedding with the computed one

		bbs_emb = self.bb_linear(bbs)  # (bs, M, o, 1029) -> (bs, M, o, 1024)

		# Spatial encoding
		obj_mask = len_to_mask(obj_len).to(self.cnf.device)
		obj_emb = self.spatial_encoder(bbs_emb, mask=obj_mask)
		obj_emb = obj_emb.mean(dim=-2)  # (bs, M, o, 1032) -> (bs, M, 1032)

		# Temporal encoding
		bb_mask = len_to_mask(bbs_len).to(self.cnf.device)
		frame_emb = self.temporal_encoder(obj_emb, mask=bb_mask, pos=bb_ind)
		frame_emb = frame_emb.mean(dim=-2)  # (bs, M, 256)

		if nl_embeddings is None and self.compute_nl:
			pos, neg = seq
			pos_embeddings = self.compute_nl_embeddings(pos)
			neg_embeddings = self.compute_nl_embeddings(neg)

			nl_embeddings = torch.stack([pos_embeddings, neg_embeddings], dim=1)  # (BS, 2, 3, 256)

		# Naive method of combining 3 NL embeddings
		nl_embeddings = self.lang_fc(nl_embeddings)  # (BS, 2, 768) -> (BS, 2, emb_dim)

		return frame_emb, nl_embeddings

	def compute_nl_embeddings(self, seq):
		tokens = self.tokenizer(seq, padding='longest', return_tensors='pt')
		mask = tokens['attention_mask'].to(self.cnf.device)
		bert_out = self.bert_model(tokens['input_ids'].to(self.cnf.device),
								   attention_mask=mask).last_hidden_state

		# (BS*3, K, 768) -> (BS,3,K,768)
		lang_embeds = bert_out.view(3, -1, bert_out.shape[-2], bert_out.shape[-1]).transpose(0, 1)
		lang_embeds = torch.mean(lang_embeds, dim=2)

		return lang_embeds

	def train_loss(self, x):
		# type: (tuple) -> torch.Tensor

		_, images_len, bbs, bbs_len, pos, neg, im_indices, bb_indices, obj_len, ego_crops, anch_id = x
		if not self.seq_pos_encoding:
			bb_indices = None, None

		bbs = bbs.to(self.cnf.device)
		if self.use_ego_crop:
			ego_crops = ego_crops.to(self.cnf.device)

		if not self.compute_nl:
			nl_embs = torch.stack([pos, neg], dim=1).to(self.cnf.device)
			seq_out, nl_out = self.forward(bbs, None, images_len, bbs_len, obj_len,bb_indices,
										   nl_embeddings=nl_embs, ego_crops=ego_crops)
		else:
			seq_out, nl_out = self.forward(bbs, (pos, neg), images_len, bbs_len, obj_len,
										   bb_indices, nl_embeddings=None, ego_crops=ego_crops)
		loss = self.criterion(seq_out, nl_out[:, 0], nl_out[:, 1])

		return loss

	def val_loss(self, x):
		# type: (tuple) -> torch.Tensor

		return self.train_loss(x)
