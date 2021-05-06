# -*- coding: utf-8 -*-
# ---------------------

import os
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models import resnet18, resnet34

from models.attention import TransformerEncoder, TransformerDecoder, len_to_mask
from models.base_models import BaseModel

from dataset import Extended_Dataset, collate_fn_padd
from models.misc import FrozenBatchNorm2d, batch_pairwise_squared_distances

models = {
	'resnet18': (resnet18, {'pretrained': False, 'num_classes': 256, 'norm_layer': FrozenBatchNorm2d},
				 'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
	'resnet34': (resnet34, {'pretrained': False, 'num_classes': 256, 'norm_layer': FrozenBatchNorm2d},
				 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'),
}

class Combined_Distance(nn.Module):

	def __init__(self, reduce=True):
		super(Combined_Distance, self).__init__()
		self.reduce = reduce

	def forward(self, x1, x2, mode="min"):

		d = batch_pairwise_squared_distances(x1, x2).view(x1.shape[0], -1, 9)
		if mode == "min":
			d = d.min(dim=-1)[0]
		elif mode == "mean":
			d = d.mean(dim=-1)
		elif mode == "sort":
			d = torch.sort(d, dim=-1)[0][:,:, :5].mean(dim=-1)

		if self.reduce:
			return torch.mean(d)

		return d

class AYCE_Loss(nn.Module):

	def __init__(self):
		super(AYCE_Loss, self).__init__()
		self.dist_fn = Combined_Distance(reduce=False,)

	def forward(self, x1, p, n):

		dp = self.dist_fn(x1, p, mode='none')
		dn = self.dist_fn(x1, n.unsqueeze(1), mode='mean')

		l2 = dp.min(dim=-1)[0].mean() * 0.1
		dp = dp.mean(dim=-1)
		l1 = torch.max(torch.zeros_like(dp), torch.max((dp-dn) + 1.0, dim=-1)[0]).mean()

		loss = l1 + l2
		return loss

class ExtendedLiteNLModel(BaseModel):

	def __init__(self, cnf):
		# type: (Conf) -> None

		super().__init__()
		self.cnf = cnf

		self.distance = Combined_Distance

		# Dataset
		self.train_set = (Extended_Dataset, {'mode': "train", 'cache': False}, collate_fn_padd)
		self.val_set = (Extended_Dataset, {'mode': "val", 'cache': False})

		self.compute_nl = cnf.model_opts.get("COMPUTE_NL_EMBS", True)
		self.seq_pos_encoding = cnf.model_opts.get("SEQ_POS_ENCODE", True)
		print(f"Loaded Model on device {self.cnf.device}")

		ego_backbone = self.cnf.model_opts.get('BACKBONE', 'resnet18')
		model, args, ck_url = models[ego_backbone]

		self.ego_cnn = model(**args)
		state_dict = load_state_dict_from_url(ck_url, progress=True)

		# remove fc weights
		del state_dict["fc.weight"]
		del state_dict["fc.bias"]

		self.ego_cnn.load_state_dict(state_dict, strict=False)

		# Loss
		self.criterion = AYCE_Loss()

		self.bb_linear = nn.Linear(261, 256)

		# Object sequence BB encoder
		self.spatial_encoder = TransformerEncoder(256, 6, dropout=0.1, pe_dropout=0.0)

		# Frame sequence transformer Encoder
		self.temporal_encoder = TransformerEncoder(256, 6, dropout=0.1, pe_dropout=0.0)

		# Output Decoder
		nl_queries = torch.zeros((3, 256), dtype=torch.float32)
		self.register_buffer('nl_queries', nl_queries)

		self.out_decoder = TransformerDecoder(256, 6, 256, output_dim=None)

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

		bbs[:, :, 0, 5:] = cnn_embs  # Replace the placeholder ego embedding with the computed one

		bbs_emb = self.bb_linear(bbs)  # (bs, M, o, 261) -> (bs, M, o, 256)

		# Spatial encoding
		obj_mask = len_to_mask(obj_len).to(self.cnf.device)
		obj_emb = self.spatial_encoder(bbs_emb, mask=obj_mask)
		obj_emb = obj_emb.mean(dim=-2)  # (bs, M, o, 256) -> (bs, M, 256)

		# Temporal encoding
		bb_mask = len_to_mask(bbs_len).to(self.cnf.device)
		frame_emb = self.temporal_encoder(obj_emb, mask=bb_mask, pos=bb_ind)

		# Decoding of 3 separate embeddings
		# (bs, M, 256) -> (bs, 3, 256)
		out_embs = self.out_decoder(self.nl_queries.repeat(frame_emb.shape[0], 1, 1), frame_emb, encoder_mask=bb_mask)

		if nl_embeddings is None and self.compute_nl:
			pos, neg = seq
			pos_embeddings = self.compute_nl_embeddings(pos)
			neg_embeddings = self.compute_nl_embeddings(neg)

			nl_embeddings = torch.stack([pos_embeddings, neg_embeddings], dim=1)  # (BS, 2, 3, 256)

		# Apply linear layer to NL output
		nl_embeddings = self.lang_fc(nl_embeddings)  # (BS, 2, 768) -> (BS, 2, emb_dim)

		return out_embs, nl_embeddings

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
			bb_indices = None

		bbs = bbs.to(self.cnf.device)
		ego_crops = ego_crops.to(self.cnf.device)

		if not self.compute_nl:
			nl_embs = torch.stack([pos, neg], dim=1).to(self.cnf.device)
			seq_out, nl_out = self.forward(bbs, None, images_len, bbs_len, obj_len, bb_indices,
										   nl_embeddings=nl_embs, ego_crops=ego_crops)
		else:
			seq_out, nl_out = self.forward(bbs, (pos, neg), images_len, bbs_len, obj_len,
										   bb_indices, nl_embeddings=None, ego_crops=ego_crops)
		loss = self.criterion(seq_out, nl_out[:, 0], nl_out[:, 1])

		return loss

	def val_loss(self, x):
		# type: (tuple) -> torch.Tensor

		return self.train_loss(x)
