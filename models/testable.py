# -*- coding: utf-8 -*-
# ---------------------

import random
import torch

from models.base_models import BaseModel, TestableModel


class ExtendedTestable(TestableModel):

	def __init__(self, cnf, backbone=None):
		# type: (Conf, BaseModel) -> None

		super().__init__()
		self.backbone = backbone

		self.cnf = cnf
		self.img_seq_len = self.cnf.model_opts["MAX_SEQ_LEN"]
		self.bb_seq_len = 80

		self.compute_nl = backbone.compute_nl
		self.seq_pos_encoding = backbone.seq_pos_encoding

	def test_step(self, images, bbs, nl, egocrop, **kwargs):

		# sanity checks
		if not (len(images.shape) == 5 and images.shape[0] == 1):
			raise Exception(f"seq tensor of shape {images.shape} is not valid, only batch size 1 is supported")

		# images: (1, N, 3, w, h)
		# bbs: (1, N, O, 5)
		# nl: (3,) or (3, emb_dim) (if using precomputed)

		seq_projections = []
		nl_projection = None

		# Use Offline computed nl embeddings
		if not self.compute_nl:
			nl_embeddings = nl.unsqueeze(0).to(self.cnf.device)  # type: torch.tensor
		else:
			nl_embeddings = self.backbone.compute_nl_embeddings(nl)

		nl_embeddings = nl_embeddings.unsqueeze(1)  # (BS, 1, 3, 768)

		averaged_samples = False

		for _ in range(10):

			im_inds = [i for i in range(images.shape[1])]
			bb_inds = im_inds.copy()
			n_samp = len(im_inds)

			assert (len(im_inds) == bbs.shape[1])  # better safe than sorry!

			if bbs.shape[1] > self.bb_seq_len:
				im_inds = sorted(random.sample(im_inds, min(n_samp, self.img_seq_len)))
				bb_inds = sorted(random.sample(bb_inds, min(n_samp, 80)))

				averaged_samples = True

			im = images[:, im_inds, ...]
			bb = bbs[:, bb_inds, ...]

			images_len = torch.tensor([len(im_inds), ])
			bbs_len = torch.tensor([len(bb_inds), ])
			im, bb = im.to(self.cnf.device), bb.to(self.cnf.device)

			# detection bbs lenght
			# Actual number of detected objects in each frame
			obj_len = torch.sum(bb[:, :, :, 0] != 0, dim=-1).cpu()

			if not bb.shape[-2] == obj_len.max():
				bb = bb[:, :, :obj_len.max(), :]  # Remove excess of padding

			# tracked vehicle crop
			ec = egocrop[:, bb_inds, ...]
			ec = ec.to(self.cnf.device)

			if self.seq_pos_encoding:
				bb_inds = torch.tensor(bb_inds)
			else:
				bb_inds = None, None

			seq_out, nl_out = self.backbone.forward(frames=im, bbs=bb, seq=None, frames_len=images_len,
			                                        bbs_len=bbs_len, bb_ind=bb_inds,
			                                        nl_embeddings=nl_embeddings, obj_len=obj_len, ego_crops=ec)

			nl_projection = nl_out.cpu()
			seq_projections.append(seq_out.cpu())

			if not averaged_samples:
				break

		# Stack
		seq_projections = torch.stack(seq_projections)

		return seq_projections[:, 0, :], nl_projection[:, 0]
