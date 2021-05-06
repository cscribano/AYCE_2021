# -*- coding: utf-8 -*-
# ---------------------

import os
import platform

import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

from dataset.turbojpeg import TurboJPEG
import json

from dataset.base_datasets import TestDataset
from conf import Conf


class Extended_TestValDataset(TestDataset):

	def __init__(self, cnf=None, mode="val"):
		# type: (Conf, str) -> None
		"""
		:param cnf: Configuration object
		:param mode: "val" or "test"
		"""

		self.cnf = cnf

		assert mode in ["val", "test"], f"Invalid mode {mode}"

		self.input_size = (cnf.model_opts['INPUT_HEIGHT'], cnf.model_opts['INPUT_WIDTH'])
		tf = self.transforms = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize(self.input_size),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # RGB
		])

		crop_width = cnf.model_opts.get('CROP_HEIGHT', 80)
		crop_height = cnf.model_opts.get('CROP_WIDTH', 80)

		self.crop_transforms = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize((crop_height, crop_width)),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

		super().__init__(tf)

		# turbojpeg
		self.turbo = os.path.join(cnf.project_root,
								  f'dataset/turbojpeg/{platform.processor()}/libturbojpeg.so')
		self.jpeg = None  # TurboJPEG(turbo)

		data_files = {"val": "train", "test": "test"}
		tracks_root = os.path.join(cnf.data_root, f'data/{data_files[mode]}-tracks.json')

		# Load test tracks
		with open(tracks_root, "r") as f:
			tracks = json.load(f)

		self.tracks = tracks

		if mode == "val":
			val_file = os.path.join(os.path.dirname(__file__), "../data/validation.txt")

			# read split uuids
			with open(val_file) as f:
				self.track_uuids = [line.rstrip() for line in f]

			self.queries = {u: self.tracks[u]["nl"] for u in self.track_uuids}
			self.queries_uuid = self.track_uuids.copy()

		else:
			self.track_uuids = list(self.tracks.keys())

			# load test queries
			queries_root = os.path.join(cnf.data_root, 'data/test-queries.json')
			with open(queries_root, "r") as f:
				queries = json.load(f)

			self.queries = queries
			self.queries_uuid = list(queries.keys())

		# Precomputed nl embeddings
		self.prec_nl_embs = None
		if not self.cnf.model_opts["COMPUTE_NL_EMBS"]:
			embd_dir = os.path.join(self.cnf.data_root, f'precomp_nl_{data_files[mode]}.pt')

			self.prec_nl_embs = torch.load(embd_dir)

		self.used_classes = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]  # COCO classes

		self.augmented_root = os.path.join(cnf.data_root, f"rcnn_embs_{data_files[mode]}")
		self.crop_root = os.path.join(self.cnf.data_root, f"egocrop_{data_files[mode]}")


	def num_queries(self):
		return len(self.queries_uuid)

	def get_query(self, query_ind):
		# type: (int) -> (str, list)
		"""
		:param query_ind:
		:return: (query_uuid, list of 3 nl descriptions/pre computed embeddings)
		"""
		key = self.queries_uuid[query_ind]

		if self.prec_nl_embs is None:
			return key, self.queries[key]
		else:
			return key, self.prec_nl_embs[key]  # (1, 3, 768)

	def __len__(self):
		return len(self.track_uuids)

	def __getitem__(self, item):
		# type: (int) -> (str, torch.tensor, torch.tensor)
		"""
		:param item: element index
		:return: All (uuid, images, boxes) in the tracking sequence
			> uuid: string
			> images: (seq_len, 3, w, h)
			> boxes: (seq_len, 4)
		"""

		images = []
		uuid = self.track_uuids[item]

		# turbojpeg decoder
		if self.jpeg is None:
			self.jpeg = TurboJPEG(self.turbo)

		# Image's original width and height
		orig_w = None
		orig_h = None

		# Load and stack all the images
		for index, frame in enumerate(self.tracks[uuid]["frames"]):
			# Read and stack all the frames without resize or crop
			frame_path = os.path.join(self.cnf.data_root, frame)

			with open(frame_path, 'rb') as f:
				im = self.jpeg.decode(f.read())

			# Store actual image's width and height
			if orig_w is None:
				orig_w, orig_h = im.shape[1], im.shape[0]

			# apply transforms
			im = im[..., ::-1].copy()  # BGR -> RGB
			image = self.transforms(im)  # (3, W, H)
			images.append(image)

		images = torch.stack(images, 0)  # (N, 3, W, H)

		boxes = torch.tensor(self.tracks[uuid]["boxes"], dtype=torch.float32)  # (len, 4)

		class_idx = torch.ones(len(boxes)) * 1  # Tracked vehicle has class -1
		boxes = torch.tensor(self.tracks[uuid]["boxes"], dtype=torch.float32)  # (len, 4)
		boxes = torch.cat([class_idx.unsqueeze(-1), boxes], dim=1).unsqueeze(1)  # (len, 1, 5)

		# Retrieve object-embeddings
		augmented_file = os.path.join(self.augmented_root, f"{uuid}.pt")
		augmented_track = torch.load(augmented_file)

		dets = augmented_track["detected_boxes"]
		ego_id = augmented_track["ego_ind"]

		# sanity check
		det_frames = augmented_track["frames"]
		assert det_frames == self.tracks[uuid]["frames"]

		# remove ego detection box, also remove low confidence boxes and boxes from useless classes
		dets_idx = [[di for di, det in enumerate(frame_dets) if det[0] in self.used_classes and
					 det[-1] > 0.85 and di != ego_id[frame_idx]] for frame_idx, frame_dets in enumerate(dets)]

		dets = [d[di][:, :5] for d, di in zip(dets, dets_idx)]  # keep only filtered detections

		# Add RCNN visual fetures
		rcnn_features = augmented_track["features"]
		rcnn_features = [f[i] for f, i in zip(rcnn_features, dets_idx)]

		# embedding vector is: bb[5] + rcnn_feature[256]
		dets = [torch.cat([d, f], dim=1) for d, f in zip(dets, rcnn_features)]

		dets, boxes_len = bb_pad(dets)

		# Add fake R-CNN embedding for ego vehicle
		ego_pad = torch.zeros(len(boxes), 1, 256)
		boxes = torch.cat([boxes, ego_pad], dim=-1)  # (N, 1, 5) -> (N, 1, 5+256)

		if len(dets.shape) == 3:  # Otherwise means there are NO objects in any of the frames!
			# Add tracking BB's
			dets[:, :, 0] += 1  # preserve the 1 class for ego, for backward compatibility...
			boxes = torch.cat([boxes, dets], dim=1)

		boxes_len += 1

		# Normalize bb's wrt image size
		boxes[:, :, 2], boxes[:, :, 4] = boxes[:, :, 2] / orig_h, boxes[:, :, 4] / orig_h
		boxes[:, :, 1], boxes[:, :, 3] = boxes[:, :, 1] / orig_w, boxes[:, :, 3] / orig_w

		# Return crops of the ego-vehicle for all selected (<80) tracking boxes
		ego_crops = []

		for i in range(len(boxes)):
			crop_file = os.path.join(self.crop_root, uuid, f"{i}.jpg")
			with open(crop_file, 'rb') as f:
				crop = self.jpeg.decode(f.read())

			crop = crop[..., ::-1].copy()  # bgr to rgb
			crop = self.crop_transforms(crop)

			ego_crops.append(crop)

		# stack
		ego_crops_tensor = torch.stack(ego_crops)  # (N_BB, 3, W, H)

		return uuid, images, boxes, ego_crops_tensor


def bb_pad(seq):
	l = torch.tensor([len(s) for s in seq])
	seq = pad_sequence([s for s in seq], batch_first=True)

	# Use -1 as "class" for padded elements, this will become 0
	# when we add 1 to preserve the 1 index for ego vehicle
	seq[:, :, 0] += torch.where(seq[:, :, 0] == 0, -1, 0)

	return seq, l

