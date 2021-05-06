# -*- coding: utf-8 -*-
# ---------------------

import os
import json
import click
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from PIL import Image
import numpy as np
from tqdm import tqdm

from pytorch_detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor

# fixed seed for reproducibility
torch.manual_seed(1234)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@click.command()
@click.option('--ds_root', type=click.Path(exists=True),
			  default='/home/carmelo/DATASETS/AIC21_Track5_NL_Retrieval')
@click.option('--mode', type=click.Choice(['train', 'test'], case_sensitive=False), default='train')
@click.option('--multi_process', type=click.BOOL, default=False)
def main(ds_root, mode, multi_process):
	"""
	:param ds_root: Cityflow-NL dataset root
	:param mode: "train" or "test"
	:param multi_process: Parallelize the execution on multiple GPUs
	:return: None
	"""

	slurm = os.environ.get("SLURM_TASK_PID") is not None

	# todo: if rank==1....
	out_dir = os.path.join(ds_root, f"rcnn_embs_{mode}")
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	if multi_process:
		world_size = int(os.environ["SLURM_NPROCS"]) if slurm else torch.cuda.device_count()
	else:
		world_size = 1

	if world_size == 1:
		compute_rcnn_embs(ds_root, mode, debug=False)
	else:
		print(f"Runing in {world_size} parallel processes")
		if slurm:
			rank = int(os.environ["SLURM_PROCID"])
			compute_rcnn_embs_parallel(rank, world_size, ds_root, mode)
		else:
			mp.spawn(
				compute_rcnn_embs_parallel, args=(world_size, ds_root, mode),
				nprocs=world_size, join=True
			)


def compute_rcnn_embs_parallel(rank, world_size, ds_root, mode='train'):
	print(f">> Running on {world_size} processes")
	compute_rcnn_embs(ds_root, mode, world_size=world_size, rank=rank)

def compute_rcnn_embs(ds_root, mode='train', debug=False, world_size=1, rank=0):
	# type: (str, str, bool, int, int) -> None
	"""
	:param ds_root: Cityflow-NL dataset root (absolute path)
	:return:
	"""
	print(f"Started process {rank}")
	# Load BERT model
	gpu_id = rank % torch.cuda.device_count()
	model = fasterrcnn_resnet50_fpn(pretrained=True).to(f"cuda:{gpu_id}")
	model.eval()

	userful_classes = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]

	# Load train json
	if mode == 'train':
		tracks_root = os.path.join(ds_root, 'data/train-tracks.json')
	elif mode == 'test':
		tracks_root = os.path.join(ds_root, 'data/test-tracks.json')
	else:
		raise Exception(f"Only train and test are valid split/modes")

	with open(tracks_root, "r") as f:
		tracks = json.load(f)

	keys = list(tracks.keys())

	# Output
	out_dir = os.path.join(ds_root, f"rcnn_embs_{mode}")
	if os.path.isdir(out_dir):
		# Remove already computed keys
		prec_keys = [k.split('.')[0] for k in os.listdir(out_dir)]
		keys = [k for k in keys if k not in prec_keys]

	for key_idx, id in tqdm(enumerate(keys), total=len(keys)):

		if (key_idx % world_size) != rank:
			continue

		result = {
			"frames": [],
			"detected_boxes": [],
			"features": [],
			"ego_ind": []
		}

		frames = tracks[id]['frames']
		ego = tracks[id]['boxes']

		for frame_path, ego in zip(frames, ego):
			frame_abspath = os.path.join(ds_root, frame_path)
			frame_orig = Image.open(frame_abspath)
			frame = to_tensor(frame_orig).to(f"cuda:{gpu_id}")

			with torch.no_grad():
				# object detection
				predictions = model([frame, ])

			boxes = predictions[0]["boxes"].cpu()
			features = predictions[0]["features"].cpu()
			labels = predictions[0]["labels"].cpu()
			scores = predictions[0]["scores"].cpu()

			# Filter boxes based on class and confidence score
			labels_filter = [i for i, l in enumerate(labels) if l in userful_classes]
			scores_filter = [i for i, s in enumerate(scores) if s > 0.65]

			indices = [i for i in range(len(boxes)) if i in labels_filter and i in scores_filter]

			# determine index of the ego vehicle (if detected)
			ego_bb = np.array(ego)[np.newaxis, :]
			# (x,y,w,h) -> (x1,y1,x2,y2)
			ego_bb[:, 2], ego_bb[:, 3] = ego_bb[:, 2] + ego_bb[:, 0], ego_bb[:, 3] + ego_bb[:, 1]

			if len(labels) > 1:  # at least one box is detected
				ious = iou_of(boxes.numpy(), ego_bb)
				ego_ind = np.argmax(ious)

				if ious[ego_ind] < 0.2:
					ego_ind = -1
				else:
					if ego_ind not in indices:
						indices.append(ego_ind.item())
						ego_ind = len(indices) - 1  # the last element
					else:
						ego_ind = indices.index(ego_ind) # position relative to indices not labels
			else:
				ego_ind = -1

			# prepare result
			boxes[:, 2], boxes[:, 3] = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]  # (x1,y1,x2,y2) -> (x,y,w,h)
			boxes = torch.cat([labels[:, None], boxes, scores[:, None]], dim=-1)  # class, (x,y,w,h), score

			if len(indices) == 0:
				print(1)

			# filter
			boxes = boxes[indices]
			features = features[indices]

			result["frames"].append(frame_path)
			result["detected_boxes"].append(boxes)
			result["features"].append(features)
			result["ego_ind"].append(ego_ind)

		# save
		out_file = os.path.join(out_dir, f'{id}.pt')
		torch.save(result, out_file)

def area_of(left_top, right_bottom):
	hw = np.clip(right_bottom - left_top, 0.0, None)
	return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
	# type: (np.array, np.array, float) -> np.array

	overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
	overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

	overlap_area = area_of(overlap_left_top, overlap_right_bottom)
	area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
	area1 = area_of(boxes1[..., :2], boxes1[..., 2:])

	return overlap_area / (area0 + area1 - overlap_area + eps)


if __name__ == '__main__':
	main()

