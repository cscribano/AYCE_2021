# -*- coding: utf-8 -*-
# ---------------------

import os
import click
import json
from tqdm import tqdm

import threading
import multiprocessing as mp

from PIL import Image

@click.command()
@click.option('--ds_root', type=click.Path(exists=True),
			  default='/home/carmelo/DATASETS/AIC21_Track5_NL_Retrieval')
@click.option('--mode', type=click.Choice(['train', 'test'], case_sensitive=False), default='test')
@click.option('--workers', type=click.INT, default=1)
def main(ds_root, mode, workers=1):
	# type: (str, str, int) -> None
	"""
	Load the dataset frames and save the crop of the tracked vehicle for each frame,
	this is required to make the training I/O efficient.

	:param ds_root: Path to Cityflow-NL root
	:param mode: "train" or "test", select the data split to process
	:param workers: Split the work across multiple threads
	:return: None
	"""
	threads = []
	print(f"Using {workers} threads")
	for c in range(workers):
		new_thread = threading.Thread(target=extract_ego_frames, args=(ds_root, mode, workers, c))
		threads.append(new_thread)
		new_thread.start()

def extract_ego_frames(ds_root, mode, ws=1, rank=1):

	# Load train json
	tracks_root = os.path.join(ds_root, f'data/{mode}-tracks.json')

	with open(tracks_root, "r") as f:
		tracks = json.load(f)

	keys = list(tracks.keys())

	for key_idx, k in tqdm(enumerate(keys),  total=len(keys)):

		if (key_idx % ws) != rank:
			continue

		bboxes = tracks[k]["boxes"]
		frames = tracks[k]["frames"]

		# frame [1], detected_boxes[N], features[N
		for index, (box, frame) in enumerate(zip(bboxes, frames)):

			frame_abspath = os.path.join(ds_root, frame)
			image = Image.open(frame_abspath)

			# crop
			image = image.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))

			# create output folder structure
			outfolder = os.path.join(ds_root, f"egocrop_{mode}", k)

			if not os.path.isdir(outfolder):
				os.makedirs(outfolder)

			# save
			out_file = f"{index}.jpg"
			outfile = os.path.join(outfolder, out_file)

			image.save(outfile)

if __name__ == '__main__':
	main()
