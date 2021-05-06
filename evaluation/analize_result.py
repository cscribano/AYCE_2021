# -*- coding: utf-8 -*-
# ---------------------

import os
import json

if __name__ == '__main__':
	# TODO: refactor
	ds_root = '/home/carmelo/DATASETS/AIC21_Track5_NL_Retrieval'

	# Load data
	tracks_root = os.path.join(ds_root, 'data/train-tracks.json')
	with open(tracks_root, "r") as f:
		tracks = json.load(f)
		f.close()

	exp_name = input('>> experiment name: ')
	with open(f"../log/AIC_2021_T5/{exp_name}/val_result.json") as f:
		results = json.load(f)

	output = {}
	for k in (results.keys()):
		expected = tracks[k]["nl"]
		predicted = tracks[results[k][0]]["nl"]

		output[k] = [expected, predicted]

	# write
	with open(f"../log/AIC_2021_T5/{exp_name}/val_analysis.json", "w") as f:
		json.dump(output, f, indent=4)
