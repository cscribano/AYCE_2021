# -*- coding: utf-8 -*-
# ---------------------

import random
import json
import os

# fixed seed for reproducibility
random.seed(1234)

def make_splits(ds_root, val_ratio=0.1):
    # type: (str, float) -> None
    """
    :param ds_root: Cityflow-NL dataset root (absolute path)
    :param val_ratio: portion of the dataset to reserve for validation
    :return: None
    """

    # Load train json
    tracks_root = os.path.join(ds_root, 'data/train-tracks.json')
    with open(tracks_root, "r") as f:
        tracks = json.load(f)

    keys = list(tracks.keys())

    # Reserve num_tracks*val_ratio sequences
    num_val = int(len(keys) * val_ratio)
    random.shuffle(keys)  # just in case

    val = keys[:num_val]
    train = keys[num_val:]

    # Save list of keys to files
    with open('validation.txt', 'w') as f:
        for item in val:
            f.write("%s\n" % item)
        f.close()

    # validation gt
    gt = {item: item for item in val}
    with open('validation-gt.json', "w") as f:
        json.dump(gt, f)

    with open('train.txt', 'w') as f:
        for item in train:
            f.write("%s\n" % item)
        f.close()

    print(f"Train Samples {len(val)} - Validation Samples {len(train)}")

if __name__ == '__main__':
    make_splits('/home/carmelo/DATASETS/AIC21_Track5_NL_Retrieval')
