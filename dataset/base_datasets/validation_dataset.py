import os
import json

import torch
from PIL import Image
from torchvision import transforms

from dataset.base_datasets.testing_dataset import TestDataset
from conf import Conf

# This is used to compute VALIDATION metrics for the model
# The VALIDATION dataset is a subset of the original train dataset
# This is necessary since we dont' have GT of the actual test dataset

class CFNL_ValDataset(TestDataset):

    def __init__(self, cnf=None, tf=None):
        # type: (Conf, transforms) -> None
        """
        :param cnf: Configuration object
        :param tf: torchvision transform, if None only ToTensor() is applied
        """

        super().__init__(tf)

        self.cnf = cnf
        if tf is None:
            self.transforms = transforms.ToTensor()

        """
        test-tracks: a dictionary like for training but without "nl" queries
        test-queries: a dictionary of:
            <query-uuid>: <list of 3 nlp description of the same track>
        """
        # Load train split json
        tracks_root = os.path.join(cnf.data_root, 'data/train-tracks.json')
        with open(tracks_root, "r") as f:
            self.tracks = json.load(f)

        # TODO: pass as argument
        val_file = os.path.join(os.path.dirname(__file__), "../../data/validation.txt")

        # read split uuids
        with open(val_file) as f:
            self.uuids = [line.rstrip() for line in f]

    def get_query(self, query_ind):
        # type: (int) -> (str, list)
        """
        :param query_ind:
        :return: (query_uuid, list of 3 nl descriptions)
        """

        key = self.uuids[query_ind]
        return key, self.tracks[key]["nl"]

    def num_queries(self):
        return len(self.uuids)

    def __len__(self):
        return len(self.uuids)

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
        uuid = self.uuids[item]

        # Load and stack all the images
        for index, frame in enumerate(self.tracks[uuid]["frames"]):
            # Read and stack all the frames without resize or crop
            frame_path = os.path.join(self.cnf.data_root, frame)
            image = Image.open(frame_path)
            image = self.transforms(image)  # (3, W, H)
            images.append(image)

        images = torch.stack(images, 0)  # (N, 3, W, H)
        boxes = torch.tensor(self.tracks[uuid]["boxes"])

        return uuid, images, boxes

if __name__ == '__main__':
    cnf = Conf(exp_name='default')
    ds = CFNL_ValDataset(cnf)

    nq = ds.num_queries()
    q = ds.get_query(10)
    print(nq, q)

    a, b, c = ds[0]
    print(a, b.shape, c.shape)
