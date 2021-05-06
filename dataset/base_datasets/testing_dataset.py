import os
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from abc import ABCMeta
from abc import abstractmethod

from conf import Conf

# Do not change this code this code to avoid reproducibility issues
# This is the dataset used to produce the output for submission

class TestDataset(Dataset, metaclass=ABCMeta):
    """
    Abstract dataset for standard test dataset,
    extend this class to use the standard testing code.
    """

    @abstractmethod
    def __init__(self, tf):
        # type: (transforms) -> None
        ...

    @abstractmethod
    def __getitem__(self, item):
        """
        Warning: the Dataloaders derivered from TestDataset must return the test images
        already pre-processed for the model to test
        """
        ...

    def __len__(self):
        return 0

    def get_query(self, query_ind):
        # type: (int) -> (str, list)
        """
        This method should return: query-uuid, [list of 3 nlp descriptions]
        """
        pass

    def num_queries(self):
        # type: () -> int
        return 0


# This template dataset return images, bb and nlp description with minimal pre processing
# Inherit from this class and override __getitem__() to implement your datastet

class CFNL_TestDataset(TestDataset):

    def __init__(self, cnf=None, tf=None):
        # type: (Conf, transforms) -> None
        """
        :param cnf: Configuration object
        :param tf: torchvision transform, if None only ToTensor() is applied
        """

        super().__init__(tf)

        self.cnf = cnf
        self.transforms = tf
        if tf is None:
            self.transforms = transforms.ToTensor()

        """
        test-tracks: a dictionary like for training but without "nl" queries
        test-queries: a dictionary of:
            <query-uuid>: <list of 3 nlp description of the same track>
        """
        # Load test tracks
        tracks_root = os.path.join(cnf.data_root, 'data/test-tracks.json')
        with open(tracks_root, "r") as f:
            tracks = json.load(f)

        self.tracks = tracks
        self.track_uuids = list(tracks.keys())

        # load test queries
        queries_root = os.path.join(cnf.data_root, 'data/test-queries.json')
        with open(queries_root, "r") as f:
            queries = json.load(f)

        self.queries = queries
        self.queries_uuid = list(queries.keys())

    def get_query(self, query_ind):
        # type: (int) -> (str, list)
        """
        :param query_ind:
        :return: (query_uuid, list of 3 nl descriptions)
        """

        key = self.queries_uuid[query_ind]
        return key, self.queries[key]

    def num_queries(self):
        return len(self.queries_uuid)

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
    ds = CFNL_TestDataset(cnf)

    nq = ds.num_queries
    q = ds.get_query(10)
    print(nq, q)

    a, b, c = ds[0]
    print(a, b.shape, c.shape)

