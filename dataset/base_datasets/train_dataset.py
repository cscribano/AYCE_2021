import os
import random
import json

import torch
#from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import platform
#from io import BytesIO

from dataset.turbojpeg import TurboJPEG
from conf import Conf

class CFNL_TrainDataset(Dataset):

    def __init__(self, cnf=None, mode="train", tf=None, cache=False, prefetch=False):
        # type: (Conf, str, transforms, bool, bool) -> None
        """
        :param cnf:  Configuration object
        :param mode:
            > "train": the uuids from ../../data/train.txt are used
            > "val": the uuids from ../../data/validation.txt are used
            > "train_all": all the uuids from train-tracks.json are used
        """

        if mode not in ["train", "val", "train_all"]:
            raise Exception(f"{mode} is an invalid dataset mode")

        self.cnf = cnf
        self.cache = cache
        self.prefetch = prefetch
        self.mode = mode

        # turbojpeg
        self.turbo = os.path.join(cnf.project_root,
                             f'dataset/turbojpeg/{platform.processor()}/libturbojpeg.so')
        self.jpeg = None # TurboJPEG(turbo)

        if cache:
            print("DATASET CACHING ENABLED!!")

        if tf is None:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = tf

        # Load train split json
        tracks_root = os.path.join(cnf.data_root, 'data/train-tracks.json')
        with open(tracks_root, "r") as f:
            tracks = json.load(f)
            f.close()

        self.tracks = tracks

        """
        tracks is a dictionary: 
        {"track-0-uuid": 
            {frames: <list-of-frames>,
            boxes: <list of bb's>, 
            nl: <3 NLPs>},
        "track-1-uuid":
            {...
            ...}
        """

        if self.mode == "train_all":
            print("[DATASET]: Using the full dataset for train!")
            self.uuids = list(tracks.keys())
        else:
            mode_files = {"train": "../../data/train.txt", "val": "../../data/validation.txt"}
            split_file = os.path.join(os.path.dirname(__file__), mode_files[mode])

            # read split uuids
            with open(split_file) as f:
                self.uuids = [line.rstrip() for line in f]
                f.close()

        # Cache
        # uuid: [img1, ..., imgN] (jpeg encoded)
        self.img_buffer = {u: [None for i in range(len(self.tracks[u]["frames"]))] for u in self.uuids}

    def __len__(self):
        return len(self.uuids)

    def prefetch(self):
        pass

    def get_image(self, uuid, index):

        if self.jpeg is None:
            self.jpeg = TurboJPEG(self.turbo)

        if self.prefetch or self.cache:
            data = self.img_buffer[uuid][index]
            if data is not None:
                image = self.jpeg.decode(data)
                return image

        frame = self.tracks[uuid]["frames"][index]
        frame_path = os.path.join(self.cnf.data_root, frame)

        if self.cache:
            with open(frame_path, "rb") as image_file:
                data = image_file.read()
                self.img_buffer[uuid][index] = data
                image = self.jpeg.decode(data)
        else:
            with open(frame_path, 'rb') as f:
                image = self.jpeg.decode(f.read())
            return image

        # pre-proc here...
        return image

    def __getitem__(self, item):
        # type: (int) -> (str, torch.tensor, torch.tensor, tuple)
        """
        :param item: element index
        :return: All (uuid, images, boxes) in the tracking sequence
            > uuid: string
            > images: (seq_len, 3, w, h)
            > boxes: (seq_len, 4)
            > nl: (3,)
        This only works with batch size = 1
        """

        images = []
        uuid = self.uuids[item]

        # Load and stack all the images
        for index in range(len(self.tracks[uuid]["frames"])):
            # Read and stack all the frames without resize or crop
            im = self.get_image(uuid, index)  # PIL image
            image = self.transforms(im)  # (3, W, H)
            images.append(image)

        images = torch.stack(images, 0)  # (N, 3, W, H)
        boxes = torch.tensor(self.tracks[uuid]["boxes"])
        nl = self.tracks[uuid]["nl"]

        return uuid, images, boxes, nl

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from time import time

    cnf = Conf(exp_name='m100')
    ds = CFNL_TrainDataset(cnf, mode="val", cache=True)
    print(len(ds))

    dataloader = DataLoader(
        ds, batch_size=1, num_workers=0, shuffle=False
    )

    # Benchmark
    for i in range(3):
        times = []
        t = time()
        for step, sample in enumerate(dataloader):
            a, b, c, d = sample
            print(step)
            if step == 5:
                break
            times.append(time() - t)
            t = time()

        print(f"Average running time with bs=1, workers=0, for 10 iters: {sum(times) / len(times)}")
