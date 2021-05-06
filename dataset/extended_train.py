# -*- coding: utf-8 -*-
# ---------------------

import os
import random
import numpy as np

import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

from dataset.utils import pad_sequence_mult
from dataset.base_datasets import CFNL_TrainDataset
from conf import Conf


class Extended_Dataset(CFNL_TrainDataset):

    def __init__(self, cnf=None, mode="train", cache=False):
        # type: (Conf, str, bool) -> None
        """
        :param cnf:  Configuration object
        :param mode: (train, train_all, val), see superclass-
        """

        self.input_size = (cnf.model_opts['INPUT_HEIGHT'], cnf.model_opts['INPUT_WIDTH'])
        self.transforms = tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        crop_width = cnf.model_opts.get('CROP_HEIGHT', 80)
        crop_height = cnf.model_opts.get('CROP_WIDTH', 80)
        self.crop_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((crop_height,crop_width)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        super().__init__(cnf, mode, tf, cache)

        # Precomputed nl embeddings
        self.prec_nl_embs = None

        if not self.cnf.model_opts["COMPUTE_NL_EMBS"]:
            embd_dir = os.path.join(self.cnf.data_root, 'precomp_nl_train.pt')
            self.prec_nl_embs = torch.load(embd_dir)

        # Sequence sampling
        self.max_seq_len = self.cnf.model_opts["MAX_SEQ_LEN"]

        # Additional inputs
        self.used_classes = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]  # COCO classes
        self.augmented_root = os.path.join(cnf.data_root, "rcnn_embs_train")
        self.crop_root = os.path.join(self.cnf.data_root, "egocrop_train")


    def __getitem__(self, item):
        # type: (int) -> (str, torch.tensor, torch.tensor, tuple)
        """
        :param item: element index
        :return: All (uuid, images, boxes) in the tracking sequence
        This only works with batch size = 1
        """
        images = []
        uuid = self.uuids[item]

        n_frames = len(self.tracks[uuid]["frames"])
        indices = [i for i in range(n_frames)]

        # Sample a subset of frames up to MAX_SEQ_LEN
        if n_frames > self.max_seq_len:
            indices = random.sample(indices, self.max_seq_len)
            indices = sorted(indices)

        # Image's original width and height
        orig_w = None
        orig_h = None

        # Load and stack all the images (full-size!)
        for index in indices:
            # Read and stack all the frames without resize or crop
            im = self.get_image(uuid, index)  # Decoded image
            if orig_w is None:
                orig_w, orig_h = im.shape[1], im.shape[0]

            im = im[..., ::-1].copy()  # BGR -> RGB
            image = self.transforms(im)  # (3, W, H)
            images.append(image)

        images = torch.stack(images, 0)  # (N, 3, W, H)

        # Retrieve BB's sequence
        boxes_ind = [i for i in range(len(self.tracks[uuid]["boxes"]))]

        # Limiting boxes seq len to 80
        if len(boxes_ind) > 80:
            boxes_ind = random.sample(boxes_ind, 80)
            boxes_ind = sorted(boxes_ind)  # LOOOL I forgot this!!

        class_idx = torch.ones(len(boxes_ind)) * 1  # Tracked vehicle has class 1
        boxes = torch.tensor(self.tracks[uuid]["boxes"], dtype=torch.float32)[boxes_ind]  # (len, 4)
        boxes = torch.cat([class_idx.unsqueeze(-1), boxes], dim=1).unsqueeze(1)  # (len, 1, 5)

        # Load object-embeddings
        augmented_file = os.path.join(self.augmented_root, f"{uuid}.pt")
        augmented_track = torch.load(augmented_file)

        dets = augmented_track["detected_boxes"]
        ego_id = augmented_track["ego_ind"]

        # Remove the unused
        dets = [dets[i] for i in boxes_ind]
        ego_id = [ego_id[i] for i in boxes_ind]

        # sanity check
        det_frames = augmented_track["frames"]
        assert det_frames == self.tracks[uuid]["frames"]

        # remove ego detection box, also remove low confidence boxes and boxes from useless classes
        dets_idx = [[di for di, det in enumerate(frame_dets) if det[0] in self.used_classes and
                     det[-1] > 0.85 and di != ego_id[frame_idx]] for frame_idx, frame_dets in enumerate(dets)]

        dets = [d[di][:, :5] for d, di in zip(dets, dets_idx)]  # keep only filtered detections

        # Add RCNN visual embeddings
        rcnn_features = augmented_track["features"]
        rcnn_features = [rcnn_features[i] for i in boxes_ind]

        rcnn_features = [f[i] for f, i in zip(rcnn_features, dets_idx)]

        # embedding vector is: bb[5] + rcnn_feature[256]
        dets = [torch.cat([d, f], dim=1) for d, f in zip(dets, rcnn_features)]

        dets, boxes_len = bb_pad(dets)

        # Add fake R-CNN embedding for ego vehicle
        ego_pad = torch.zeros(len(boxes_ind), 1, 256)
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

        for i in boxes_ind:
            crop_file = os.path.join(self.crop_root, uuid, f"{i}.jpg")
            with open(crop_file, 'rb') as f:
                crop = self.jpeg.decode(f.read())

            crop = crop[..., ::-1].copy()  # bgr to rgb
            crop = self.crop_transforms(crop)

            ego_crops.append(crop)

        # stack
        ego_crops_tensor = torch.stack(ego_crops)  # (N_BB, 3, W, H)

        # pick uuid of negative embedding
        if self.mode != "val":
            # For negative samples pick a random one
            ix = item
            while ix == item:
                # Avoid picking the real index
                ix = random.randint(0, len(self.uuids) - 1)
            negative_uuid = self.uuids[ix]
        else:
            # Make validation deterministic
            negative_uuid = self.uuids[(item ** 2 + 8) % len(self.uuids)]

        assert negative_uuid != uuid  # better safe than sorry

        # Retrieve embedding
        if self.prec_nl_embs is None:
            positive = self.tracks[uuid]["nl"]
            negative = self.tracks[negative_uuid]["nl"]
        else:
            # Use pre-computed ones
            positive = self.prec_nl_embs[uuid]
            negative = self.prec_nl_embs[negative_uuid]

        return images, boxes, positive, negative, torch.tensor(indices),\
               torch.tensor(boxes_ind), boxes_len, ego_crops_tensor, uuid

def bb_pad(seq):
    l = torch.tensor([len(s) for s in seq])
    seq = pad_sequence([s for s in seq], batch_first=True)

    # Use -1 as "class" for padded elements, this will become 0
    # when we add 1 to preserve the 1 index for ego vehicle
    seq[:, :, 0] += torch.where(seq[:, :, 0] == 0, -1, 0)

    return seq, l

def collate_fn_padd(batch):
    # type: (list) -> tuple

    images = pad_sequence([b[0] for b in batch], batch_first=True)
    images_len = torch.tensor([len(b[4]) for b in batch])

    bbs = pad_sequence_mult([b[1] for b in batch], paddable_shapes=len(batch[0][1].shape) - 1)
    bbs_len = torch.tensor([len(b[5]) for b in batch])

    im_indices = pad_sequence([b[4] for b in batch], batch_first=True, padding_value=-1)
    bb_indices = pad_sequence([b[5] for b in batch], batch_first=True, padding_value=-1)

    if type(batch[0][2]) == torch.Tensor:
        # pre-computed embeddings
        pos = torch.stack([b[2] for b in batch])
        neg = torch.stack([b[3] for b in batch])
    else:
        # tuples
        pos = [s for b in batch for s in b[2]]
        neg = [s for b in batch for s in b[3]]

    obj_lenghts = pad_sequence([b[6] for b in batch], batch_first=True)
    ego_crops = pad_sequence([b[7] for b in batch], batch_first=True)
    anch_uuid = [b[8] for b in batch]

    return images, images_len, bbs, bbs_len, pos, neg, \
           im_indices, bb_indices, obj_lenghts, ego_crops, anch_uuid
