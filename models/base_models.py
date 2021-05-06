# -*- coding: utf-8 -*-
# ---------------------

from typing import *

import torch
from path import Path
from torch import nn
from abc import ABCMeta
from abc import abstractmethod


class BaseModel(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

        # Dataset
        self.train_set = (None, {'arg1', None, "argN"}, None)
        self.val_set = (None, {'arg1', None, "argN"})

    def forward(self, *args, **kwargs):
        # type: (*str, **int) -> torch.Tensor
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    @abstractmethod
    def train_loss(self, x):
        # type: (tuple) -> torch.Tensor
        """
        Return a computed value that can be used for backward pass computation
        Should be overridden by all subclasses.
        :param x: Data returned from Dataloader
        :return: Computed Loss
        """
        ...

    @abstractmethod
    def val_loss(self, x):
        # type: (tuple) -> torch.Tensor
        """
        Return a computed value that will be used for model evaluation after each epoch.
        This value won't be used for gradient computation.
        :param x: Data returned from Dataloader
        :return: Computed Validation score
        """
        ...

    def load_weights(self, state_dict):
        # type: (dict) -> None
        self.load_state_dict(state_dict)

    @property
    def is_cuda(self):
        # type: () -> bool
        """
        :return: `True` if the model is on Cuda; `False` otherwise
        """
        return next(self.parameters()).is_cuda


    def requires_grad(self, flag):
        # type: (bool) -> None
        """
        :param flag: True if the model requires gradient, False otherwise
        """
        for p in self.parameters():
            p.requires_grad = flag

class TestableModel:

    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = None
        ...

    @abstractmethod
    def test_step(self, images, bbs, nl, precomp, **kwargs):
        # type: (torch.tensor, torch.tensor, tuple, any, **int) -> float
        """
        :param images: (1, N, W, H), sequence of pre-processed images
        :param bbs: (1, N, 4), sequence of pre-processed bounding boxed
        :param nl: (3,), sequence of 3 nl descriptions
        :param precomp: (Optional), precomputed values that might be used
        :return: a single float value, measure of distance between sequence and (all) nl descriptions
        """
        ...

