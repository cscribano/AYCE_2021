# -*- coding: utf-8 -*-
# ---------------------

import math
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable

def clones(module, N):
    # Produce N identical layers.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def len_to_mask(lenghts):
    # type: (torch.tensor) -> torch.tensor
    max_len = max(lenghts.view(-1))
    mask = torch.arange(max_len)[None, ...] < lenghts[... , None]

    return mask.unsqueeze(-2)

class PositionalEncoding(nn.Module):
    """
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=None):
        if pos is None:
            x = x + Variable(self.pe[:, :x.size(-2)],
                             requires_grad=False)
        else:
            x = x + Variable(self.pe[:, pos], requires_grad=False)[0]

        return self.dropout(x)
