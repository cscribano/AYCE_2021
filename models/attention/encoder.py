# -*- coding: utf-8 -*-
# ---------------------

import torch
import torch.nn as nn

from models.attention import MultiHeadAttention, PositionalEncoding, clones

class TransformerEncoder(nn.Module):

    def __init__(self, emb_dim=512, n_blocks=6, max_len=5000, dropout=0.1, pe_dropout=0.0, **kwargs):
        """
        :param emb_dim: the dimensionality of the input embeddings
        :param n_blocks: Number of identical stacked blocks
        :param max_len: Maximum length of a target sequence, required for positional embeddings
        :param dropout: the amount of dropout to use in attention block
        """

        super().__init__()

        self.emb_dim = 512
        self.b_blocks = n_blocks

        self.pos_embedding = PositionalEncoding(emb_dim, pe_dropout, max_len)

        '''
        The encoder is made of N=6 identical blocks
        '''
        self.encoder_blocks = clones(EncoderBlock(emb_dim, dropout=dropout, **kwargs), n_blocks)

    def forward(self, src, mask=None, pos=None):
        """
        :param src: Embedding for source sequence (padded, batch first): (BS, seq_len, emb_dim)
        :param mask: Mask to avoid attending to padding: (BS, seq_len, 1)
        :return: (BS, seq_len, emb_dim)
        """

        x = self.pos_embedding(src, pos)
        for b in self.encoder_blocks:
            x = b(x, mask)

        return x

class EncoderBlock(nn.Module):

    def __init__(self, emb_dim=512, dropout=0.1, heads=8):
        """
        :param emb_dim: length of input sequence embedding space
        :param dropout: dropout rate [0,1]
        """
        super().__init__()

        self.emb_dim = emb_dim
        assert emb_dim % heads == 0, "The embeddings dim must be divisible by heads number"

        self.attention_layer = MultiHeadAttention(emb_dim, emb_dim // heads, heads)
        self.dropout = nn.Dropout(dropout)
        self.atn_norm = nn.LayerNorm(emb_dim)

        self.linear_block = nn.Sequential(
            nn.Linear(emb_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, emb_dim),
            nn.Dropout(dropout)
        )
        self.linear_norm = nn.LayerNorm(emb_dim)


    def forward(self, input, mask=None):
        """
        :param input: Embedding for source sequence (padded, batch first): (BS, seq_len, emb_dim)
        :param mask: Mask to avoid attending to padding: (BS, seq_len, 1)
        :return: (BS, seq_len, emb_dim)
        """

        # Sub-layer 1
        # Compute Multi head-attention and apply residual connection
        atn = self.attention_layer(input, input, input, mask=mask)
        atn = self.dropout(atn)
        # In Encoder self-attention Query, Key and Value are the same thing
        atn = self.atn_norm(atn+input)

        # Sub-layer 2
        lin = self.linear_block(atn)
        lin = self.linear_norm(lin+atn)

        return lin


if __name__ == '__main__':

    from models.attention import TransformerDecoder

    s_emb = torch.ones(32, 12, 256, dtype=torch.long).to('cuda')  # (BS, SL, emb_dim)
    t_emb = torch.ones(32, 56, 512, dtype=torch.long).to('cuda')  # (BS, TL, emb_dim)

    e = TransformerEncoder(256).to('cuda')
    d = TransformerDecoder(512, enc_emb_dim=256).to('cuda')

    enc_out = e(s_emb)
    dec_out = d(t_emb, enc_out)

    print(enc_out.shape, dec_out.shape)
