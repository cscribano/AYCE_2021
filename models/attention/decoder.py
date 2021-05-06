# -*- coding: utf-8 -*-
# ---------------------

import torch
import torch.nn as nn

from models.attention import MultiHeadAttention, PositionalEncoding, clones

class TransformerDecoder(nn.Module):

    def __init__(self, emb_dim=512, n_blocks=6, enc_emb_dim=None,
                 output_dim=100, max_len=5000, dropout=0.1):
        """
        :param emb_dim: the dimensionality of the input embeddings-
        :param n_blocks: Number of identical stacked blocks
        :param max_len: Maximum length of a target sequence, required for positional embeddings
        :param dropout: the amount of dropout to use
        """
        super().__init__()

        self.n_blocks = n_blocks

        self.pos_embedding = PositionalEncoding(emb_dim, 0.0, max_len)

        '''
        The Decoder is also made of N=6 identical blocks
        '''

        self.decoder_blocks = clones(DecoderBlock(emb_dim, dropout=dropout, enc_emb_dim=enc_emb_dim), n_blocks)

        if output_dim is not None:
            self.output = nn.Linear(emb_dim, output_dim)
        else:
            self.output = None

    def forward(self, trg, encoder_out, mask=None, encoder_mask=None, pos=None):

        # this is identical to encoder
        x = self.pos_embedding(trg, pos)

        for db in self.decoder_blocks:
            x = db(x, encoder_out, mask, encoder_mask)

        if self.output is not None:
            x = self.output(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, emb_dim=512, dropout=0.1, heads=8, enc_emb_dim=None):
        """
        :param emb_dim:
        :param dropout:
        """
        super().__init__()

        self.emb_dim = emb_dim

        if enc_emb_dim is None:
            enc_emb_dim = emb_dim

        assert (emb_dim % heads == 0 and enc_emb_dim % heads == 0)

        # 1-st sub-layer: (masked) self-attention)
        self.self_attention_layer = MultiHeadAttention(emb_dim, emb_dim // heads, heads)
        self.self_atn_dropout = nn.Dropout(dropout)
        self.self_atn_norm = nn.LayerNorm(emb_dim)

        # 2-nd sub-layer: encoder-decoder attention
        self.attention_layer = MultiHeadAttention(emb_dim, emb_dim // heads, heads,
                                                  kv_emb_dim=enc_emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.atn_norm = nn.LayerNorm(emb_dim)

        # 3-rd sub-layer: Feed Forward(s)
        self.linear_block = nn.Sequential(
            nn.Linear(emb_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, emb_dim),
            nn.Dropout(dropout)
        )
        self.linear_norm = nn.LayerNorm(emb_dim)

    def forward(self, input, encoder_out, mask=None, encoder_mask=None):
        """
        :param input:
        :param encoder_out:
        :param mask:
        :param encoder_mask:
        :return:
        """

        # Sub-layer 1
        # Compute Multi head-self attention and apply residual connection
        self_atn = self.self_attention_layer(input, input, input, mask=mask)
        self_atn = self.self_atn_dropout(self_atn)
        # In self-attention Query, Key and Value are the same thing
        self_atn = self.self_atn_norm(self_atn+input)

        '''
        Sub-layer 2: Encoder-Decoder attention
        Here attention Query is the Decoder state, aka the output from self-attention
        Key and Value is the decoder output.
        '''
        ed_atn = self.attention_layer(self_atn, encoder_out, encoder_out, mask=encoder_mask)
        ed_atn = self.dropout(ed_atn)
        # In Encoder self-attention Query, Key and Value are the same thing
        ed_atn = self.atn_norm(ed_atn+self_atn)

        # Sub-layer 3
        lin = self.linear_block(ed_atn)
        lin = self.linear_norm(lin+ed_atn)

        return lin
