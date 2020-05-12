# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_keyphrase.dataloader import (TOKENS, TOKENS_LENS, TARGET)


class Attention(nn.Module):
    """

    """

    def __init__(self, dim_size):
        super().__init__()
        self.in_proj = nn.Linear(dim_size, dim_size)

    def forward(self, x, target_embedding, encoder_input, encoder_output, encoder_mask):
        pass


class CopyCnn(nn.Module):
    def __init__(self, args, vocab2id):
        super().__init__()
        self.args = args
        self.vocab2id = vocab2id
        self.embedding = nn.Embedding(len(vocab2id), args.dim_size)
        self.encoder = CopyCnnEncoder(vocab2id=vocab2id, embedding=self.embedding, args=args)
        self.decoder = CopyCnnDecoder(vocab2id=vocab2id, embedding=self.embedding, args=args)

    def forward(self, src_dict, encoder_output):
        if encoder_output is None:
            encoder_output = self.encoder(src_dict)


class CopyCnnEncoder(nn.Module):
    def __init__(self, vocab2id, embedding, args):
        super().__init__()
        self.vocab2id = vocab2id
        self.embedding = embedding
        self.args = args
        self.dim_size = args.dim_size
        self.kernel_size = (args.kernal_width, self.dim_size)
        self.dropout = args.dropout
        self.convolution_layers = []
        for i in range(args.encoder_layer_num):
            layer = nn.Conv2d(in_channels=1, out_channels=2 * self.dim_size,
                              kernel_size=self.kernel_size, bias=True)
            self.convolution_layers.append(layer)

    def forward(self, src_dict):
        tokens = src_dict[TOKENS]
        x = self.embedding(tokens).unsqueeze(1)
        # x = tokens.unsqueeze(1)
        layer_output = [x]
        for layer in self.convolution_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x)
            x = F.glu(x, dim=1) + layer_output[-1]
            layer_output.append(x)
        return x


class CopyCnnDecoder(nn.Module):
    def __init__(self, vocab2id, embedding, args):
        super().__init__()
        self.vocab2id = vocab2id
        self.embedding = embedding
        self.args = args
        self.vocab_size = self.args.vocab_size
        self.max_oov_count = self.args.max_oov_count
        self.total_vocab_size = self.vocab_size + self.max_oov_count
        self.dim_size = args.dim_size
        self.kernel_size = (args.kernal_width, self.dim_size)
        self.dropout = args.dropout
        self.convolution_layers = []
        self.attn_linear_layers = []
        self.decoder_layer_num = args.decoder_layer_num
        for i in range(self.decoder_layer_num):
            conv_layer = nn.Conv2d(in_channels=1, out_channels=2 * self.dim_size,
                                   kernel_size=self.kernel_size, bias=True)
            self.convolution_layers.append(conv_layer)
            attn_linear_layer = nn.Linear(self.dim_size, self.dim_size, bias=True)
            self.attn_linear_layers.append(attn_linear_layer)
        self.generate_proj = nn.Linear(self.dim_size, self.vocab_size)
        self.copy_proj = nn.Linear(self.dim_size, self.total_vocab_size)

    def forward(self, src_dict, prev_tokens, encoder_output):
        """

        :param src_dict:
        :param prev_tokens:
        :param encoder_output:
        :return:
        """
        src_tokens = src_dict[TOKENS]
        tokens = src_dict[TARGET][:, :-1]
        x = self.embedding(tokens).unsqueeze(1)
        prev_x = self.embedding(prev_tokens)
        src_x = self.embedding(src_tokens)
        layer_output = [x]
        for conv_layer, linear_layer in zip(self.convolution_layers, self.attn_linear_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv_layer(x)
            x = F.glu(x, dim=1) + layer_output[-1]
            # attention
            d = linear_layer(x) + prev_x
            attn_weights = torch.softmax(torch.bmm(encoder_output, d.unsqueeze(2)), dim=1)
            c = attn_weights * (encoder_output + src_x)
            # residual connection
            final_output = x + c.squeeze(2)
            layer_output.append(final_output)
        generate_logits = self.generate_proj(layer_output[-1])

    def forward_one_pass(self):
        pass

    def forward_auto_regressive(self):
        pass

    def get_attn_read(self, encoder_output, src_tokens_with_oov, decoder_output, encoder_output_mask):
        pass
