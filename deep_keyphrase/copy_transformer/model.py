# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.nn.modules.transformer import (TransformerEncoder, TransformerDecoder,
                                          TransformerEncoderLayer, TransformerDecoderLayer)
from deep_keyphrase.dataloader import TOKENS, TOKENS_LENS, TOKENS_OOV, UNK_WORD


def get_position_encoding(input_tensor, position, dim_size):
    batch_size, seq_len = input_tensor.size()
    half_len = seq_len // 2 + seq_len % 2
    sin_idx = torch.arange(half_len, dtype=torch.float).expand(batch_size, half_len) * 2
    cos_idx = sin_idx + 1
    sin_embed = torch.sin(position / torch.pow(10000, sin_idx / dim_size))
    cos_embed = torch.sin(position / torch.pow(10000, cos_idx / dim_size))
    position_embed = torch.stack([sin_embed, cos_embed], dim=2).view(batch_size, -1)
    # truncate the real embedding when seq_len is odd
    position_embed = position_embed[:, :seq_len, :]
    return position_embed


class CopyTransformer(nn.Module):
    def __init__(self, input_dim, src_head_size, target_head_size, feed_forward_dim,
                 src_dropout, target_dropout, src_layers, target_layers, target_max_len):
        super().__init__()
        self.encoder = CopyTransformerEncoder(input_dim=input_dim,
                                              head_size=src_head_size,
                                              feed_forward_dim=feed_forward_dim,
                                              dropout=src_dropout,
                                              num_layers=src_layers)
        self.decoder = CopyTransformerDecoder(input_dim=input_dim,
                                              head_size=target_head_size,
                                              feed_forward_dim=feed_forward_dim,
                                              dropout=target_dropout,
                                              num_layers=target_layers,
                                              target_max_len=target_max_len)

    def forward(self, src_dict, prev_output_tokens, encoder_output, encoder_mask,
                prev_hidden_state, position):
        if encoder_output is None:
            encoder_output, encoder_mask = self.encoder(src_dict=src_dict)
        output = self.decoder(prev_output_tokens=prev_output_tokens,
                              prev_hidden_state=prev_hidden_state,
                              position=position,
                              encoder_output=encoder_output,
                              encoder_mask=encoder_mask,
                              src_dict=src_dict)
        return output


class CopyTransformerEncoder(nn.Module):
    def __init__(self, input_dim, head_size, feed_forward_dim, dropout, num_layers):
        super().__init__()
        layer = TransformerEncoderLayer(d_model=input_dim,
                                        nhead=head_size,
                                        dim_feedforward=feed_forward_dim,
                                        dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer=layer, num_layers=num_layers)

    def forward(self, src_dict):
        batch_size, max_len = src_dict[TOKENS].size()
        mask = torch.arange(max_len).expand(batch_size, max_len) < src_dict[TOKENS_LENS]
        output = self.encoder(src_dict[TOKENS], mask)
        return output, mask


class CopyTransformerDecoder(nn.Module):
    def __init__(self, input_dim, head_size, feed_forward_dim, dropout, num_layers, target_max_len):
        super().__init__()
        layer = TransformerDecoderLayer(d_model=input_dim,
                                        nhead=head_size,
                                        dim_feedforward=feed_forward_dim,
                                        dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layer=layer, num_layers=num_layers)
        self.input_copy_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.generate_proj = nn.Linear(input_dim, self.vocab_size, bias=False)
        self.input_dim = input_dim
        self.target_max_len = target_max_len

    def forward(self, prev_output_tokens, prev_hidden_state, position,
                encoder_output, encoder_mask, src_dict):
        src_tokens = src_dict[TOKENS]
        src_tokens_with_oov = src_dict[TOKENS_OOV]
        batch_size, src_max_len = src_tokens.size()
        prev_output_tokens = torch.as_tensor(prev_output_tokens, dtype=torch.int64)
        if torch.cuda.is_available():
            prev_output_tokens = prev_output_tokens.cuda()
        copy_state = self.get_attn_read_input(encoder_output,
                                              prev_hidden_state,
                                              prev_output_tokens,
                                              src_tokens_with_oov,
                                              src_max_len)
        # map copied oov tokens to OOV idx to avoid embedding lookup error
        prev_output_tokens[prev_output_tokens >= self.vocab_size] = self.vocab2id[UNK_WORD]
        token_embed = self.embedding(prev_output_tokens)
        pos_embed = get_position_encoding(prev_output_tokens, position, self.input_dim)
        src_embed = token_embed + pos_embed
        decoder_input = torch.cat([src_embed, copy_state], dim=2)
        decoder_input_mask = torch.triu(torch.ones(self.input_dim, self.input_dim), 1)
        # B x decoder_input_seq_len x H
        decoder_output = self.decoder(tgt=decoder_input, memory=encoder_output,
                                      tgt_mask=decoder_input_mask, memory_mask=encoder_mask)
        # B x 1 x H
        decoder_output = decoder_output[:, -1:, :]
        generation_logits = self.generate_proj(decoder_output).squeeze(1)
        generation_oov_logits = torch.zeros(batch_size, self.max_oov_count)
        generation_logits = torch.cat([generation_logits, generation_oov_logits], dim=1)
        copy_logits = self.get_copy_score(encoder_output,
                                          src_tokens_with_oov,
                                          decoder_output,
                                          encoder_mask)
        total_logit = torch.exp(generation_logits) + copy_logits
        total_prob = total_logit / torch.sum(total_logit, 1).unsqueeze(1)
        total_prob = torch.log(total_prob)
        return total_prob, decoder_output.squeeze(1)

    def get_attn_read_input(self, encoder_output, prev_context_state,
                            prev_output_tokens, src_tokens_with_oov, src_max_len):
        """
        build CopyNet decoder input of "attentive read" part.
        :param encoder_output:
        :param prev_context_state:
        :param prev_output_tokens:
        :param src_tokens_with_oov:
        :return:
        """
        # mask : B x SL x 1
        mask_bool = torch.eq(prev_output_tokens.repeat(1, src_max_len), src_tokens_with_oov).unsqueeze(2)
        mask = mask_bool.type_as(encoder_output)
        # B x SL x H
        aggregate_weight = torch.tanh(self.input_copy_proj(torch.mul(mask, encoder_output)))
        # when all prev_tokens are not in src_tokens, don't execute mask -inf to avoid nan result in softmax
        no_zero_mask = ((mask != 0).sum(dim=1) != 0).repeat(1, src_max_len).unsqueeze(2)
        input_copy_logit_mask = no_zero_mask * mask_bool
        input_copy_logit = torch.bmm(aggregate_weight, prev_context_state.unsqueeze(2))
        input_copy_logit.masked_fill_(input_copy_logit_mask, float('-inf'))
        input_copy_weight = torch.softmax(input_copy_logit.squeeze(2), 1)
        # B x 1 x H
        copy_state = torch.bmm(input_copy_weight.unsqueeze(1), encoder_output)
        return copy_state

    def get_copy_score(self, encoder_out, src_tokens_with_oov, decoder_output, encoder_output_mask):
        # copy_score: B x L
        copy_score_in_seq = torch.bmm(torch.tanh(self.copy_proj(encoder_out)),
                                      decoder_output.permute(0, 2, 1)).squeeze(2)
        copy_score_in_seq.masked_fill_(encoder_output_mask, float('-inf'))
        copy_score_in_seq = torch.exp(copy_score_in_seq)
        copy_score_in_vocab = torch.zeros(len(encoder_out), self.vocab_size + self.max_oov_count)
        if torch.cuda.is_available():
            copy_score_in_vocab = copy_score_in_vocab.cuda()
        copy_score_in_vocab.scatter_add_(1, src_tokens_with_oov, copy_score_in_seq)
        return copy_score_in_vocab
