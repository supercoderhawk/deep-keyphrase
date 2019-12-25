# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import (TransformerEncoder, TransformerDecoder,
                                          TransformerEncoderLayer, TransformerDecoderLayer)
from deep_keyphrase.dataloader import (TOKENS, TOKENS_LENS, TOKENS_OOV,
                                       UNK_WORD, PAD_WORD, OOV_COUNT, TARGET)


def get_position_encoding(input_tensor):
    batch_size, position, dim_size = input_tensor.size()
    assert dim_size % 2 == 0
    num_timescales = dim_size // 2
    time_scales = torch.arange(0, position + 1, dtype=torch.float).unsqueeze(1)
    dim_scales = torch.arange(0, num_timescales, dtype=torch.float).unsqueeze(0)
    dim_val = torch.pow(1.0e4, 2 * dim_scales / dim_size)
    matrix = torch.matmul(time_scales, 1.0 / dim_val)
    position_embed = torch.cat([torch.sin(matrix), torch.cos(matrix)], dim=1).repeat(batch_size, 1, 1)

    if torch.cuda.is_available():
        position_embed = position_embed.cuda()

    return position_embed


class CopyTransformer(nn.Module):
    def __init__(self, args, vocab2id):
        super().__init__()
        embedding = nn.Embedding(len(vocab2id), args.input_dim, vocab2id[PAD_WORD])
        self.encoder = CopyTransformerEncoder(embedding=embedding,
                                              input_dim=args.input_dim,
                                              head_size=args.src_head_size,
                                              feed_forward_dim=args.feed_forward_dim,
                                              dropout=args.src_dropout,
                                              num_layers=args.src_layers)
        self.decoder = CopyTransformerDecoder(embedding=embedding,
                                              vocab2id=vocab2id,
                                              args=args)

    def forward(self, src_dict, prev_output_tokens, encoder_output, encoder_mask,
                prev_decoder_state, position, prev_copy_state):
        if torch.cuda.is_available():
            src_dict[TOKENS] = src_dict[TOKENS].cuda()
            src_dict[TOKENS_LENS] = src_dict[TOKENS_LENS].cuda()
            src_dict[TOKENS_OOV] = src_dict[TOKENS_OOV].cuda()
            src_dict[OOV_COUNT] = src_dict[OOV_COUNT].cuda()
            prev_output_tokens = prev_output_tokens.cuda()
            prev_decoder_state = prev_decoder_state.cuda()
        if encoder_output is None:
            encoder_output, encoder_mask = self.encoder(src_dict=src_dict)
        output = self.decoder(prev_output_tokens=prev_output_tokens,
                              prev_decoder_state=prev_decoder_state,
                              position=position,
                              encoder_output=encoder_output,
                              encoder_mask=encoder_mask,
                              src_dict=src_dict,
                              prev_copy_state=prev_copy_state)
        return output


class CopyTransformerEncoder(nn.Module):
    def __init__(self, embedding, input_dim, head_size,
                 feed_forward_dim, dropout, num_layers):
        super().__init__()
        self.embedding = embedding
        self.dropout = dropout
        layer = TransformerEncoderLayer(d_model=input_dim,
                                        nhead=head_size,
                                        dim_feedforward=feed_forward_dim,
                                        dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer=layer, num_layers=num_layers)

    def forward(self, src_dict):
        batch_size, max_len = src_dict[TOKENS].size()
        mask_range = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1)

        if torch.cuda.is_available():
            mask_range = mask_range.cuda()
        mask = mask_range >= src_dict[TOKENS_LENS]
        # mask = (mask_range > src_dict[TOKENS_LENS].unsqueeze(1)).expand(batch_size, max_len, max_len)
        src_embed = self.embedding(src_dict[TOKENS]).transpose(1, 0)
        pos_embed = get_position_encoding(src_embed)
        src_embed = src_embed + pos_embed
        src_embed = F.dropout(src_embed, p=self.dropout, training=self.training)
        output = self.encoder(src_embed, src_key_padding_mask=mask).transpose(1, 0)
        return output, mask


class CopyTransformerDecoder(nn.Module):
    def __init__(self, embedding, vocab2id, args):
        super().__init__()
        self.embedding = embedding
        self.vocab2id = vocab2id
        self.args = args
        self.input_dim = args.input_dim
        self.head_size = args.target_head_size
        self.feed_forward_dim = args.feed_forward_dim
        self.dropout = args.target_dropout
        self.num_layers = args.target_layers
        self.target_max_len = args.max_target_len
        self.max_oov_count = args.max_oov_count
        self.vocab_size = embedding.num_embeddings

        layer = TransformerDecoderLayer(d_model=self.input_dim,
                                        nhead=self.head_size,
                                        dim_feedforward=self.feed_forward_dim,
                                        dropout=self.dropout)
        self.decoder = TransformerDecoder(decoder_layer=layer, num_layers=self.num_layers)
        self.input_copy_proj = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.copy_proj = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.embed_proj = nn.Linear(2 * self.input_dim, self.input_dim, bias=False)
        self.generate_proj = nn.Linear(self.input_dim, self.vocab_size, bias=False)

    def forward(self, prev_output_tokens, prev_decoder_state, position,
                encoder_output, encoder_mask, src_dict, prev_copy_state):
        if self.args.input_feeding and not self.training:
            output = self.forward_auto_regressive(prev_output_tokens, prev_decoder_state, position,
                                                  encoder_output, encoder_mask, src_dict, prev_copy_state)
        else:
            output = self.forward_one_pass(encoder_output, encoder_mask, src_dict)
        return output

    def forward_one_pass(self, encoder_output, encoder_mask, src_dict):
        batch_size = len(src_dict[TOKENS])
        token_embed = self.embedding(src_dict[TARGET][:, :-1])
        src_tokens_with_oov = src_dict[TOKENS_OOV]
        pos_embed = get_position_encoding(token_embed)
        # B x seq_len x H
        src_embed = token_embed + pos_embed
        decoder_input = F.dropout(src_embed, p=self.dropout, training=self.training)
        decoder_input_mask = torch.triu(torch.ones(self.input_dim, self.input_dim), 1)
        decoder_output = self.decoder(tgt=decoder_input,
                                      memory=encoder_output.transpose(1, 0),
                                      memory_key_padding_mask=decoder_input_mask)
        # B x seq_len x H
        decoder_output = decoder_output.transpose(1, 0)
        generation_logits = torch.exp(self.generate_proj(decoder_output).squeeze(1))
        generation_oov_logits = torch.zeros(batch_size, self.max_oov_count)
        if torch.cuda.is_available():
            generation_oov_logits = generation_oov_logits.cuda()
        generation_logits = torch.cat([generation_logits, generation_oov_logits], dim=1)
        copy_logits = self.get_copy_score(encoder_output,
                                          src_tokens_with_oov,
                                          decoder_output,
                                          encoder_mask)
        total_logit = generation_logits + copy_logits
        total_prob = total_logit / torch.sum(total_logit, 1).unsqueeze(1)
        total_prob = torch.log(total_prob)

        return total_prob, decoder_output.squeeze(1), None, encoder_output, encoder_mask

    def forward_auto_regressive(self, prev_output_tokens, prev_decoder_state, position,
                encoder_output, encoder_mask, src_dict, prev_copy_state):
        src_tokens = src_dict[TOKENS]
        src_tokens_with_oov = src_dict[TOKENS_OOV]
        batch_size, src_max_len = src_tokens.size()
        prev_output_tokens = torch.as_tensor(prev_output_tokens, dtype=torch.int64)
        if torch.cuda.is_available():
            prev_output_tokens = prev_output_tokens.cuda()
        copy_state = self.get_attn_read_input(encoder_output,
                                              prev_decoder_state,
                                              prev_output_tokens[:, -1:],
                                              src_tokens_with_oov,
                                              src_max_len,
                                              prev_copy_state)
        # map copied oov tokens to OOV idx to avoid embedding lookup error
        prev_output_tokens[prev_output_tokens >= self.vocab_size] = self.vocab2id[UNK_WORD]
        token_embed = self.embedding(prev_output_tokens)

        pos_embed = get_position_encoding(token_embed)
        # B x seq_len x H
        src_embed = token_embed + pos_embed
        decoder_input = self.embed_proj(torch.cat([src_embed, copy_state], dim=2)).transpose(1, 0)
        decoder_input = F.dropout(decoder_input, p=self.dropout, training=self.training)
        decoder_input_mask = torch.triu(torch.ones(self.input_dim, self.input_dim), 1)
        # B x seq_len x H
        decoder_output = self.decoder(tgt=decoder_input,
                                      memory=encoder_output.transpose(1, 0),
                                      memory_key_padding_mask=decoder_input_mask)
        decoder_output = decoder_output.transpose(1, 0)
        
        # B x 1 x H
        decoder_output = decoder_output[:, -1:, :]
        generation_logits = self.generate_proj(decoder_output).squeeze(1)
        generation_oov_logits = torch.zeros(batch_size, self.max_oov_count)
        if torch.cuda.is_available():
            generation_oov_logits = generation_oov_logits.cuda()
        generation_logits = torch.cat([generation_logits, generation_oov_logits], dim=1)
        copy_logits = self.get_copy_score(encoder_output,
                                          src_tokens_with_oov,
                                          decoder_output,
                                          encoder_mask)
        total_logit = torch.exp(generation_logits) + copy_logits
        total_prob = total_logit / torch.sum(total_logit, 1).unsqueeze(1)
        total_prob = torch.log(total_prob)
        return total_prob, decoder_output.squeeze(1), copy_state, encoder_output, encoder_mask

    def get_attn_read_input(self, encoder_output, prev_context_state,
                            prev_output_tokens, src_tokens_with_oov,
                            src_max_len, prev_copy_state):
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
        if prev_copy_state is not None:
            copy_state = torch.cat([prev_copy_state, copy_state], dim=1)
        return copy_state

    def get_copy_score(self, encoder_out, src_tokens_with_oov, decoder_output, encoder_output_mask):
        """

        :param encoder_out: B x L x SH
        :param src_tokens_with_oov: B x L
        :param decoder_output: B x dec_len x TH
        :param encoder_output_mask: B x L
        :return: B x dec_len x V
        """
        # copy_score: B x L
        dec_len = decoder_output.size(1)
        batch_size = len(encoder_out)
        # copy_score: B x L x dec_len
        copy_score_in_seq = torch.bmm(torch.tanh(self.copy_proj(encoder_out)),
                                      decoder_output.permute(0, 2, 1))
        copy_score_mask = encoder_output_mask.unsqueeze(2).repeat(1, 1, dec_len)
        copy_score_in_seq.masked_fill_(copy_score_mask, float('-inf'))
        copy_score_in_seq = torch.exp(copy_score_in_seq)
        total_vocab_size = self.vocab_size + self.max_oov_count
        copy_score_in_vocab = torch.zeros(batch_size, total_vocab_size, dec_len)
        if torch.cuda.is_available():
            copy_score_in_vocab = copy_score_in_vocab.cuda()
        token_ids = src_tokens_with_oov.unsqueeze(2).repeat(1, 1, dec_len)
        copy_score_in_vocab.scatter_add_(1, token_ids, copy_score_in_seq)
        copy_score_in_vocab = copy_score_in_vocab.permute(0, 2, 1)

        return copy_score_in_vocab
