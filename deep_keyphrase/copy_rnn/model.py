# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
from deep_keyphrase.utils.constants import *


class Attention(nn.Module):
    def __init__(self, input_dim, output_dim, score_mode='general'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.score_mode = score_mode
        if self.score_mode == 'general':
            self.attn = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.output_proj = nn.Linear(self.input_dim + self.output_dim, self.output_dim)

    def score(self, query, key, encoder_padding_mask):
        if self.score_mode == 'general':
            attn_weights = torch.bmm(self.attn(query), key.permute(0, 2, 1))
            attn_weights = torch.squeeze(attn_weights, 1)
        else:
            raise ValueError('value error')
        # mask input padding to -Inf, they will be zero after softmax.
        if encoder_padding_mask is not None:
            attn_weights = attn_weights.float().masked_fill_(
                encoder_padding_mask,
                float('-inf'))
        attn_weights = torch.softmax(attn_weights, 1)
        return attn_weights

    def forward(self, decoder_output, encoder_outputs, encoder_padding_mask):
        """

        :param decoder_output: B x tgt_dim
        :param encoder_outputs: B x src_dim
        :param encoder_padding_mask:
        :return:
        """
        attn_weights = self.score(decoder_output, encoder_outputs, encoder_padding_mask)
        context_embed = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        attn_outputs = torch.tanh(self.output_proj(torch.cat([decoder_output, context_embed], dim=2)))
        return attn_outputs, attn_weights


class CopyRNN(nn.Module):
    def __init__(self, args, vocab2id):
        super().__init__()
        src_hidden_size = args.src_hidden_size
        target_hidden_size = args.target_hidden_size
        max_oov_count = args.max_oov_count
        max_len = args.max_src_len
        embed_size = args.embed_size
        embedding = nn.Embedding(len(vocab2id), embed_size, vocab2id[PAD_WORD])
        self.encoder = CopyRnnEncoder(vocab2id, embedding, src_hidden_size, False, args.dropout)
        self.decoder = CopyRnnDecoder(vocab2id,
                                      embedding,
                                      target_hidden_size,
                                      src_hidden_size,
                                      max_len,
                                      False,
                                      args.dropout,
                                      max_oov_count)

    def forward(self, src_tokens, src_lens, prev_output_tokens, encoder_output,
                src_tokens_with_oov, oov_counts, decoder_state, hidden_state):
        if torch.cuda.is_available():
            src_tokens = src_tokens.cuda()
            src_lens = src_lens.cuda()
            prev_output_tokens = prev_output_tokens.cuda()
            src_tokens_with_oov = src_tokens_with_oov.cuda()
            oov_counts = oov_counts.cuda()
            decoder_state = decoder_state.cuda()
        if encoder_output is None:
            encoder_output = self.encoder(src_tokens, src_lens)
            hidden_state = encoder_output['encoder_hidden']

        decoder_prob, decoder_state, hidden_state = self.decoder(prev_output_tokens,
                                                                 encoder_output,
                                                                 decoder_state,
                                                                 hidden_state,
                                                                 src_tokens,
                                                                 src_tokens_with_oov,
                                                                 oov_counts)
        if torch.cuda.is_available():
            decoder_prob = decoder_prob.cpu()
        return decoder_prob, encoder_output, decoder_state, hidden_state


class CopyRnnEncoder(nn.Module):
    def __init__(self, vocab2id, embedding, hidden_size,
                 bidirectional, dropout):
        super().__init__()
        embed_dim = embedding.embedding_dim
        self.embed_dim = embed_dim
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_layers = 1
        self.pad_idx = vocab2id[PAD_WORD]

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        batch_size = len(src_tokens)
        src_embed = self.embedding(src_tokens)
        total_length = src_embed.size(1)
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(src_embed,
                                                             src_lengths.data.tolist(),
                                                             batch_first=True,
                                                             enforce_sorted=False)
        state_size = (self.num_layers, batch_size, self.hidden_size)
        h0 = src_embed.new_zeros(state_size)
        c0 = src_embed.new_zeros(state_size)
        hidden_states, (final_hiddens, final_cells) = self.lstm(packed_src_embed, (h0, c0))
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(hidden_states,
                                                            padding_value=self.pad_idx,
                                                            batch_first=True,
                                                            total_length=total_length)
        encoder_padding_mask = src_tokens.eq(self.pad_idx)
        output = {'encoder_output': hidden_states,
                  'encoder_padding_mask': encoder_padding_mask,
                  'encoder_hidden': (final_hiddens, final_cells)}
        return output


class CopyRnnDecoder(nn.Module):
    def __init__(self, vocab2id, embedding, hidden_size, src_hidden_size, max_len,
                 bidirectional, dropout, max_oov_len
                 ):
        super().__init__()
        self.vocab2id = vocab2id
        vocab_size = embedding.num_embeddings
        embed_dim = embedding.embedding_dim
        self.vocab_size = vocab_size
        self.embed_size = embed_dim
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.src_hidden_size = src_hidden_size
        self.max_len = max_len
        self.max_oov_len = max_oov_len
        self.pad_idx = vocab2id[PAD_WORD]
        self.lstm = nn.LSTM(
            input_size=embed_dim + src_hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )
        self.attn_layer = Attention(src_hidden_size, hidden_size)
        self.copy_proj = nn.Linear(src_hidden_size, hidden_size, bias=False)
        self.input_copy_proj = nn.Linear(src_hidden_size, hidden_size, bias=False)
        self.generate_proj = nn.Linear(hidden_size, self.vocab_size, bias=False)

    def forward(self, prev_output_tokens, encoder_output_dict, prev_context_state,
                prev_rnn_state,
                src_tokens, src_tokens_with_oov, oov_counts):
        """

        :param prev_output_tokens: B x 1
        :param prev_output_lens: B x 1
        :param encoder_output_dict:
        :param prev_context_state: B x TH
        :param src_tokens: B x max_len
        :param src_tokens_with_oov: B x max_len
        :param oov_counts: B
        :return:
        """
        batch_size = len(src_tokens)
        prev_output_tokens = torch.as_tensor(prev_output_tokens, dtype=torch.int64)
        if torch.cuda.is_available():
            prev_output_tokens = prev_output_tokens.cuda()

        encoder_output = encoder_output_dict['encoder_output']
        encoder_output_mask = encoder_output_dict['encoder_padding_mask']

        # mask : B x L x 1
        mask = torch.eq(prev_output_tokens.repeat(1, self.max_len), src_tokens).type_as(encoder_output)
        mask = mask.unsqueeze(2)

        # B x 1 x SH
        aggregate_weight = torch.tanh(self.input_copy_proj(torch.mul(mask, encoder_output)))
        copy_weight = torch.softmax(torch.bmm(aggregate_weight, prev_context_state.unsqueeze(2)), 1)
        # B x L x 1
        #
        copy_state = torch.sum(torch.mul(copy_weight, encoder_output), dim=1).unsqueeze(1)

        src_embed = self.embedding(prev_output_tokens)
        decoder_input = torch.cat([src_embed, copy_state], dim=2)

        rnn_output, rnn_state = self.lstm(decoder_input, prev_rnn_state)
        attn_output, attn_weights = self.attn_layer(rnn_output, encoder_output, encoder_output_mask)
        generate_logits = self.generate_proj(attn_output).squeeze(1)
        generate_oov_logits = torch.zeros(batch_size, self.max_oov_len)
        if torch.cuda.is_available():
            generate_oov_logits = generate_oov_logits.cuda()
        generate_logits = torch.cat([generate_logits, generate_oov_logits], dim=1)
        copy_logits = self.get_copy_score(encoder_output,
                                          src_tokens_with_oov,
                                          oov_counts,
                                          attn_output)
        total_logit = generate_logits + copy_logits
        total_prob = torch.softmax(total_logit, 1)

        return total_prob, attn_output.squeeze(1), rnn_state

    def get_copy_score(self, encoder_out, src_tokens_with_oov, oov_counts, decoder_output):
        copy_score = torch.bmm(torch.tanh(self.copy_proj(encoder_out)), decoder_output.permute(0, 2, 1))
        copy_score = copy_score.squeeze(2)

        copy_score_initial = self.get_copy_score_initial(oov_counts)
        copy_score_initial.scatter_(1, src_tokens_with_oov, copy_score)

        return copy_score_initial

    def get_copy_score_initial(self, oov_counts):
        copy_score_initial = []
        for oov_count in oov_counts:
            if oov_count == self.max_oov_len:
                copy_score_initial.append(torch.zeros(self.vocab_size + self.max_oov_len))
            else:
                prefix = torch.zeros(self.vocab_size + oov_count)
                suffix = torch.ones(self.max_oov_len - oov_count) * float('-inf')
                copy_score_initial.append(torch.cat([prefix, suffix]))

        copy_score_initial = torch.stack(copy_score_initial)
        if torch.cuda.is_available():
            copy_score_initial = copy_score_initial.cuda()
        return copy_score_initial
