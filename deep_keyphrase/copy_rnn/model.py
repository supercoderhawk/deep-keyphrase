# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
from deep_keyphrase.utils.vocab_loader import load_vocab


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
            attn_weights = torch.bmm(self.attn(query), key)
            attn_weights = torch.squeeze(attn_weights, -1)
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
        attn_outputs = torch.tanh(self.output_proj(torch.cat([decoder_output, context_embed])))
        return attn_outputs, attn_weights


class CopyRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        src_hidden_size = args.src_hidden_size
        target_hidden_size = args.target_hidden_size
        max_oov_count = args.max_oov_count
        vocab2id = load_vocab(args.vocab_path)
        embed_size = args.embed_size
        embedding = nn.Embedding(len(vocab2id), embed_size)
        self.encoder = CopyRnnEncoder(embedding, src_hidden_size, False, 0.5, 0.5)
        self.decoder = CopyRnnDecoder(embedding,
                                      target_hidden_size,
                                      src_hidden_size,
                                      10,
                                      False,
                                      0.5,
                                      max_oov_count)

    def forward(self, src_tokens, src_lens, prev_output_tokens, prev_output_lens, encoder_output):
        batch_size = len(src_tokens)
        if encoder_output is None:
            encoder_output = self.encoder(src_tokens, src_lens)

        decoder_state = torch.zeros(batch_size)
        decoder_prob, decoder_state = self.decoder(prev_output_tokens,
                                                   prev_output_lens,
                                                   encoder_output,
                                                   decoder_state,
                                                   src_tokens)
        return decoder_prob, encoder_output


class CopyRnnEncoder(nn.Module):
    def __init__(self, embedding, hidden_size,
                 bidirectional, dropout_in, dropout_out):
        super().__init__()
        # self.dictionary = dictionary
        vocab_size, embed_dim = embedding.size()
        # assert vocab_size == len(dictionary)
        self.embed_dim = embed_dim
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_in
        )

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        src_embed = self.embedding(src_tokens)
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(src_embed,
                                                             src_lengths.data.tolist())
        batch_size, seq_length = src_lengths.size()
        state_size = (self.num_layers, batch_size, self.hidden_size)
        h0 = src_embed.new_zeros(state_size)
        c0 = src_embed.new_zeros(state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_src_embed, (h0, c0))
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(packed_outs,
                                                            padding_value=self.padding_value)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        output = {'encoder_output': hidden_states,
                  'encoder_padding_mask': encoder_padding_mask}
        return output


class CopyRnnDecoder(nn.Module):
    def __init__(self, embedding, hidden_size, src_hidden_size, max_len,
                 bidirectional, dropout, max_oov_len

                 ):
        super().__init__()
        vocab_size, embed_size = embedding.size()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.src_hidden_size = src_hidden_size
        self.max_len = max_len
        self.max_oov_len = max_oov_len
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout
        )
        self.attn_layer = Attention(src_hidden_size, hidden_size)
        self.copy_proj = nn.Linear(src_hidden_size, hidden_size, bias=False)
        self.generate_proj = nn.Linear(hidden_size, self.vocab_size, bias=False)

    def forward(self, prev_output_tokens, prev_output_lens, encoder_output_dict, prev_hidden_state,
                src_tokens, src_tokens_with_oov, oov_counts):
        batch_size = len(src_tokens)

        prev_output_tokens = torch.as_tensor(prev_output_tokens, dtype=torch.int64)
        encoder_output = encoder_output_dict['encoder_output']
        encoder_output_mask = encoder_output_dict['encoder_padding_mask']
        mask = torch.eq(prev_output_tokens.unsqueeze(1).repeat(1, self.max_len), src_tokens)
        mask = mask.unsqueeze(1)

        # batch_idx_list = []
        # batch_val = []
        # gen = enumerate(zip(src_tokens, prev_output_tokens))
        # for batch_idx, (single_batch_src_token, single_batch_prev_token) in gen:
        #     for token_idx, token in enumerate(single_batch_src_token):
        #         if token == single_batch_prev_token[token_idx]:
        #             batch_idx_list.append((batch_idx, token_idx, token_idx))
        #             batch_val.append(1.0)
        # batch_idx_list = torch.LongTensor(batch_idx_list)
        # batch_val = torch.FloatTensor(batch_val)
        # src_map_size = torch.Size([batch_size, self.max_len, self.max_len])
        # src_map = torch.sparse.FloatTensor(batch_idx_list.t(), batch_val, src_map_size).to_dense()
        aggregate_weight = torch.tanh(self.copy_proj(torch.bmm(mask, encoder_output)))
        copy_weight = torch.softmax(torch.bmm(aggregate_weight, prev_hidden_state), 1).unsqueeze(0)
        copy_state = torch.bmm(copy_weight, encoder_output)

        src_embed = self.embedding(prev_output_tokens)
        decoder_input = torch.cat([src_embed, copy_state], dim=1)

        state_size = (1, batch_size, self.hidden_size)
        h0 = src_embed.new_zeros(state_size)
        c0 = src_embed.new_zeros(state_size)
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(decoder_input,
                                                             prev_output_lens)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_src_embed, (h0, c0))
        hidden_state, _ = nn.utils.rnn.pad_packed_sequence(packed_outs,
                                                           padding_value=self.padding_value)
        attn_output, attn_weights = self.attn_layer(hidden_state, encoder_output, encoder_output_mask)
        generate_logits = self.generate_proj(attn_output)

        copy_score = self.get_copy_score(batch_size, encoder_output, src_tokens_with_oov, oov_counts)
        copy_logits = torch.bmm(copy_score, attn_output)
        total_prob = torch.softmax(generate_logits + copy_logits, 1)

        return total_prob, attn_output

    def get_copy_score(self, batch_size, encoder_out, src_tokens_with_oov, oov_counts):
        batch_idx_list = []
        batch_val = []
        for batch_idx, single_batch_src_token in enumerate(src_tokens_with_oov):
            for token_idx, token in enumerate(single_batch_src_token):
                batch_idx_list.append((batch_idx, token, token_idx))
                batch_val.append(1.0)
        batch_idx_list = torch.LongTensor(batch_idx_list)
        batch_val = torch.FloatTensor(batch_val)
        src_map_size = torch.Size([batch_size, self.vocab_size + self.max_oov_len, self.max_len])
        src_map = torch.sparse.FloatTensor(batch_idx_list.t(), batch_val, src_map_size).to_dense()
        if oov_counts < self.max_oov_len:
            zero_tensor = torch.zeros(batch_size, self.vocab_size + oov_counts)
            inf_tensor = torch.as_tensor([float('-inf') * self.max_oov_len - oov_counts], torch.float)
            copy_score_initial = torch.cat([zero_tensor, inf_tensor], dim=1)
        else:
            copy_score_initial = torch.zeros(batch_size, self.vocab_size + self.max_oov_len)

        copy_score = torch.tanh(self.copy_proj(torch.bmm(src_map, encoder_out)))
        copy_score += copy_score_initial

        return copy_score
