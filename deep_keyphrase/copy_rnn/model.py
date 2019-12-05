# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_keyphrase.utils.constants import *
from deep_keyphrase.dataloader import TOKENS, TOKENS_OOV, TOKENS_LENS, OOV_COUNT


class Attention(nn.Module):
    """
    implement attention mechanism
    """

    def __init__(self, input_dim, output_dim, score_mode='general'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.score_mode = score_mode
        if self.score_mode == 'general':
            self.attn = nn.Linear(self.output_dim, self.input_dim, bias=False)
        self.output_proj = nn.Linear(self.input_dim + self.output_dim, self.output_dim)

    def score(self, query, key, encoder_padding_mask):
        if self.score_mode == 'general':
            attn_weights = torch.bmm(self.attn(query), key.permute(0, 2, 1))
            attn_weights = torch.squeeze(attn_weights, 1)
        else:
            raise ValueError('value error')
        # mask input padding to -Inf, they will be zero after softmax.
        if encoder_padding_mask is not None:
            attn_weights.masked_fill_(encoder_padding_mask, float('-inf'))
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
        attn_outputs = torch.tanh(self.output_proj(torch.cat([context_embed, decoder_output], dim=2)))
        return attn_outputs, attn_weights


class CopyRNN(nn.Module):
    """
    Abbreviation Noting:
    B: batch size
    L: source max len
    SH: source hidden size
    TH: target hidden size
    GV: generative vocab size
    V: total vocab size (generative vocab size and copy vocab size)
    """

    def __init__(self, args, vocab2id):
        super().__init__()
        src_hidden_size = args.src_hidden_size
        target_hidden_size = args.target_hidden_size
        embed_size = args.embed_size
        embedding = nn.Embedding(len(vocab2id), embed_size, vocab2id[PAD_WORD])
        self.encoder = CopyRnnEncoder(vocab2id=vocab2id,
                                      embedding=embedding,
                                      hidden_size=src_hidden_size,
                                      bidirectional=args.bidirectional,
                                      dropout=args.dropout)
        if args.bidirectional:
            decoder_src_hidden_size = 2 * src_hidden_size
        else:
            decoder_src_hidden_size = src_hidden_size
        self.decoder = CopyRnnDecoder(vocab2id=vocab2id, embedding=embedding, args=args)
        if decoder_src_hidden_size != target_hidden_size:
            self.encoder2decoder_state = nn.Linear(decoder_src_hidden_size, target_hidden_size)
            self.encoder2decoder_cell = nn.Linear(decoder_src_hidden_size, target_hidden_size)

    def forward(self, src_dict, prev_output_tokens, encoder_output_dict,
                prev_decoder_state, prev_hidden_state):
        """

        :param src_dict:
        :param prev_output_tokens:
        :param encoder_output_dict:
        :param prev_decoder_state:
        :param prev_hidden_state:
        :return:
        """
        if torch.cuda.is_available():
            src_dict[TOKENS] = src_dict[TOKENS].cuda()
            src_dict[TOKENS_LENS] = src_dict[TOKENS_LENS].cuda()
            src_dict[TOKENS_OOV] = src_dict[TOKENS_OOV].cuda()
            src_dict[OOV_COUNT] = src_dict[OOV_COUNT].cuda()
            prev_output_tokens = prev_output_tokens.cuda()
            prev_decoder_state = prev_decoder_state.cuda()
        if encoder_output_dict is None:
            encoder_output_dict = self.encoder(src_dict)
            prev_hidden_state = encoder_output_dict['encoder_hidden']
            prev_hidden_state[0] = self.encoder2decoder_state(prev_hidden_state[0])
            prev_hidden_state[1] = self.encoder2decoder_cell(prev_hidden_state[1])

        decoder_prob, prev_decoder_state, prev_hidden_state = self.decoder(
            src_dict=src_dict,
            prev_output_tokens=prev_output_tokens,
            encoder_output_dict=encoder_output_dict,
            prev_context_state=prev_decoder_state,
            prev_rnn_state=prev_hidden_state)
        return decoder_prob, encoder_output_dict, prev_decoder_state, prev_hidden_state


class CopyRnnEncoder(nn.Module):
    def __init__(self, vocab2id, embedding, hidden_size,
                 bidirectional, dropout):
        super().__init__()
        embed_dim = embedding.embedding_dim
        self.embed_dim = embed_dim
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = 1
        self.pad_idx = vocab2id[PAD_WORD]
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
            batch_first=True)

    def forward(self, src_dict):
        """

        :param src_dict:
        :return:
        """
        src_tokens = src_dict[TOKENS]
        src_lengths = src_dict[TOKENS_LENS]
        batch_size = len(src_tokens)
        src_embed = self.embedding(src_tokens)
        src_embed = F.dropout(src_embed, p=self.dropout, training=self.training)

        total_length = src_embed.size(1)
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(src_embed,
                                                             src_lengths,
                                                             batch_first=True)
        state_size = [self.num_layers, batch_size, self.hidden_size]
        if self.bidirectional:
            state_size[0] *= 2
        h0 = src_embed.new_zeros(state_size)
        c0 = src_embed.new_zeros(state_size)
        hidden_states, (final_hiddens, final_cells) = self.lstm(packed_src_embed, (h0, c0))
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(hidden_states,
                                                            padding_value=self.pad_idx,
                                                            batch_first=True,
                                                            total_length=total_length)
        encoder_padding_mask = src_tokens.eq(self.pad_idx)
        if self.bidirectional:
            final_hiddens = torch.cat((final_hiddens[0], final_hiddens[1]), dim=1).unsqueeze(0)
            final_cells = torch.cat((final_cells[0], final_cells[1]), dim=1).unsqueeze(0)
        output = {'encoder_output': hidden_states,
                  'encoder_padding_mask': encoder_padding_mask,
                  'encoder_hidden': [final_hiddens, final_cells]}
        return output


class CopyRnnDecoder(nn.Module):
    def __init__(self, vocab2id, embedding, args):
        super().__init__()
        self.vocab2id = vocab2id
        vocab_size = embedding.num_embeddings
        embed_dim = embedding.embedding_dim
        self.vocab_size = vocab_size
        self.embed_size = embed_dim
        self.embedding = embedding
        self.target_hidden_size = args.target_hidden_size
        if args.bidirectional:
            self.src_hidden_size = args.src_hidden_size * 2
        else:
            self.src_hidden_size = args.src_hidden_size
        self.max_src_len = args.max_src_len
        self.max_oov_count = args.max_oov_count
        self.dropout = args.dropout
        self.pad_idx = vocab2id[PAD_WORD]
        self.is_copy = args.copy_net
        self.input_feeding = args.input_feeding

        if self.is_copy:
            decoder_input_size = embed_dim + self.src_hidden_size
        else:
            decoder_input_size = embed_dim

        if args.input_feeding:
            decoder_input_size += self.target_hidden_size

        self.lstm = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=self.target_hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.attn_layer = Attention(self.src_hidden_size, self.target_hidden_size)
        self.copy_proj = nn.Linear(self.src_hidden_size, self.target_hidden_size, bias=False)
        self.input_copy_proj = nn.Linear(self.src_hidden_size, self.target_hidden_size, bias=False)
        self.generate_proj = nn.Linear(self.target_hidden_size, self.vocab_size, bias=False)

    def forward(self, prev_output_tokens, encoder_output_dict, prev_context_state,
                prev_rnn_state, src_dict):
        """

        :param prev_output_tokens: B x 1
        :param encoder_output_dict:
        :param prev_context_state: B x TH
        :param prev_rnn_state:
        :param src_dict:
        :return:
        """
        if self.is_copy:
            output = self.forward_copyrnn(encoder_output_dict=encoder_output_dict,
                                          prev_context_state=prev_context_state,
                                          prev_output_tokens=prev_output_tokens,
                                          prev_rnn_state=prev_rnn_state,
                                          src_dict=src_dict)
        else:
            output = self.forward_rnn(encoder_output_dict=encoder_output_dict,
                                      prev_output_tokens=prev_output_tokens,
                                      prev_rnn_state=prev_rnn_state,
                                      prev_context_state=prev_context_state)
        return output

    def forward_copyrnn(self,
                        encoder_output_dict,
                        prev_context_state,
                        prev_output_tokens,
                        prev_rnn_state,
                        src_dict):
        """

        :param encoder_output_dict:
        :param prev_context_state:
        :param prev_output_tokens:
        :param prev_rnn_state:
        :param src_dict:
        :return:
        """
        src_tokens = src_dict[TOKENS]
        src_tokens_with_oov = src_dict[TOKENS_OOV]
        batch_size = len(src_tokens)
        prev_output_tokens = torch.as_tensor(prev_output_tokens, dtype=torch.int64)
        if torch.cuda.is_available():
            prev_output_tokens = prev_output_tokens.cuda()

        encoder_output = encoder_output_dict['encoder_output']
        encoder_output_mask = encoder_output_dict['encoder_padding_mask']
        # B x 1 x L
        copy_state = self.get_attn_read_input(encoder_output,
                                              prev_context_state,
                                              prev_output_tokens,
                                              src_tokens_with_oov)

        # map copied oov tokens to OOV idx to avoid embedding lookup error
        prev_output_tokens[prev_output_tokens >= self.vocab_size] = self.vocab2id[UNK_WORD]
        src_embed = self.embedding(prev_output_tokens)
        if self.input_feeding:
            prev_context_state = prev_context_state.unsqueeze(1)
            decoder_input = torch.cat([src_embed, prev_context_state, copy_state], dim=2)
        else:
            decoder_input = torch.cat([src_embed, copy_state], dim=2)
        decoder_input = F.dropout(decoder_input, p=self.dropout, training=self.training)
        rnn_output, rnn_state = self.lstm(decoder_input, prev_rnn_state)
        rnn_state = list(rnn_state)
        # attn_output is the final hidden state of decoder layer
        # attn_output B x 1 x TH
        attn_output, attn_weights = self.attn_layer(rnn_output, encoder_output, encoder_output_mask)
        generate_logits = self.generate_proj(attn_output).squeeze(1)
        generate_oov_logits = torch.zeros(batch_size, self.max_oov_count)
        if torch.cuda.is_available():
            generate_oov_logits = generate_oov_logits.cuda()
        generate_logits = torch.cat([generate_logits, generate_oov_logits], dim=1)
        copy_logits = self.get_copy_score(encoder_output,
                                          src_tokens_with_oov,
                                          attn_output,
                                          encoder_output_mask)
        total_logit = torch.exp(generate_logits) + copy_logits
        total_prob = total_logit / torch.sum(total_logit, 1).unsqueeze(1)
        total_prob = torch.log(total_prob)
        return total_prob, attn_output.squeeze(1), rnn_state

    def forward_rnn(self, encoder_output_dict, prev_output_tokens, prev_rnn_state, prev_context_state):
        encoder_output = encoder_output_dict['encoder_output']
        encoder_output_mask = encoder_output_dict['encoder_padding_mask']
        src_embed = self.embedding(prev_output_tokens)
        if self.input_feeding:
            prev_context_state = prev_context_state.unsqueeze(1)
            decoder_input = torch.cat([src_embed, prev_context_state], dim=2)
        else:
            decoder_input = src_embed
        rnn_output, rnn_state = self.lstm(decoder_input, prev_rnn_state)
        rnn_state = list(rnn_state)
        attn_output, attn_weights = self.attn_layer(rnn_output, encoder_output, encoder_output_mask)
        probs = torch.log_softmax(self.generate_proj(attn_output).squeeze(1), 1)
        return probs, attn_output.squeeze(1), rnn_state

    def get_attn_read_input(self, encoder_output, prev_context_state,
                            prev_output_tokens, src_tokens_with_oov):
        """
        build CopyNet decoder input of "attentive read" part.
        :param encoder_output:
        :param prev_context_state:
        :param prev_output_tokens:
        :param src_tokens_with_oov:
        :return:
        """
        # mask : B x L x 1
        mask_bool = torch.eq(prev_output_tokens.repeat(1, self.max_src_len),
                             src_tokens_with_oov).unsqueeze(2)
        mask = mask_bool.type_as(encoder_output)
        # B x L x SH
        aggregate_weight = torch.tanh(self.input_copy_proj(torch.mul(mask, encoder_output)))
        # when all prev_tokens are not in src_tokens, don't execute mask -inf to avoid nan result in softmax
        no_zero_mask = ((mask != 0).sum(dim=1) != 0).repeat(1, self.max_src_len).unsqueeze(2)
        input_copy_logit_mask = no_zero_mask * mask_bool
        input_copy_logit = torch.bmm(aggregate_weight, prev_context_state.unsqueeze(2))
        input_copy_logit.masked_fill_(input_copy_logit_mask, float('-inf'))
        input_copy_weight = torch.softmax(input_copy_logit.squeeze(2), 1)
        # B x 1 x SH
        copy_state = torch.bmm(input_copy_weight.unsqueeze(1), encoder_output)
        return copy_state

    def get_copy_score(self, encoder_out, src_tokens_with_oov, decoder_output, encoder_output_mask):
        """

        :param encoder_out:
        :param src_tokens_with_oov:
        :param decoder_output:
        :param encoder_output_mask:
        :return:
        """
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
