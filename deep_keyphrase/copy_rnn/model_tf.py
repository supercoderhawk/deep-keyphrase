# -*- coding: UTF-8 -*-
import argparse
import tensorflow as tf
from ..dataloader import UNK_WORD, BOS_WORD


def mask_fill(t, mask, num):
    """

    :param t: input tensor
    :param mask: mask indicator True for keeping value and False for mask
    :param num: num to be fill in mask tensor
    :return:
    """
    t_dtype = t.dtype
    mask = tf.cast(mask, dtype=t_dtype)
    neg_mask = 1 - mask
    filled_t = t * mask + neg_mask * num
    return filled_t


class Attention(tf.keras.layers.Layer):
    def __init__(self, encoder_dim, decoder_dim, score_mode='general'):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.score_mode = score_mode
        self.permuate_1_2 = tf.keras.layers.Permute((2, 1))
        if self.score_mode == 'general':
            self.attn = tf.keras.layers.Dense(self.encoder_dim, use_bias=False)

        self.output_layer = tf.keras.layers.Dense(self.decoder_dim)
        self.concat_layer = tf.keras.layers.Concatenate()

    def score(self, query, key, mask):
        if self.score_mode == 'general':
            attn_weights = tf.matmul(self.attn(query), self.permuate_1_2(key))
        elif self.score_mode == 'concat':
            pass
        elif self.score_mode == 'dot':
            attn_weights = tf.matmul(query, self.permuate_1_2(key))

        mask = tf.repeat(tf.expand_dims(mask, 1), repeats=query.shape[1], axis=1)
        attn_weights = mask_fill(attn_weights, mask, float('-inf') / 2)
        attn_weights = tf.nn.softmax(attn_weights, axis=2)
        return attn_weights

    def call(self, decoder_output, encoder_output, enc_mask):
        attn_weights = self.score(decoder_output, encoder_output, enc_mask)
        context_embed = tf.matmul(attn_weights, encoder_output)
        attn_output = tf.tanh(self.output_layer(self.concat_layer([context_embed, decoder_output])))
        return attn_output


class CopyRnnTF(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, vocab2id):
        super().__init__()
        self.args = args
        self.vocab_size = len(vocab2id)
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, args.embed_dim)
        self.encoder = Encoder(args, self.embedding)
        self.decoder = Decoder(args, self.embedding)
        self.max_target_len = self.args.max_target_len
        self.total_vocab_size = self.vocab_size + args.max_oov_count
        self.encoder2decoder_state = tf.keras.layers.Dense(args.decoder_hidden_size)
        self.encoder2decoder_cell = tf.keras.layers.Dense(args.decoder_hidden_size)
        self.beam_size = args.beam_size
        self.unk_idx = vocab2id[UNK_WORD]
        self.bos_idx = vocab2id[BOS_WORD]

    def call(self, x, x_with_oov, x_len, enc_output, dec_x, prev_h, prev_c):
        if enc_output is None:
            enc_output, prev_h, prev_c = self.encoder(x, x_len)
            prev_h = self.encoder2decoder_state(prev_h)
            prev_c = self.encoder2decoder_state(prev_c)

        probs, prev_h, prev_c = self.decoder(dec_x, x, x_with_oov, x_len, enc_output, prev_h, prev_c)

        return probs, enc_output, prev_h, prev_c

    @tf.function
    def beam_search(self, x, x_with_oov, x_len):
        batch_size = x.shape[0]
        beam_batch_size = self.beam_size * batch_size
        prev_output_tokens = tf.ones([batch_size, 1], dtype=tf.int64) * self.bos_idx

        probs, enc_output, prev_h, prev_c = self.call(x, x_with_oov, x_len, None,
                                                      prev_output_tokens, None, None)
        probs = tf.squeeze(probs, axis=1)
        prev_best_probs, prev_best_index = tf.math.top_k(probs, k=self.beam_size)

        prev_h = tf.repeat(prev_h, self.beam_size, axis=0)
        prev_c = tf.repeat(prev_c, self.beam_size, axis=0)
        enc_output = tf.repeat(enc_output, self.beam_size, axis=0)

        result_sequences = prev_best_index

        prev_best_index = mask_fill(prev_best_index, prev_best_index >= self.vocab_size, self.unk_idx)
        prev_best_index = tf.reshape(prev_best_index, [beam_batch_size, -1])
        x = tf.repeat(x, repeats=self.beam_size, axis=0)

        x_with_oov = tf.repeat(x_with_oov, repeats=self.beam_size, axis=0)
        x_len = tf.repeat(x_len, repeats=self.beam_size, axis=0)

        for target_idx in range(1, self.max_target_len):
            probs, enc_output, prev_h, prev_c = self.call(x, x_with_oov, x_len, enc_output,
                                                          prev_best_index,
                                                          prev_h, prev_c)
            probs = tf.squeeze(probs, axis=1)
            accumulated_probs = tf.reshape(prev_best_probs, [beam_batch_size, -1])
            accumulated_probs = tf.repeat(accumulated_probs, repeats=self.total_vocab_size, axis=1)
            accumulated_probs += probs
            accumulated_probs = tf.reshape(accumulated_probs, [batch_size, -1])
            beam_search_best_probs, top_token_index = tf.math.top_k(-accumulated_probs, self.beam_size)

            select_idx_factor = tf.range(0, batch_size) * self.beam_size
            select_idx_factor = tf.repeat(tf.expand_dims(select_idx_factor, axis=1), self.beam_size, axis=1)
            state_select_idx = tf.reshape(top_token_index, [beam_batch_size]) // probs.shape[1]
            state_select_idx += tf.reshape(select_idx_factor, [beam_batch_size])

            prev_best_index = top_token_index % probs.shape[1]
            prev_h = tf.gather(prev_h, state_select_idx, axis=0)
            prev_c = tf.gather(prev_c, state_select_idx, axis=0)

            result_sequences = tf.reshape(result_sequences, [beam_batch_size, -1])
            result_sequences = tf.gather(result_sequences, state_select_idx, axis=0)
            result_sequences = tf.reshape(result_sequences, [batch_size, self.beam_size, -1])

            result_sequences = tf.concat([result_sequences, tf.expand_dims(prev_best_index, axis=2)], axis=2)

            prev_best_index = tf.reshape(prev_best_index, [beam_batch_size, -1])
            prev_best_index = mask_fill(prev_best_index, prev_best_index >= self.vocab_size, self.unk_idx)

        return result_sequences


class Encoder(tf.keras.layers.Layer):
    def __init__(self, args, embedding):
        super().__init__()
        self.args = args
        self.embedding = embedding
        self.lstm = tf.keras.layers.LSTM(self.args.encoder_hidden_size,
                                         return_state=True, return_sequences=True)
        if args.bidirectional:
            self.lstm = tf.keras.layers.Bidirectional(self.lstm)

        self.max_dec = self.args.max_src_len

    def call(self, x, x_len):
        embed_x = self.embedding(x)
        mask = tf.sequence_mask(x_len, maxlen=self.max_dec)
        if self.args.bidirectional:
            lstm_output, state_fw_h, state_fw_c, state_bw_h, state_bw_c = self.lstm(embed_x, mask=mask)
            state_h = tf.concat([state_fw_h, state_bw_h], axis=1)
            state_c = tf.concat([state_fw_c, state_bw_c], axis=1)
        else:
            lstm_output, state_h, state_c = self.lstm(embed_x)
        return lstm_output, state_h, state_c


class Decoder(tf.keras.layers.Layer):
    def __init__(self, args, embedding):
        super().__init__()
        self.args = args
        self.embedding = embedding
        self.vocab_size = self.embedding.input_dim
        self.max_oov_count = self.args.max_oov_count
        self.max_dec = self.args.max_src_len
        self.max_enc = self.args.max_src_len
        self.lstm = tf.keras.layers.LSTM(self.args.decoder_hidden_size,
                                         return_state=True, return_sequences=True)
        if self.args.bidirectional:
            enc_hidden_size = self.args.encoder_hidden_size * 2
        else:
            enc_hidden_size = self.args.encoder_hidden_size
        self.attention = Attention(enc_hidden_size,
                                   self.args.decoder_hidden_size)
        self.generate_layer = tf.keras.layers.Dense(self.vocab_size, use_bias=False)
        self.concat_layer = tf.keras.layers.Concatenate()
        self.copy_layer = tf.keras.layers.Dense(self.args.decoder_hidden_size)
        self.permuate_1_2 = tf.keras.layers.Permute((2, 1))

    def call(self, dec_x, enc_x, enc_x_with_oov, enc_len, enc_output, enc_h, enc_c):
        return self.call_one_pass(dec_x, enc_x, enc_x_with_oov, enc_len, enc_output, enc_h, enc_c)

    def call_one_pass(self, dec_x, enc_x: tf.Tensor, enc_x_with_oov, enc_len, enc_output, enc_h, enc_c):
        """
        use for teacher forcing during training
        :return:
        """
        embed_dec_x = self.embedding(dec_x)
        mask = tf.sequence_mask(enc_len, maxlen=self.max_dec)
        hidden_states, state_h, state_c = self.lstm(embed_dec_x, (enc_h, enc_c))
        attn_output = self.attention(hidden_states, enc_output, mask)
        generation_logits = tf.exp(self.generate_layer(attn_output))

        generation_logits = tf.pad(generation_logits, [[0, 0], [0, 0], [0, self.max_oov_count]],
                                   constant_values=1e-10)

        copy_logits = self.get_copy_score(enc_output, enc_x_with_oov, attn_output)
        total_logits = generation_logits + copy_logits
        total_prob = total_logits / tf.reduce_sum(total_logits, axis=1, keepdims=True)
        # total_prob = tf.math.log1p(total_prob)
        return total_prob, state_h, state_c

    def call_auto_regressive(self, x, prev_state):
        embed_x = self.embedding(x)
        hidden_states, state_h, state_c = self.lstm(embed_x, prev_state)

    def get_copy_score(self, src_output, x_with_oov: tf.Tensor, tgt_output):
        batch_size = x_with_oov.shape[0]
        total_vocab_size = self.vocab_size + self.max_oov_count
        dec_len = tgt_output.shape[1]

        tgt_output = self.permuate_1_2(tgt_output)
        copy_score_in_seq = tf.matmul(tf.tanh(self.copy_layer(src_output)), tgt_output)

        copy_score_in_seq = self.permuate_1_2(copy_score_in_seq)
        copy_score_in_seq = tf.exp(copy_score_in_seq)

        batch_idx, src_idx, _ = tf.meshgrid(tf.range(batch_size, dtype=tf.int64),
                                            tf.range(dec_len, dtype=tf.int64),
                                            tf.range(self.max_dec, dtype=tf.int64), indexing="ij")
        # batch_idx = tf.transpose(tf.broadcast_to(tf.range(batch_size, dtype=tf.int64),
        #                                          [self.max_dec, dec_len, batch_size]))
        # src_idx = tf.broadcast_to(tf.range(dec_len, dtype=tf.int64), [batch_size, dec_len])
        # src_idx = tf.repeat(tf.expand_dims(src_idx, axis=2), repeats=self.max_dec, axis=2)
        x_with_oov = tf.repeat(tf.expand_dims(x_with_oov, axis=1), repeats=dec_len, axis=1)
        # Create final indices
        score_idx = tf.stack([batch_idx, src_idx, x_with_oov], axis=-1)
        # Output shape
        to_shape = [batch_size, dec_len, total_vocab_size]
        # Get scattered tensor
        copy_score_in_vocab = tf.scatter_nd(score_idx, copy_score_in_seq, to_shape)
        return copy_score_in_vocab
