# -*- coding: UTF-8 -*-
import argparse
import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self, encoder_dim, decoder_dim, dec_len, score_mode='general'):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.dec_len = dec_len
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

        mask = tf.repeat(tf.expand_dims(mask, 1), repeats=self.dec_len, axis=1)
        attn_weights = tf.cast(mask, tf.float32) * attn_weights
        attn_weights = tf.nn.softmax(attn_weights, axis=2)
        return attn_weights

    def call(self, decoder_output, encoder_output, enc_mask):
        attn_weights = self.score(decoder_output, encoder_output, enc_mask)
        context_embed = tf.matmul(attn_weights, encoder_output)
        attn_output = tf.tanh(self.output_layer(self.concat_layer([context_embed, decoder_output])))
        return attn_output


class CopyRnnTF(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, vocab_size):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, args.embed_dim)
        self.encoder = Encoder(args, self.embedding)
        self.decoder = Decoder(args, self.embedding)

    def call(self, x, x_with_oov, x_len, enc_output, dec_x, prev_h, prev_c):
        if enc_output is None:
            enc_output, prev_h, prev_c = self.encoder(x, x_len)

        probs, prev_h, prev_c = self.decoder(dec_x, x, x_with_oov, x_len, enc_output, prev_h, prev_c)

        return probs, enc_output, prev_h, prev_c


class Encoder(tf.keras.layers.Layer):
    def __init__(self, args, embedding):
        super().__init__()
        self.args = args
        self.embedding = embedding
        self.lstm = tf.keras.layers.LSTM(self.args.encoder_hidden_size,
                                         return_state=True, return_sequences=True)
        self.max_dec = self.args.max_src_len

    def call(self, x, x_len):
        embed_x = self.embedding(x)
        mask = tf.sequence_mask(x_len, maxlen=self.max_dec)
        lstm_output, state_h, state_c = self.lstm(embed_x, mask=mask)
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
        self.attention = Attention(self.args.encoder_hidden_size,
                                   self.args.decoder_hidden_size,
                                   self.args.max_target_len)
        self.generate_layer = tf.keras.layers.Dense(self.vocab_size, use_bias=False)
        self.concat_layer = tf.keras.layers.Concatenate()
        self.copy_layer = tf.keras.layers.Dense(self.args.decoder_hidden_size)
        self.permuate_1_2 = tf.keras.layers.Permute((2, 1))
        i1, i2 = tf.meshgrid(tf.range(self.args.batch_size, dtype=tf.int64),
                             tf.range(self.args.max_target_len, dtype=tf.int64), indexing="ij")
        self.i1 = tf.tile(i1[:, :, tf.newaxis], [1, 1, self.args.max_src_len])
        self.i2 = tf.tile(i2[:, :, tf.newaxis], [1, 1, self.args.max_src_len])

    def call(self, dec_x, enc_x, enc_x_with_oov, enc_len, enc_output, enc_h, enc_c):
        return self.call_one_pass(dec_x, enc_x, enc_x_with_oov, enc_len, enc_output, enc_h, enc_c)

    def call_one_pass(self, dec_x, enc_x: tf.Tensor, enc_x_with_oov, enc_len, enc_output, enc_h, enc_c):
        """
        use for teacher forcing during training
        :return:
        """
        # batch_size = enc_x.shape[0]
        embed_dec_x = self.embedding(dec_x)
        mask = tf.sequence_mask(enc_len, maxlen=self.max_dec)
        # print(mask.shape)
        hidden_states, state_h, state_c = self.lstm(embed_dec_x, (enc_h, enc_c))
        attn_output = self.attention(hidden_states, enc_output, mask)
        generation_logits = tf.exp(self.generate_layer(attn_output))

        generation_logits = tf.pad(generation_logits, [[0, 0], [0, 0], [0, self.max_oov_count]],
                                   constant_values=1e-10)

        copy_logits = self.get_copy_score(enc_output, enc_x_with_oov, attn_output)
        total_logits = generation_logits + copy_logits
        total_prob = total_logits / tf.reduce_sum(total_logits, axis=1, keepdims=True)
        total_prob = tf.math.log1p(total_prob)
        return total_prob, state_h, state_c

    def call_auto_regressive(self, x, prev_state):
        embed_x = self.embedding(x)
        hidden_states, state_h, state_c = self.lstm(embed_x, prev_state)

    def get_copy_score(self, src_output, x_with_oov: tf.Tensor, tgt_output):
        batch_size = x_with_oov.shape[0]
        total_vocab_size = self.vocab_size + self.max_oov_count
        dec_len = tgt_output.shape[1]
        enc_len = src_output.shape[1]

        tgt_output = self.permuate_1_2(tgt_output)
        copy_score_in_seq = tf.matmul(tf.tanh(self.copy_layer(src_output)), tgt_output)

        copy_score_in_seq = self.permuate_1_2(copy_score_in_seq)
        copy_score_in_seq = tf.exp(copy_score_in_seq)

        i1, i2 = tf.meshgrid(tf.range(batch_size, dtype=tf.int64),
                             tf.range(self.max_dec, dtype=tf.int64), indexing="ij")

        # Create final indices
        idx = tf.stack([i1, i2, x_with_oov], axis=-1)
        idx = tf.expand_dims(idx, axis=1)
        idx = tf.repeat(idx, repeats=dec_len, axis=1)
        # Output shape
        to_shape = [batch_size, dec_len, total_vocab_size]
        # Get scattered tensor
        copy_score_in_vocab = tf.scatter_nd(idx, copy_score_in_seq, to_shape)
        return copy_score_in_vocab
