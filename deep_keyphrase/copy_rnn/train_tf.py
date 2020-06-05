# -*- coding: UTF-8 -*-
import argparse
import tensorflow as tf
from deep_keyphrase.dataloader import *
from deep_keyphrase.utils.vocab_loader import load_vocab
from deep_keyphrase.copy_rnn.model_tf import CopyRnnTF


# from deep_keyphrase.utils.constants import *

class CopyRnnTrainerTF(object):
    def __init__(self):
        self.args = self.parse_args()
        self.vocab2id = load_vocab(self.args.vocab_path)
        self.writer = tf.summary.create_file_writer('logs')

    def train(self):
        model = CopyRnnTF(self.args, len(self.vocab2id))
        dataloader = KeyphraseDataLoader(data_source=self.args.train_filename,
                                         vocab2id=self.vocab2id,
                                         mode='train',
                                         args=self.args)
        optimizer = tf.keras.optimizers.Adam()
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        @tf.function
        def train_step(x, x_with_oov, x_len, target):
            with tf.GradientTape() as tape:
                loss = 0
                # print('x_oov',x_with_oov.shape)
                # print('len',x_len.shape)
                probs, enc_output, prev_h,prev_c = model(x, x_with_oov, x_len, None,
                                                      target[:, :-1], None, None)
                dec_len = probs.shape[1]
                for dec_idx in range(dec_len):
                    loss += loss_func(target[:, dec_idx], probs[:, dec_idx])
                loss /= target.shape[1]
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            return loss
            # tf.print(loss)

        for epoch in range(self.args.epochs):
            for batch_idx, batch in enumerate(dataloader):
                # print('x_oov_input', batch[TOKENS_OOV].shape)
                loss = train_step(batch[TOKENS], batch[TOKENS_OOV], batch[TOKENS_LENS], batch[TARGET])
                with self.writer.as_default():
                    tf.summary.scalar('loss', loss, step=batch_idx)
                # self.writer.scalar('loss', loss, step=batch_idx)

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        # train and evaluation parameter
        parser.add_argument("-exp_name", required=True, type=str, help='')
        parser.add_argument("-train_filename", required=True, type=str, help='')
        parser.add_argument("-valid_filename", required=True, type=str, help='')
        parser.add_argument("-test_filename", required=True, type=str, help='')
        parser.add_argument("-dest_base_dir", required=True, type=str, help='')
        parser.add_argument("-vocab_path", required=True, type=str, help='')
        parser.add_argument("-vocab_size", type=int, default=500000, help='')
        parser.add_argument("-train_from", default='', type=str, help='')
        parser.add_argument("-token_field", default='tokens', type=str, help='')
        parser.add_argument("-keyphrase_field", default='keyphrases', type=str, help='')
        parser.add_argument("-auto_regressive", action='store_true', help='')
        parser.add_argument("-epochs", type=int, default=10, help='')
        parser.add_argument("-batch_size", type=int, default=64, help='')
        parser.add_argument("-learning_rate", type=float, default=1e-4, help='')
        parser.add_argument("-eval_batch_size", type=int, default=50, help='')
        parser.add_argument("-dropout", type=float, default=0.0, help='')
        parser.add_argument("-grad_norm", type=float, default=0.0, help='')
        parser.add_argument("-max_grad", type=float, default=5.0, help='')
        parser.add_argument("-shuffle", action='store_true', help='')
        parser.add_argument("-teacher_forcing", action='store_true', help='')
        parser.add_argument("-beam_size", type=float, default=50, help='')
        parser.add_argument('-tensorboard_dir', type=str, default='', help='')
        parser.add_argument('-logfile', type=str, default='train_log.log', help='')
        parser.add_argument('-save_model_step', type=int, default=5000, help='')
        parser.add_argument('-early_stop_tolerance', type=int, default=100, help='')
        parser.add_argument('-train_parallel', action='store_true', help='')
        parser.add_argument('-schedule_lr', action='store_true', help='')
        parser.add_argument('-schedule_step', type=int, default=10000, help='')
        parser.add_argument('-schedule_gamma', type=float, default=0.1, help='')
        parser.add_argument('-processed', action='store_true', help='')
        parser.add_argument('-prefetch', action='store_true', help='')
        parser.add_argument('-backend', type=str, default='tf', help='')
        parser.add_argument('-lazy_loading', action='store_true', help='')

        # model specific parameter
        parser.add_argument("-embed_dim", type=int, default=200, help='')
        parser.add_argument("-max_oov_count", type=int, default=100, help='')
        parser.add_argument("-max_src_len", type=int, default=1500, help='')
        parser.add_argument("-max_target_len", type=int, default=8, help='')
        parser.add_argument("-encoder_hidden_size", type=int, default=100, help='')
        parser.add_argument("-decoder_hidden_size", type=int, default=100, help='')
        parser.add_argument('-src_num_layers', type=int, default=1, help='')
        parser.add_argument('-target_num_layers', type=int, default=1, help='')
        parser.add_argument("-attention_mode", type=str, default='general',
                            choices=['general', 'dot', 'concat'], help='')
        parser.add_argument("-bidirectional", action='store_true', help='')
        parser.add_argument("-copy_net", action='store_true', help='')
        parser.add_argument("-input_feeding", action='store_true', help='')

        args = parser.parse_args(args)
        return args


if __name__ == '__main__':
    CopyRnnTrainerTF().train()
