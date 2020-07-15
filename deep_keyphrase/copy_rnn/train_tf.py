# -*- coding: UTF-8 -*-
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import argparse
import tensorflow as tf
from pysenal import write_jsonline, append_jsonlines, get_logger, write_json
from munch import Munch
from deep_keyphrase.dataloader import *
from deep_keyphrase.utils.vocab_loader import load_vocab
from deep_keyphrase.copy_rnn.model_tf import CopyRnnTF
from deep_keyphrase.dataloader import PAD_WORD
from deep_keyphrase.copy_rnn.predict_tf import PredictorTF
from deep_keyphrase.evaluation import KeyphraseEvaluator


class CopyRnnTrainerTF(object):
    def __init__(self):
        self.args = self.parse_args()
        self.vocab2id = load_vocab(self.args.vocab_path)
        self.dest_base_dir = self.args.dest_base_dir
        self.writer = tf.summary.create_file_writer(self.dest_base_dir + '/logs')
        self.exp_name = self.args.exp_name
        self.pad_idx = self.vocab2id[PAD_WORD]
        self.eval_topn = (5, 10)
        self.macro_evaluator = KeyphraseEvaluator(self.eval_topn, 'macro',
                                                  self.args.token_field, self.args.keyphrase_field)
        self.micro_evaluator = KeyphraseEvaluator(self.eval_topn, 'micro',
                                                  self.args.token_field, self.args.keyphrase_field)
        self.best_f1 = None
        self.best_step = 0
        self.not_update_count = 0
        self.logger = get_logger(__name__)
        self.total_vocab_size = len(self.vocab2id) + self.args.max_oov_count

    def train(self):
        # avoid tensorflow hold all gpus
        with tf.device('/device:GPU:0'):
            self.train_func()

    def train_func(self):
        model = CopyRnnTF(self.args, self.vocab2id)
        dataloader = KeyphraseDataLoader(data_source=self.args.train_filename,
                                         vocab2id=self.vocab2id,
                                         mode='train',
                                         args=self.args)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)

        @tf.function
        def train_step(x, x_with_oov, x_len, target):
            batch_size = x.shape[0]
            dec_len = self.args.max_target_len
            with tf.GradientTape() as tape:
                loss = 0
                probs, enc_output, prev_h, prev_c = model(x, x_with_oov, x_len, tf.constant(0),
                                                          target[:, :-1], None, None,
                                                          tf.convert_to_tensor(batch_size), dec_len)
                for batch_idx in range(batch_size):
                    dec_target = target[batch_idx, 1:]
                    target_idx = tf.one_hot(dec_target, self.total_vocab_size)
                    dec_step_loss = -tf.reduce_sum(probs[batch_idx, :] * target_idx, axis=1)
                    mask = tf.cast(dec_target != self.pad_idx, dtype=tf.float32)

                    dec_step_loss *= mask
                    loss += tf.reduce_sum(dec_step_loss) / tf.reduce_sum(mask)

                loss /= batch_size
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [(tf.clip_by_value(grad, -0.1, 0.1)) for grad in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            return loss

        step_idx = 0
        for epoch in range(self.args.epochs):
            for batch in dataloader:
                loss = train_step(batch[TOKENS], batch[TOKENS_OOV], batch[TOKENS_LENS], batch[TARGET])
                with self.writer.as_default():
                    tf.summary.scalar('loss', loss, step=step_idx)
                step_idx += 1
                if not step_idx % self.args.save_model_step:
                    model_basename = self.dest_base_dir + '/{}_step{}'.format(self.exp_name, step_idx)
                    # write_json(model_basename + '.json', vars(self.args))
                    # beam_search_graph = model.beam_search.get_concrete_function(
                    #     x=tf.TensorSpec(shape=[None, self.args.max_src_len], dtype=tf.int64),
                    #     x_with_oov=tf.TensorSpec(shape=[None, self.args.max_src_len], dtype=tf.int64),
                    #     x_len=tf.TensorSpec(shape=[None], dtype=tf.int64),
                    #     batch_size=tf.TensorSpec(shape=[None], dtype=tf.int64)
                    # )
                    # tf.saved_model.save(model, model_basename, signatures=beam_search_graph)
                    model.save_weights(model_basename + '.ckpt', save_format='tf')
                    write_json(model_basename + '.json', vars(self.args))
                    f1 = self.evaluate(model, step_idx)
                    self.logger.info('step {}, f1 {}'.format(step_idx, f1))

    def evaluate(self, model: CopyRnnTF, step):
        test_basename = '/{}_step_{}.pred.jsonl'.format(self.args.exp_name, step)
        pred_test_filename = self.dest_base_dir + test_basename
        predictor = PredictorTF(model, self.vocab2id, self.args)
        args_dict = vars(self.args)
        args_dict['batch_size'] = args_dict['eval_batch_size']
        args = Munch(args_dict)
        loader = KeyphraseDataLoader(data_source=self.args.test_filename,
                                     vocab2id=self.vocab2id,
                                     mode=EVAL_MODE,
                                     args=args)

        for batch in loader:
            kp_result = predictor.eval_predict(batch)
            result = []
            for item, pred_keyphrases in zip(batch[RAW_BATCH], kp_result):
                result_item = {'patent_id': item['patent_id'], 'pred_keyphrases': pred_keyphrases,
                               self.args.token_field: item[self.args.token_field],
                               self.args.keyphrase_field: item[self.args.keyphrase_field]}
                result.append(result_item)
            append_jsonlines(pred_test_filename, result)

        macro_all_ret = self.macro_evaluator.evaluate(pred_test_filename)
        macro_present_ret = self.macro_evaluator.evaluate(pred_test_filename, 'present')
        macro_absent_ret = self.macro_evaluator.evaluate(pred_test_filename, 'absent')
        stage = 'test'

        for n, counter in macro_all_ret.items():
            for k, v in counter.items():
                name = '{}_macro_{}_{}'.format('test', k, n)
                tf.summary.scalar(name, v, step=step)
        for n in self.eval_topn:
            name = 'present_{}_macro_f1_{}'.format(stage, n)
            tf.summary.scalar(name, macro_present_ret[n]['f1'], step=step)
        for n in self.eval_topn:
            absent_f1_name = 'absent_{}_macro_f1_{}'.format(stage, n)
            tf.summary.scalar(absent_f1_name, macro_absent_ret[n]['f1'], step=step)
            absent_recall_name = 'absent_{}_macro_recall_{}'.format(stage, n)
            tf.summary.scalar(absent_recall_name, macro_absent_ret[n]['recall'], step=step)
        return macro_all_ret[self.eval_topn[-1]]['f1']

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
        parser.add_argument("-batch_size", type=int, default=128, help='')
        parser.add_argument("-learning_rate", type=float, default=1e-3, help='')
        parser.add_argument("-eval_batch_size", type=int, default=20, help='')
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
        parser.add_argument('-fix_batch_size', action='store_true', help='')

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
