# -*- coding: UTF-8 -*-
import os
import argparse
from collections import OrderedDict
from munch import Munch
import torch
from pysenal import write_json, read_json
from deep_keyphrase.utils.vocab_loader import load_vocab
from deep_keyphrase.copy_rnn.model import CopyRNN
from deep_keyphrase.base_trainer import BaseTrainer
from deep_keyphrase.dataloader import TOKENS, TARGET
from deep_keyphrase.copy_rnn.predict import CopyRnnPredictor


class CopyRnnTrainer(BaseTrainer):
    def __init__(self):
        self.args = self.parse_args()
        self.vocab2id = load_vocab(self.args.vocab_path, self.args.vocab_size)
        model = self.load_model()
        super().__init__(self.args, model)

    def load_model(self):
        if not self.args.train_from:
            model = CopyRNN(self.args, self.vocab2id)
        else:
            model_path = self.args.train_from
            config_path = os.path.join(os.path.dirname(model_path),
                                       self.get_basename(model_path) + '.json')

            old_config = read_json(config_path)
            old_config['train_from'] = model_path
            old_config['step'] = int(model_path.rsplit('_', 1)[-1].split('.')[0])
            self.args = Munch(old_config)
            self.vocab2id = load_vocab(self.args.vocab_path, self.args.vocab_size)

            model = CopyRNN(self.args, self.vocab2id)

            if torch.cuda.is_available():
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            state_dict = OrderedDict()
            # avoid error when load parallel trained model
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    k = k[7:]
                state_dict[k] = v
            model.load_state_dict(state_dict)

        return model

    def train_batch(self, batch):
        loss = 0
        self.optimizer.zero_grad()
        targets = batch[TARGET]
        if torch.cuda.is_available():
            targets = targets.cuda()
        batch_size = len(batch[TOKENS])
        encoder_output = None
        decoder_state = torch.zeros(batch_size, self.args.target_hidden_size)
        hidden_state = None
        for target_index in range(self.args.max_target_len):
            if target_index == 0:
                # bos indices
                prev_output_tokens = targets[:, target_index].unsqueeze(1)
            else:
                if self.args.teacher_forcing:
                    prev_output_tokens = targets[:, target_index].unsqueeze(1)
                else:
                    best_probs, prev_output_tokens = torch.topk(decoder_prob, 1, 1)

            output = self.model(src_dict=batch,
                                prev_output_tokens=prev_output_tokens,
                                encoder_output_dict=encoder_output,
                                prev_decoder_state=decoder_state,
                                prev_hidden_state=hidden_state)
            decoder_prob, encoder_output, decoder_state, hidden_state = output
            true_indices = targets[:, target_index + 1]
            loss += self.loss_func(decoder_prob, true_indices)

        loss.backward()

        # clip norm, this is very import for avoiding nan gradient and misconvergence
        if self.args.max_grad:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.max_grad)
        if self.args.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)

        self.optimizer.step()
        if self.args.schedule_lr:
            self.scheduler.step()
        return loss

    def evaluate(self, step):
        predictor = CopyRnnPredictor(model_info={'model': self.model, 'config': self.args},
                                     vocab_info=self.vocab2id,
                                     beam_size=self.args.beam_size,
                                     max_target_len=self.args.max_target_len,
                                     max_src_length=self.args.max_src_len)
        pred_valid_filename = self.dest_dir + self.get_basename(self.args.valid_filename)
        pred_valid_filename += '.batch_{}.pred.jsonl'.format(step)
        eval_filename = self.dest_dir + self.args.exp_name + '.batch_{}.eval.json'.format(step)
        predictor.eval_predict(self.args.valid_filename, pred_valid_filename,
                               self.args.eval_batch_size, self.model, True,
                               token_field=self.args.token_field,
                               keyphrase_field=self.args.keyphrase_field)
        valid_macro_all_ret = self.macro_evaluator.evaluate(pred_valid_filename)
        valid_macro_present_ret = self.macro_evaluator.evaluate(pred_valid_filename, 'present')
        valid_macro_absent_ret = self.macro_evaluator.evaluate(pred_valid_filename, 'absent')

        for n, counter in valid_macro_all_ret.items():
            for k, v in counter.items():
                name = 'valid/macro_{}@{}'.format(k, n)
                self.writer.add_scalar(name, v, step)
        for n in self.eval_topn:
            name = 'present/valid macro_f1@{}'.format(n)
            self.writer.add_scalar(name, valid_macro_present_ret[n]['f1'], step)
        for n in self.eval_topn:
            name = 'absent/valid macro_f1@{}'.format(n)
            self.writer.add_scalar(name, valid_macro_absent_ret[n]['f1'], step)
        pred_test_filename = self.dest_dir + self.get_basename(self.args.test_filename)
        pred_test_filename += '.batch_{}.pred.jsonl'.format(step)

        predictor.eval_predict(self.args.test_filename, pred_test_filename,
                               self.args.eval_batch_size, self.model, True)
        test_macro_all_ret = self.macro_evaluator.evaluate(pred_test_filename)
        test_macro_present_ret = self.macro_evaluator.evaluate(pred_test_filename, 'present')
        test_macro_absent_ret = self.macro_evaluator.evaluate(pred_test_filename, 'absent')
        for n, counter in test_macro_all_ret.items():
            for k, v in counter.items():
                name = 'test/macro_{}@{}'.format(k, n)
                self.writer.add_scalar(name, v, step)
        for n in self.eval_topn:
            name = 'present/test macro_f1@{}'.format(n)
            self.writer.add_scalar(name, test_macro_present_ret[n]['f1'], step)
        for n in self.eval_topn:
            name = 'absent/test macro_f1@{}'.format(n)
            self.writer.add_scalar(name, test_macro_absent_ret[n]['f1'], step)
        total_statistics = {'valid_macro': valid_macro_all_ret,
                            'valid_present_macro': valid_macro_present_ret,
                            'valid_absent_macro': valid_macro_absent_ret,
                            'test_macro': test_macro_all_ret,
                            'test_present_macro': test_macro_present_ret,
                            'test_absent_macro': test_macro_absent_ret}
        write_json(eval_filename, total_statistics)
        return valid_macro_all_ret[self.eval_topn[-1]]['f1']

    def parse_args(self):
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
        parser.add_argument("-epochs", type=int, default=10, help='')
        parser.add_argument("-batch_size", type=int, default=64, help='')
        parser.add_argument("-learning_rate", type=float, default=1e-3, help='')
        parser.add_argument("-eval_batch_size", type=int, default=50, help='')
        parser.add_argument("-dropout", type=float, default=0.1, help='')
        parser.add_argument("-grad_norm", type=float, default=0.0, help='')
        parser.add_argument("-max_grad", type=float, default=5.0, help='')
        parser.add_argument("-shuffle_in_batch", action='store_true', help='')
        parser.add_argument("-teacher_forcing", action='store_true', help='')
        parser.add_argument("-beam_size", type=float, default=50, help='')
        parser.add_argument('-tensorboard_dir', type=str, default='', help='')
        parser.add_argument('-logfile', type=str, default='train_log.log', help='')
        parser.add_argument('-save_model_step', type=int, default=5000, help='')
        parser.add_argument('-early_stop_tolerance', type=int, default=100, help='')
        parser.add_argument('-train_parallel', action='store_true', help='')
        parser.add_argument('-schedule_lr', action='store_true', help='')
        parser.add_argument('-schedule_step', type=int, default=100000, help='')
        parser.add_argument('-schedule_gamma', type=float, default=0.5, help='')

        # model specific parameter
        parser.add_argument("-embed_size", type=int, default=200, help='')
        parser.add_argument("-max_oov_count", type=int, default=100, help='')
        parser.add_argument("-max_src_len", type=int, default=1500, help='')
        parser.add_argument("-max_target_len", type=int, default=8, help='')
        parser.add_argument("-src_hidden_size", type=int, default=100, help='')
        parser.add_argument("-target_hidden_size", type=int, default=100, help='')
        parser.add_argument('-src_num_layers', type=int, default=1, help='')
        parser.add_argument('-target_num_layers', type=int, default=1, help='')
        parser.add_argument("-bidirectional", action='store_true', help='')
        parser.add_argument("-copy_net", action='store_true', help='')
        parser.add_argument("-input_feeding", action='store_true', help='')

        args = parser.parse_args()
        return args


def accuracy(probs, true_indices, pad_idx):
    pred_indices = torch.argmax(probs, dim=1)
    mask = torch.eq(true_indices, torch.ones(*true_indices.size(), dtype=torch.int64) * pad_idx)
    tp_result = torch.eq(pred_indices, true_indices).type(torch.int) * (~mask).type(torch.int)
    return torch.sum(tp_result).numpy() / true_indices.numel()


if __name__ == '__main__':
    CopyRnnTrainer().train()
