# -*- coding: UTF-8 -*-
import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pysenal import write_json, get_logger
from deep_keyphrase.utils.vocab_loader import load_vocab
from deep_keyphrase.copy_rnn.model import CopyRNN
from deep_keyphrase.copy_rnn.dataloader import (CopyRnnDataLoader, TOKENS, TOKENS_LENS,
                                                TOKENS_OOV, OOV_COUNT, TARGET)
from deep_keyphrase.copy_rnn.predict import CopyRnnPredictor
from deep_keyphrase.utils.constants import PAD_WORD, BOS_WORD
from deep_keyphrase.utils.evaluation import KeyphraseEvaluator


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.vocab2id = load_vocab(args.vocab_path, args.vocab_size)

        self.model = CopyRNN(args, self.vocab2id)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.loss_func = nn.NLLLoss(ignore_index=self.vocab2id[PAD_WORD])
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.logger = get_logger('train')
        self.train_loader = CopyRnnDataLoader(args.train_filename,
                                              self.vocab2id,
                                              args.batch_size,
                                              args.max_src_len,
                                              args.max_oov_count,
                                              args.max_target_len,
                                              'train')
        timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        self.dest_dir = os.path.join(args.dest_base_dir, args.exp_name + '-' + timemark) + '/'
        os.mkdir(self.dest_dir)
        if not args.tensorboard_dir:
            tensorboard_dir = self.dest_dir + 'logs/'
        else:
            tensorboard_dir = args.tensorboard_dir
        self.writer = SummaryWriter(tensorboard_dir)
        self.predictor = CopyRnnPredictor(model_info={'model': self.model, 'config': args},
                                          vocab_info=self.vocab2id,
                                          beam_size=args.beam_size,
                                          max_target_len=args.max_target_len,
                                          max_src_length=args.max_src_len)
        self.eval_topn = (5, 10)
        self.macro_evaluator = KeyphraseEvaluator(self.eval_topn, 'macro')
        self.micro_evaluator = KeyphraseEvaluator(self.eval_topn, 'micro')
        self.best_f1 = None
        self.not_update_count = 0

    def train(self):
        step = 0
        is_stop = False
        self.logger.info('destination dir:{}'.format(self.dest_dir))
        for epoch in range(1, self.args.epochs + 1):
            for batch_idx, batch in enumerate(self.train_loader):
                self.model.train()
                try:
                    loss = self.train_batch(batch)
                except Exception as e:
                    print(e)
                    loss = 0.0
                step += 1
                self.writer.add_scalar('loss', loss, step)
                if step and step % self.args.save_model_step == 0:
                    self.save_model(step, epoch)
                    if self.not_update_count >= self.args.early_stop_tolerance:
                        is_stop = True
                        break
            if is_stop:
                break

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
            prev_output_tokens = targets[:, target_index].unsqueeze(1)
            true_indices = targets[:, target_index + 1]
            output = self.model(src_dict=batch,
                                prev_output_tokens=prev_output_tokens,
                                encoder_output_dict=encoder_output,
                                prev_decoder_state=decoder_state,
                                prev_hidden_state=hidden_state)
            decoder_prob, encoder_output, decoder_state, hidden_state = output
            loss += self.loss_func(decoder_prob, true_indices)

        loss.backward()

        # clip norm, this is very import for avoiding nan gradient and misconvergence
        if self.args.max_grad:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.max_grad)
        if self.args.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)

        self.optimizer.step()
        return loss

    def save_model(self, step, epoch):
        valid_f1 = self.evaluate(step)
        if self.best_f1 is None:
            self.best_f1 = valid_f1
        elif valid_f1 >= self.best_f1:
            self.best_f1 = valid_f1
            self.not_update_count = 0
        else:
            self.not_update_count += 1
        exp_name = self.args.exp_name
        model_basename = self.dest_dir + '{}_epoch_{}_batch_{}'.format(exp_name, epoch, step)
        torch.save(self.model.state_dict(), model_basename + '.model')
        write_json(model_basename + '.json', vars(self.args))
        self.logger.info('epoch {} step {}, model saved'.format(epoch, step))

    def evaluate(self, step):
        pred_valid_filename = self.dest_dir + self.get_basename(self.args.valid_filename)
        pred_valid_filename += '.batch_{}.pred.jsonl'.format(step)
        eval_filename = self.dest_dir + self.args.exp_name + '.batch_{}.eval.json'.format(step)
        self.predictor.eval_predict(self.args.valid_filename, pred_valid_filename,
                                    self.args.eval_batch_size, self.model, True)
        valid_macro_ret = self.macro_evaluator.evaluate(pred_valid_filename)
        # valid_micro_ret = self.micro_evaluator.evaluate(pred_valid_filename)
        for n, counter in valid_macro_ret.items():
            for k, v in counter.items():
                name = 'valid/macro_{}@{}'.format(k, n)
                self.writer.add_scalar(name, v, step)
        pred_test_filename = self.dest_dir + self.get_basename(self.args.test_filename)
        pred_test_filename += '.batch_{}.pred.jsonl'.format(step)

        self.predictor.eval_predict(self.args.test_filename, pred_test_filename,
                                    self.args.batch_size, self.model, True)
        test_macro_ret = self.macro_evaluator.evaluate(pred_test_filename)
        for n, counter in test_macro_ret.items():
            for k, v in counter.items():
                name = 'test/macro_{}@{}'.format(k, n)
                self.writer.add_scalar(name, v, step)
        write_json(eval_filename, {'valid_macro': valid_macro_ret, 'test_macro': test_macro_ret})
        # valid_micro_ret = self.micro_evaluator.evaluate(pred_test_filename)
        return valid_macro_ret[self.eval_topn[-1]]['recall']

    def get_basename(self, filename):
        return os.path.splitext(os.path.basename(filename))[0]


def accuracy(probs, true_indices, pad_idx):
    pred_indices = torch.argmax(probs, dim=1)
    mask = torch.eq(true_indices, torch.ones(*true_indices.size(), dtype=torch.int64) * pad_idx)
    tp_result = torch.eq(pred_indices, true_indices).type(torch.int) * (~mask).type(torch.int)
    return torch.sum(tp_result).numpy() / true_indices.numel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_name", required=True, type=str, help='')
    parser.add_argument("-train_filename", required=True, type=str, help='')
    parser.add_argument("-valid_filename", required=True, type=str, help='')
    parser.add_argument("-test_filename", required=True, type=str, help='')
    parser.add_argument("-dest_base_dir", required=True, type=str, help='')
    parser.add_argument("-vocab_path", required=True, type=str, help='')
    parser.add_argument("-vocab_size", type=int, default=500000, help='')
    parser.add_argument("-embed_size", type=int, default=200, help='')
    parser.add_argument("-max_oov_count", type=int, default=100, help='')
    parser.add_argument("-max_src_len", type=int, default=1500, help='')
    parser.add_argument("-max_target_len", type=int, default=8, help='')
    parser.add_argument("-src_hidden_size", type=int, default=100, help='')
    parser.add_argument("-target_hidden_size", type=int, default=100, help='')
    parser.add_argument('-src_num_layers', type=int, default=1, help='')
    parser.add_argument('-target_num_layers', type=int, default=1, help='')
    parser.add_argument("-epochs", type=int, default=10, help='')
    parser.add_argument("-batch_size", type=int, default=50, help='')
    parser.add_argument("-eval_batch_size", type=int, default=10, help='')
    parser.add_argument("-dropout", type=float, default=0.5, help='')
    parser.add_argument("-grad_norm", type=float, default=2.0, help='')
    parser.add_argument("-max_grad", type=float, default=10.0, help='')
    parser.add_argument("-bidirectional", action='store_true', help='')
    parser.add_argument("-use_vanilla_rnn_search", action='store_false', help='')
    parser.add_argument("-teacher_forcing", action='store_true', help='')
    parser.add_argument("-beam_size", type=float, default=50, help='')
    parser.add_argument('-tensorboard_dir', type=str, default='', help='')
    parser.add_argument('-logfile', type=str, default='train_log.log', help='')
    parser.add_argument('-save_model_step', type=int, default=5000, help='')
    parser.add_argument('-early_stop_tolerance', type=int, default=30, help='')
    args = parser.parse_args()
    Trainer(args).train()


if __name__ == '__main__':
    main()
