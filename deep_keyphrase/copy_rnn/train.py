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

    def train_batch(self, batch, step):
        self.model.train()
        loss = 0
        self.optimizer.zero_grad()
        if torch.cuda.is_available():
            batch[TARGET] = batch[TARGET].cuda()
        targets = batch[TARGET]
        if self.args.auto_regressive:
            loss = self.get_auto_regressive_loss(batch, loss, targets)
        else:
            loss = self.get_one_pass_loss(batch, targets)

        loss.backward()

        # clip norm, this is very import for avoiding nan gradient and misconvergence
        if self.args.max_grad:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.max_grad)
        if self.args.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)

        self.optimizer.step()
        if self.args.schedule_lr and step <= self.args.schedule_step:
            self.scheduler.step()
        return loss

    def get_one_pass_loss(self, batch, targets):
        batch_size = len(batch)
        encoder_output = None
        decoder_state = torch.zeros(batch_size, self.args.target_hidden_size)
        hidden_state = None
        prev_output_tokens = None
        output = self.model(src_dict=batch,
                            prev_output_tokens=prev_output_tokens,
                            encoder_output_dict=encoder_output,
                            prev_decoder_state=decoder_state,
                            prev_hidden_state=hidden_state)
        decoder_prob, encoder_output, decoder_state, hidden_state = output
        vocab_size = decoder_prob.size(-1)
        decoder_prob = decoder_prob.view(-1, vocab_size)
        loss = self.loss_func(decoder_prob, targets[:, 1:].flatten())
        return loss

    def get_auto_regressive_loss(self, batch, loss, targets):
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
            prev_output_tokens = prev_output_tokens.clone()
            output = self.model(src_dict=batch,
                                prev_output_tokens=prev_output_tokens,
                                encoder_output_dict=encoder_output,
                                prev_decoder_state=decoder_state,
                                prev_hidden_state=hidden_state)
            decoder_prob, encoder_output, decoder_state, hidden_state = output
            true_indices = targets[:, target_index + 1].clone()
            loss += self.loss_func(decoder_prob, true_indices)
        loss /= self.args.max_target_len
        return loss

    def evaluate(self, step):
        predictor = CopyRnnPredictor(model_info={'model': self.model, 'config': self.args},
                                     vocab_info=self.vocab2id,
                                     beam_size=self.args.beam_size,
                                     max_target_len=self.args.max_target_len,
                                     max_src_length=self.args.max_src_len)

        def pred_callback(stage):
            if stage == 'valid':
                src_filename = self.args.valid_filename
                dest_filename = self.dest_dir + self.get_basename(self.args.valid_filename)
            elif stage == 'test':
                src_filename = self.args.test_filename
                dest_filename = self.dest_dir + self.get_basename(self.args.test_filename)
            else:
                raise ValueError('stage name error, must be in `valid` and `test`')
            dest_filename += '.batch_{}.pred.jsonl'.format(step)
            def predict_func():
                predictor.eval_predict(src_filename=src_filename,
                                       dest_filename=dest_filename,
                                       args=self.args,
                                       model=self.model,
                                       remove_existed=True)

            return predict_func

        valid_statistics = self.evaluate_stage(step, 'valid', pred_callback('valid'))
        test_statistics = self.evaluate_stage(step, 'test', pred_callback('test'))
        total_statistics = {**valid_statistics, **test_statistics}

        eval_filename = self.dest_dir + self.args.exp_name + '.batch_{}.eval.json'.format(step)
        write_json(eval_filename, total_statistics)
        return valid_statistics['valid_macro'][self.eval_topn[-1]]['f1']

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
        parser.add_argument('-lazy_loading', action='store_true', help='')

        # model specific parameter
        parser.add_argument("-embed_size", type=int, default=200, help='')
        parser.add_argument("-max_oov_count", type=int, default=100, help='')
        parser.add_argument("-max_src_len", type=int, default=1500, help='')
        parser.add_argument("-max_target_len", type=int, default=8, help='')
        parser.add_argument("-src_hidden_size", type=int, default=100, help='')
        parser.add_argument("-target_hidden_size", type=int, default=100, help='')
        parser.add_argument('-src_num_layers', type=int, default=1, help='')
        parser.add_argument('-target_num_layers', type=int, default=1, help='')
        parser.add_argument("-attention_mode", type=str, default='general',
                            choices=['general', 'dot', 'concat'], help='')
        parser.add_argument("-bidirectional", action='store_true', help='')
        parser.add_argument("-copy_net", action='store_true', help='')
        parser.add_argument("-input_feeding", action='store_true', help='')

        args = parser.parse_args(args)
        return args


def accuracy(probs, true_indices, pad_idx):
    pred_indices = torch.argmax(probs, dim=1)
    mask = torch.eq(true_indices, torch.ones(*true_indices.size(), dtype=torch.int64) * pad_idx)
    tp_result = torch.eq(pred_indices, true_indices).type(torch.int) * (~mask).type(torch.int)
    return torch.sum(tp_result).numpy() / true_indices.numel()


if __name__ == '__main__':
    CopyRnnTrainer().train()
