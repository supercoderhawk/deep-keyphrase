# -*- coding: UTF-8 -*-
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pysenal import write_json, get_logger
from deep_keyphrase.utils.vocab_loader import load_vocab
from deep_keyphrase.copy_rnn.model import CopyRNN
from deep_keyphrase.copy_rnn.dataloader import (CopyRnnDataLoader, TOKENS, TOKENS_LENS,
                                                TOKENS_OOV, OOV_COUNT, TARGET)
from deep_keyphrase.utils.constants import PAD_WORD, BOS_WORD


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
        self.train_loader = CopyRnnDataLoader(args.src_filename,
                                              self.vocab2id,
                                              args.batch_size,
                                              args.max_src_len,
                                              args.max_oov_count,
                                              args.max_target_len,
                                              'train')

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            for batch_idx, batch in enumerate(self.train_loader):
                loss = self.train_batch(batch)
                if batch_idx and batch_idx % 10 == 0:
                    print(batch_idx, loss.data.cpu().numpy())
                if batch_idx and batch_idx % 1000 == 0:
                    self.save_model()

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
        loss /= batch_size

        loss.backward()

        # clip norm, this is very import for avoiding nan gradient and misconvergence
        if self.args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

        self.optimizer.step()
        return loss

    def save_model(self):
        model_basename = self.args.dest_dir + 'copy_rnn_batch_{}'.format(batch_idx)
        torch.save(self.model.state_dict(), model_basename + '.model')
        write_json(model_basename + '.json', vars(self.args))
        print('saved checkpoint')


def accuracy(probs, true_indices, pad_idx):
    pred_indices = torch.argmax(probs, dim=1)
    mask = torch.eq(true_indices, torch.ones(*true_indices.size(), dtype=torch.int64) * pad_idx)
    tp_result = torch.eq(pred_indices, true_indices).type(torch.int) * (~mask).type(torch.int)
    return torch.sum(tp_result).numpy() / true_indices.numel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src_filename", type=str, help='')
    parser.add_argument("-dest_dir", type=str, help='')
    parser.add_argument("-vocab_path", type=str, help='')
    parser.add_argument("-vocab_size", type=int, default=500000, help='')
    parser.add_argument("-embed_size", type=int, default=150, help='')
    parser.add_argument("-max_oov_count", type=int, default=100, help='')
    parser.add_argument("-max_src_len", type=int, default=1500, help='')
    parser.add_argument("-max_target_len", type=int, default=8, help='')
    parser.add_argument("-src_hidden_size", type=int, default=100, help='')
    parser.add_argument("-target_hidden_size", type=int, default=100, help='')
    parser.add_argument('-src_num_layers', type=int, default=1, help='')
    parser.add_argument('-target_num_layers', type=int, default=1, help='')
    parser.add_argument("-epochs", type=int, default=10, help='')
    parser.add_argument("-batch_size", type=int, default=128, help='')
    parser.add_argument("-dropout", type=float, default=0.5, help='')
    parser.add_argument("-max_grad_norm", type=float, default=5, help='')
    parser.add_argument("-bidirectional", type=float, default=5, help='')
    args = parser.parse_args()
    Trainer(args).train()


if __name__ == '__main__':
    main()
