# -*- coding: UTF-8 -*-
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pysenal import write_json
from deep_keyphrase.utils.vocab_loader import load_vocab
from deep_keyphrase.copy_rnn.model import CopyRNN
from deep_keyphrase.copy_rnn.dataloader import CopyRnnDataLoader
from deep_keyphrase.utils.constants import PAD_WORD, BOS_WORD


def train(args):
    vocab2id = load_vocab(args.vocab_path, args.vocab_size)

    model = CopyRNN(args, vocab2id)
    if torch.cuda.is_available():
        model = model.cuda()
    loss_func = nn.NLLLoss(ignore_index=vocab2id[PAD_WORD])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = CopyRnnDataLoader(args.src_filename,
                                     vocab2id,
                                     args.batch_size,
                                     args.max_src_len,
                                     args.max_oov_count,
                                     args.max_target_len,
                                     'train')
    for epoch in range(1, args.epochs + 1):
        for batch_idx, batch in enumerate(train_loader):
            loss = 0
            src_tokens = batch['tokens']
            src_tokens_with_oov = batch['tokens_with_oov']
            oov_counts = batch['oov_count']
            src_lens = batch['tokens_len']
            targets = batch['target']
            batch_size = len(src_tokens)
            encoder_output = None
            decoder_state = torch.zeros(batch_size, args.target_hidden_size)
            hidden_state = None
            for target_index in range(args.max_target_len):
                prev_output_tokens = targets[:, target_index].unsqueeze(1)
                true_indices = targets[:, target_index + 1]
                decoder_prob, encoder_output, decoder_state, hidden_state = model(src_tokens,
                                                                                  src_lens,
                                                                                  prev_output_tokens,
                                                                                  encoder_output,
                                                                                  src_tokens_with_oov,
                                                                                  oov_counts,
                                                                                  decoder_state,
                                                                                  hidden_state)
                loss += loss_func(decoder_prob, true_indices)
            print(batch_idx, loss.data.numpy())
            loss.backward()

            # clip norm, this is very import for avoiding nan gradient and misconvergence
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            if batch_idx and batch_idx % 1000 == 0:
                model_basename = args.dest_dir + 'copy_rnn_batch_{}'.format(batch_idx)
                torch.save(model.state_dict(), model_basename + '.model')
                write_json(model_basename + '.json', vars(args))
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
    parser.add_argument("-embed_size", type=int, default=500, help='')
    parser.add_argument("-max_oov_count", type=int, default=100, help='')
    parser.add_argument("-max_src_len", type=int, default=1500, help='')
    parser.add_argument("-max_target_len", type=int, default=8, help='')
    parser.add_argument("-src_hidden_size", type=int, default=100, help='')
    parser.add_argument("-target_hidden_size", type=int, default=100, help='')
    parser.add_argument('-src_num_layers', type=int, default=1, help='')
    parser.add_argument('-target_num_layers', type=int, default=1, help='')
    parser.add_argument("-epochs", type=int, default=10, help='')
    parser.add_argument("-batch_size", type=int, default=64, help='')
    parser.add_argument("-dropout", type=float, default=0.5, help='')
    parser.add_argument("-max_grad_norm", type=float, default=2, help='')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
