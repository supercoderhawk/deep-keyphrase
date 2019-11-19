# -*- coding: UTF-8 -*-
import argparse
import torch
from deep_keyphrase.base_trainer import BaseTrainer
from deep_keyphrase.copy_transformer.model import CopyTransformer
from deep_keyphrase.dataloader import (TARGET, TOKENS)
from deep_keyphrase.utils.vocab_loader import load_vocab


class CopyTransformerTrainer(BaseTrainer):
    def __init__(self):
        args = self.parse_args()
        vocab2id = load_vocab(args.vocab_path, vocab_size=args.vocab_size)
        model = CopyTransformer(args, vocab2id)
        super().__init__(args, model)

    def train_batch(self, batch):
        torch.autograd.set_detect_anomaly(True)
        loss = 0
        self.optimizer.zero_grad()
        targets = batch[TARGET]
        if torch.cuda.is_available():
            targets = targets.cuda()
        batch_size = len(batch[TOKENS])
        encoder_output = encoder_mask = None
        prev_copy_state = None
        prev_decoder_state = torch.zeros(batch_size, self.args.input_dim)
        for target_index in range(self.args.max_target_len):
            prev_output_tokens = targets[:, :target_index + 1].clone()
            true_indices = targets[:, target_index + 1].clone()
            output = self.model(src_dict=batch,
                                prev_output_tokens=prev_output_tokens,
                                encoder_output=encoder_output,
                                encoder_mask=encoder_mask,
                                prev_decoder_state=prev_decoder_state,
                                position=target_index,
                                prev_copy_state=prev_copy_state)
            probs, prev_decoder_state, prev_copy_state, encoder_output, encoder_mask = output
            loss += self.loss_func(probs, true_indices)

        loss.backward()
        self.optimizer.step()
        # torch.cuda.empty_cache()
        return loss

    def evaluate(self, step):
        pass

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
        parser.add_argument("-epochs", type=int, default=10, help='')
        parser.add_argument("-batch_size", type=int, default=12, help='')
        parser.add_argument("-learning_rate", type=float, default=1e-4, help='')
        parser.add_argument("-eval_batch_size", type=int, default=1, help='')
        parser.add_argument("-dropout", type=float, default=0.0, help='')
        parser.add_argument("-grad_norm", type=float, default=0.0, help='')
        parser.add_argument("-max_grad", type=float, default=5.0, help='')
        parser.add_argument("-shuffle_in_batch", action='store_true', help='')
        parser.add_argument("-teacher_forcing", action='store_true', help='')
        parser.add_argument("-beam_size", type=float, default=50, help='')
        parser.add_argument('-tensorboard_dir', type=str, default='', help='')
        parser.add_argument('-logfile', type=str, default='train_log.log', help='')
        parser.add_argument('-save_model_step', type=int, default=5000, help='')
        parser.add_argument('-early_stop_tolerance', type=int, default=50, help='')
        parser.add_argument('-train_parallel', action='store_true', help='')

        # model specific parameter
        parser.add_argument("-input_dim", type=int, default=256, help='')
        parser.add_argument("-src_head_size", type=int, default=4, help='')
        parser.add_argument("-target_head_size", type=int, default=4, help='')
        parser.add_argument("-feed_forward_dim", type=int, default=1024, help='')
        parser.add_argument("-src_dropout", type=int, default=0.1, help='')
        parser.add_argument("-target_dropout", type=int, default=0.1, help='')
        parser.add_argument("-src_layers", type=int, default=6, help='')
        parser.add_argument("-target_layers", type=int, default=6, help='')
        parser.add_argument("-max_src_len", type=int, default=1000, help='')
        parser.add_argument("-max_target_len", type=int, default=8, help='')
        parser.add_argument("-max_oov_count", type=int, default=100, help='')
        parser.add_argument("-copy_net", action='store_true', help='')

        args = parser.parse_args()
        return args


if __name__ == '__main__':
    CopyTransformerTrainer().train()
