# -*- coding: UTF-8 -*-
import os
import argparse
import torch
from collections import OrderedDict
from munch import Munch
from pysenal import read_json
from deep_keyphrase.base_trainer import BaseTrainer
from deep_keyphrase.utils.vocab_loader import load_vocab
from deep_keyphrase.copy_cnn.model import CopyCnn


class CopyCnnTrainer(BaseTrainer):
    def __init__(self):
        self.args = self.parse_args()
        self.vocab2id = load_vocab(self.args.vocab_path, self.args.vocab_size)
        model = self.load_model()
        super().__init__(self.args, model)

    def load_model(self):
        if not self.args.train_from:
            model = CopyCnn(self.args, self.vocab2id)
        else:
            model_path = self.args.train_from
            config_path = os.path.join(os.path.dirname(model_path),
                                       self.get_basename(model_path) + '.json')

            old_config = read_json(config_path)
            old_config['train_from'] = model_path
            old_config['step'] = int(model_path.rsplit('_', 1)[-1].split('.')[0])
            self.args = Munch(old_config)
            self.vocab2id = load_vocab(self.args.vocab_path, self.args.vocab_size)

            model = CopyCnn(self.args, self.vocab2id)

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

    def evaluate(self, step):
        pass

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
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
        # parser.add_argument("-auto_regressive", action='store_true', help='')
        parser.add_argument("-epochs", type=int, default=10, help='')
        parser.add_argument("-batch_size", type=int, default=64, help='')
        parser.add_argument("-learning_rate", type=float, default=1e-4, help='')
        parser.add_argument("-eval_batch_size", type=int, default=50, help='')
        parser.add_argument("-dropout", type=float, default=0.0, help='')
        parser.add_argument("-grad_norm", type=float, default=0.0, help='')
        parser.add_argument("-max_grad", type=float, default=5.0, help='')
        parser.add_argument("-shuffle", action='store_true', help='')
        # parser.add_argument("-teacher_forcing", action='store_true', help='')
        parser.add_argument("-beam_size", type=float, default=50, help='')
        parser.add_argument('-tensorboard_dir', type=str, default='', help='')
        parser.add_argument('-logfile', type=str, default='train_log.log', help='')
        parser.add_argument('-save_model_step', type=int, default=5000, help='')
        parser.add_argument('-early_stop_tolerance', type=int, default=100, help='')
        parser.add_argument('-train_parallel', action='store_true', help='')
        # parser.add_argument('-schedule_lr', action='store_true', help='')
        # parser.add_argument('-schedule_step', type=int, default=100000, help='')
        # parser.add_argument('-schedule_gamma', type=float, default=0.5, help='')
        # parser.add_argument('-processed', action='store_true', help='')
        parser.add_argument('-prefetch', action='store_true', help='')

        parser.add_argument('-dim_size', type=int, default=100, help='')
        parser.add_argument('-kernel_width', type=int, default=5, help='')
        parser.add_argument('-encoder_layer_num', type=int, default=6, help='')
        parser.add_argument('-decoder_layer_num', type=int, default=6, help='')

        args = parser.parse_args(args)
        return args


if __name__ == '__main__':
    CopyCnnTrainer().train()
