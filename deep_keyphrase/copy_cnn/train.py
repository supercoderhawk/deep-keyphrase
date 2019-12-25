# -*- coding: UTF-8 -*-
import argparse
from deep_keyphrase.base_trainer import BaseTrainer
from deep_keyphrase.utils.vocab_loader import load_vocab


class CopyCnnTrainer(BaseTrainer):
    def __init__(self):
        self.args = self.parse_args()
        self.vocab2id = load_vocab(self.args.vocab_path, self.args.vocab_size)
        model = self.load_model()
        super().__init__(self.args, model)

    def load_model(self):
        pass

    def train_batch(self, batch, step):
        pass

    def evaluate(self, step):
        pass

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        # parser.add_argument()
        args = parser.parse_args(args)
        return args
