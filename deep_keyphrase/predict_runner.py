# -*- coding: UTF-8 -*-
import argparse
from munch import Munch
from pysenal import get_chunk, read_jsonline_lazy
from deep_keyphrase.copy_rnn.predict import CopyRnnPredictor
from deep_keyphrase.dataloader import KeyphraseDataLoader
from deep_keyphrase.utils.vocab_loader import load_vocab


class PredictRunner(object):
    def __init__(self):
        self.args = self.parse_args()
        self.predictor = CopyRnnPredictor(model_info=self.args.model_path,
                                          vocab_info=self.args.vocab_path,
                                          beam_size=self.args.beam_size,
                                          max_src_length=self.args.max_src_len,
                                          max_target_len=self.args.max_target_len)

        self.config = {**self.predictor.config, 'batch_size': self.args.batch_size}
        self.config = Munch(self.config)

    def predict(self):
        # vocab2id = load_vocab(self.args.vocab_size, vocab_size=self.config.vocab_size)
        # loader = KeyphraseDataLoader(self.args.src_filename,
        #                              vocab2id=vocab2id,
        #                              mode='inference', args=self.config)
        # for batch in loader:
        # for batch in loader:
        self.predictor.eval_predict(self.args.src_filename, self.args.dest_filename,
                                    args=self.config)
        # chunk_size =
        # for item_chunk in get_chunk(read_jsonline_lazy())
        #     pass

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-src_filename', type=str, help='')
        parser.add_argument('-mode_path', type=str, help='')
        parser.add_argument('-vocab_path', type=str, help='')
        parser.add_argument('-batch_size', type=int, default=10, help='')
        parser.add_argument('-beam_size', type=int, default=200, help='')
        parser.add_argument('-max_src_len', type=int, default=1500, help='')
        parser.add_argument('-max_target_len', type=int, default=8, help='')
        args = parser.parse_args()
        return args


if __name__ == '__main__':
    PredictRunner().predict()
