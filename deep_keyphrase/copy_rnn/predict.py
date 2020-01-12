# -*- coding: UTF-8 -*-
import os
import torch
from munch import Munch
from pysenal import read_file, append_jsonlines
from deep_keyphrase.base_predictor import BasePredictor
from deep_keyphrase.copy_rnn.model import CopyRNN
from deep_keyphrase.copy_rnn.beam_search import BeamSearch
from deep_keyphrase.dataloader import KeyphraseDataLoader, RAW_BATCH, TOKENS, INFERENCE_MODE, EVAL_MODE
from deep_keyphrase.utils.vocab_loader import load_vocab
from deep_keyphrase.utils.constants import BOS_WORD
from deep_keyphrase.utils.tokenizer import token_char_tokenize


class CopyRnnPredictor(BasePredictor):
    def __init__(self, model_info, vocab_info, beam_size, max_target_len, max_src_length):
        """

        :param model_info: input the model information.
                            str type: model path
                            dict type: must have `model` and `config` field,
                                        indicate the model object and config object

        :param vocab_info: input the vocab information.
                            str type: vocab path
                            dict type: vocab2id dict which map word to id
        :param beam_size: beam size
        :param max_target_len: max keyphrase token length
        :param max_src_length: max source text length
        """
        super().__init__(model_info)
        if isinstance(vocab_info, str):
            self.vocab2id = load_vocab(vocab_info)
        elif isinstance(vocab_info, dict):
            self.vocab2id = vocab_info
        else:
            raise ValueError('vocab info type error')
        self.id2vocab = dict(zip(self.vocab2id.values(), self.vocab2id.keys()))
        self.config = self.load_config(model_info)
        self.model = self.load_model(model_info,  CopyRNN(self.config, self.vocab2id))
        self.model.eval()
        self.beam_size = beam_size
        self.max_target_len = max_target_len
        self.max_src_len = max_src_length
        self.beam_searcher = BeamSearch(model=self.model,
                                        beam_size=self.beam_size,
                                        max_target_len=self.max_target_len,
                                        id2vocab=self.id2vocab,
                                        bos_idx=self.vocab2id[BOS_WORD],
                                        args=self.config)
        self.pred_base_config = {'max_oov_count': self.config.max_oov_count,
                                 'max_src_len': self.max_src_len,
                                 'max_target_len': self.max_target_len,
                                 'prefetch': False,
                                 'shuffle_in_batch': False,
                                 'token_field': TOKENS,
                                 'keyphrase_field': 'keyphrases'}

    def predict(self, text_list, batch_size=10, delimiter=None, tokenized=False):
        """

        :param text_list:
        :param batch_size:
        :param delimiter:
        :param tokenized:
        :return:
        """
        # eval mode closes dropout, triggers auto regression in decoding stage
        self.model.eval()
        if len(text_list) < batch_size:
            batch_size = len(text_list)

        if tokenized:
            text_list = [{TOKENS: i} for i in text_list]
        else:
            text_list = [{TOKENS: token_char_tokenize(i)} for i in text_list]
        args = Munch({'batch_size': batch_size, **self.pred_base_config})
        loader = KeyphraseDataLoader(data_source=text_list,
                                     vocab2id=self.vocab2id,
                                     mode=INFERENCE_MODE,
                                     args=args)
        result = []
        for batch in loader:
            with torch.no_grad():
                result.extend(self.beam_searcher.beam_search(batch, delimiter=delimiter))
        return result

    def eval_predict(self, src_filename, dest_filename, args,
                     model=None, remove_existed=False):
        args_dict = vars(args)
        args_dict['batch_size'] = args_dict['eval_batch_size']
        args = Munch(args_dict)
        loader = KeyphraseDataLoader(data_source=src_filename,
                                     vocab2id=self.vocab2id,
                                     mode=EVAL_MODE,
                                     args=args)

        if os.path.exists(dest_filename):
            print('destination filename {} existed'.format(dest_filename))
            if remove_existed:
                os.remove(dest_filename)
        if model is not None:
            model.eval()
            self.beam_searcher = BeamSearch(model=model,
                                            beam_size=self.beam_size,
                                            max_target_len=self.max_target_len,
                                            id2vocab=self.id2vocab,
                                            bos_idx=self.vocab2id[BOS_WORD],
                                            args=self.config)

        for batch in loader:
            with torch.no_grad():
                batch_result = self.beam_searcher.beam_search(batch, delimiter=None)
                final_result = []
                assert len(batch_result) == len(batch[RAW_BATCH])
                for item_input, item_output in zip(batch[RAW_BATCH], batch_result):
                    item_input['pred_keyphrases'] = item_output
                    final_result.append(item_input)
                append_jsonlines(dest_filename, final_result)
