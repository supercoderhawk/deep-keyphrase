# -*- coding: UTF-8 -*-
import os
import torch
import json
from pysenal import read_file, append_jsonlines
from collections import namedtuple, OrderedDict
from deep_keyphrase.copy_rnn.model import CopyRNN
from deep_keyphrase.copy_rnn.beam_search import BeamSearch
from deep_keyphrase.dataloader import KeyphraseDataLoader, RAW_BATCH, TOKENS
from deep_keyphrase.utils.vocab_loader import load_vocab
from deep_keyphrase.utils.constants import BOS_WORD
from deep_keyphrase.utils.tokenizer import token_char_tokenize


class CopyRnnPredictor(object):
    def __init__(self, model_info, vocab_info, beam_size, max_target_len, max_src_length):
        if isinstance(vocab_info, str):
            self.vocab2id = load_vocab(vocab_info)
        elif isinstance(vocab_info, dict):
            self.vocab2id = vocab_info
        else:
            raise ValueError('vocab info type error')
        self.id2vocab = dict(zip(self.vocab2id.values(), self.vocab2id.keys()))
        self.config = self.load_config(model_info)
        self.model = self.load_model(model_info, self.vocab2id)
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

    def load_config(self, model_info):
        if 'config' not in model_info:
            if isinstance(model_info['model'], str):
                config_path = os.path.splitext(model_info['model'])[0] + '.json'
            else:
                raise ValueError('config path is not assigned')
        else:
            config_info = model_info['config']
            if isinstance(config_info, str):
                config_path = config_info
            else:
                return config_info
        # json to object
        config = json.loads(read_file(config_path),
                            object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        return config

    def load_model(self, model_info, vocab2id):
        if isinstance(model_info['model'], torch.nn.Module):
            return model_info['model']

        model_path = model_info['model']
        if not isinstance(model_path, str):
            raise TypeError('model path should be str')
        model = CopyRNN(self.config, vocab2id)
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
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def predict(self, text_list, batch_size=10, delimiter=None):
        """

        :param text_list:
        :param batch_size:
        :return:
        """
        self.model.eval()
        if len(text_list) < batch_size:
            batch_size = len(text_list)
        text_list = [{TOKENS: token_char_tokenize(i)} for i in text_list]
        loader = KeyphraseDataLoader(data_source=text_list,
                                     vocab2id=self.vocab2id,
                                     batch_size=batch_size,
                                     max_oov_count=self.config.max_oov_count,
                                     max_src_len=self.max_src_len,
                                     max_target_len=self.max_target_len,
                                     mode='valid')
        result = []
        for batch in loader:
            with torch.no_grad():
                result.extend(self.beam_searcher.beam_search(batch, delimiter=delimiter))
        return result

    def eval_predict(self, src_filename, dest_filename, batch_size, model=None, remove_existed=False):
        loader = KeyphraseDataLoader(data_source=src_filename,
                                     vocab2id=self.vocab2id,
                                     batch_size=batch_size,
                                     max_oov_count=self.config.max_oov_count,
                                     max_src_len=self.max_src_len,
                                     max_target_len=self.max_target_len,
                                     mode='valid')

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
