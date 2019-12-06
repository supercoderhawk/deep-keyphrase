# -*- coding: UTF-8 -*-
import os
import torch
from pysenal import append_jsonlines
from deep_keyphrase.base_predictor import BasePredictor
from deep_keyphrase.dataloader import KeyphraseDataLoader, TOKENS, RAW_BATCH
from deep_keyphrase.utils.constants import BOS_WORD
from deep_keyphrase.utils.vocab_loader import load_vocab
from deep_keyphrase.utils.tokenizer import token_char_tokenize
from .model import CopyTransformer
from .beam_search import TransformerBeamSearch


class CopyTransformerPredictor(BasePredictor):
    def __init__(self, model_info, vocab_info, beam_size, max_target_len, max_src_length):
        super().__init__(model_info)
        if isinstance(vocab_info, str):
            self.vocab2id = load_vocab(vocab_info)
        elif isinstance(vocab_info, dict):
            self.vocab2id = vocab_info
        else:
            raise ValueError('vocab info type error')
        self.id2vocab = dict(zip(self.vocab2id.values(), self.vocab2id.keys()))
        self.config = self.load_config(model_info)
        self.model = self.load_model(model_info, CopyTransformer(self.config, self.vocab2id))
        self.model.eval()
        self.beam_size = beam_size
        self.max_target_len = max_target_len
        self.max_src_len = max_src_length
        self.beam_searcher = TransformerBeamSearch(model=self.model,
                                                   beam_size=self.beam_size,
                                                   max_target_len=self.max_target_len,
                                                   id2vocab=self.id2vocab,
                                                   bos_idx=self.vocab2id[BOS_WORD],
                                                   args=self.config)

    def predict(self, text_list, batch_size, delimiter=None):
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
                                     mode='inference')
        result = []
        for batch in loader:
            with torch.no_grad():
                result.extend(self.beam_searcher.beam_search(batch, delimiter=delimiter))
        return result

    def eval_predict(self, src_filename, dest_filename, batch_size,
                     model=None, remove_existed=False,
                     token_field='tokens', keyphrase_field='keyphrases'):
        loader = KeyphraseDataLoader(data_source=src_filename,
                                     vocab2id=self.vocab2id,
                                     batch_size=batch_size,
                                     max_oov_count=self.config.max_oov_count,
                                     max_src_len=self.max_src_len,
                                     max_target_len=self.max_target_len,
                                     mode='inference',
                                     pre_fetch=True,
                                     token_field=token_field,
                                     keyphrase_field=keyphrase_field)
        if os.path.exists(dest_filename):
            print('destination filename {} existed'.format(dest_filename))
            if remove_existed:
                os.remove(dest_filename)
        if model is not None:
            model.eval()
            self.beam_searcher = TransformerBeamSearch(model=model,
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
