# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from .model_tf import CopyRnnTF
from deep_keyphrase.dataloader import UNK_WORD, PAD_WORD, TOKENS, TOKENS_OOV, TOKENS_LENS, OOV_LIST, EOS_WORD
from ..utils.tokenizer import token_char_tokenize


class PredictorTF(object):
    def __init__(self, model: CopyRnnTF, vocab2id, args):
        self.model = model
        self.max_src_len = args.max_src_len
        self.vocab2id = vocab2id
        self.id2vocab = dict(zip(self.vocab2id.values(), self.vocab2id.keys()))
        self.vocab_size = len(self.vocab2id)
        self.max_oov_count = args.max_oov_count
        self.pad_idx = vocab2id[PAD_WORD]

    def eval_predict(self, batch, model=None, delimiter=None):
        if model is None:
            model = self.model
        batch_size = len(batch[TOKENS])
        result_tensor = model.beam_search(batch[TOKENS], batch[TOKENS_OOV], batch[TOKENS_LENS],
                                          np.array([batch_size], dtype=np.int64))
        result_np = result_tensor.numpy()
        oov_list = batch[OOV_LIST]
        return self.__idx2result_beam(delimiter, oov_list, result_np)

    def predict(self, text_list):
        x_batch, x_oov_batch, sent_len_batch, oov_list_batch = self.generate_input_batch(text_list)
        batch_size = len(x_batch)
        batch_size_np = np.array([batch_size], dtype=np.long)
        result_tensor = self.model.beam_search(x_batch, x_oov_batch, sent_len_batch, batch_size_np)
        result_np = result_tensor.numpy()
        return self.__idx2result_beam('', oov_list_batch, result_np)

    def __idx2result_beam(self, delimiter, oov_list, result_sequences):
        results = []
        for batch_idx, batch in enumerate(result_sequences):
            beam_list = []
            item_oov_list = oov_list[batch_idx]
            for beam in batch:
                phrase = []
                for idx in beam:
                    if self.id2vocab.get(idx) == EOS_WORD:
                        break
                    if idx in self.id2vocab:
                        phrase.append(self.id2vocab[idx])
                    else:
                        oov_idx = idx - len(self.id2vocab)
                        if oov_idx < len(item_oov_list):
                            phrase.append(item_oov_list[oov_idx])
                        else:
                            phrase.append(UNK_WORD)

                if delimiter is not None:
                    phrase = delimiter.join(phrase)
                if phrase not in beam_list:
                    beam_list.append(phrase)
            results.append(beam_list)
        return results

    def generate_input_batch(self, text_list):
        x_batch = []
        x_oov_batch = []
        sent_len_batch = []
        oov_list_batch = []
        for text in text_list:
            tokens = token_char_tokenize(text)
            x, x_oov, oov_list, sent_len = self.generate_input(tokens)
            x_batch.append(x)
            x_oov_batch.append(x_oov)
            sent_len_batch.append(sent_len)
            oov_list_batch.append(oov_list)
        x_batch = tf.convert_to_tensor(x_batch)
        x_oov_batch = tf.convert_to_tensor(x_oov_batch)
        sent_len_batch = tf.convert_to_tensor(sent_len_batch)
        return x_batch, x_oov_batch, sent_len_batch, oov_list_batch

    def generate_input(self, tokens):
        if len(tokens) > self.max_src_len:
            tokens = tokens[:self.max_src_len]
        token_ids_with_oov = []
        token_ids = []
        oov_list = []

        for token in tokens:
            idx = self.vocab2id.get(token, self.vocab_size)
            if idx == self.vocab_size:
                token_ids.append(self.vocab2id[UNK_WORD])
                if token not in oov_list:
                    if len(oov_list) >= self.max_oov_count:
                        token_ids_with_oov.append(self.vocab_size + self.max_oov_count - 1)
                    else:
                        token_ids_with_oov.append(self.vocab_size + len(oov_list))
                        oov_list.append(token)
                else:
                    token_ids_with_oov.append(self.vocab_size + oov_list.index(token))
            else:
                token_ids.append(idx)
                token_ids_with_oov.append(idx)
        sent_len = len(token_ids)

        if len(token_ids) < self.max_src_len:
            token_ids.extend([PAD_WORD] * (self.max_src_len - len(token_ids)))
            token_ids_with_oov.extend([PAD_WORD] * (self.max_src_len - len(token_ids)))
        elif len(token_ids) > self.max_src_len:
            token_ids = token_ids[:self.max_src_len]
            token_ids_with_oov = token_ids_with_oov[:self.max_src_len]
            sent_len = self.max_src_len
        return token_ids, token_ids_with_oov, oov_list, sent_len
