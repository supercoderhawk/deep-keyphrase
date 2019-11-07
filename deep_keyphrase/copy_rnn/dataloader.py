# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
from pysenal import read_jsonline_lazy, get_chunk
from deep_keyphrase.utils.constants import *


class CopyRnnDataLoader(object):
    def __init__(self, filename, vocab2id, batch_size, max_src_len, max_oov_count, max_target_len, mode):
        self.filename = filename
        self.vocab2id = vocab2id
        self.vocab_size = len(self.vocab2id)
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_oov_count = max_oov_count
        self.max_target_len = max_target_len
        self.mode = mode

    def __iter__(self):
        return iter(CopyRnnDataIterator(self))

    def collate_fn(self, item):
        tokens = item['tokens']
        token_ids_with_oov = []
        token_ids = []
        oov_list = []
        target_ids_list = []
        for token in tokens:
            idx = self.vocab2id.get(token, self.vocab_size)
            if idx == self.vocab_size:
                token_ids.append(self.vocab2id[UNK_WORD])
                if token not in oov_list:
                    if len(oov_list) >= self.max_oov_count:
                        token_ids_with_oov.append(self.vocab_size + self.max_oov_count - 1)
                    else:
                        token_ids_with_oov.append(self.vocab_size + len(oov_list))
                        oov_list.append(oov_list)
                else:
                    token_ids_with_oov.append(self.vocab_size + oov_list.index(token))
            else:
                token_ids.append(idx)
                token_ids_with_oov.append(idx)
        for keyphrase in item['keyphrases']:
            target_ids = [self.vocab2id[BOS_WORD]]
            for token in keyphrase:
                target_ids.append(self.vocab2id.get(token, self.vocab2id[UNK_WORD]))
            target_ids.append(self.vocab2id[EOS_WORD])
            target_ids_list.append(target_ids)

        final_item = {'tokens': token_ids,
                      'tokens_with_oov': token_ids_with_oov,
                      'targets': target_ids_list,
                      'oov_count': len(oov_list),
                      'oov_list': oov_list}
        return final_item


class CopyRnnDataIterator(object):
    def __init__(self, loader):
        self.loader = loader
        self.filename = loader.filename
        self.batch_size = loader.batch_size

    def __iter__(self):
        if self.loader.mode == 'train':
            yield from self.iter_train()
        else:
            yield from self.iter_inference()

    def iter_train(self):
        batch = []
        for item in read_jsonline_lazy(self.filename):
            item = self.loader.collate_fn(item)
            tokens = item['tokens']
            token_with_oov = item['tokens_with_oov']
            oov_count = item['oov_count']
            flatten_items = []
            for phrase in item['targets']:
                one2one_item = {'tokens': tokens,
                                'tokens_with_oov': token_with_oov,
                                'oov_count': oov_count,
                                'target': phrase}
                flatten_items.append(one2one_item)
            if len(batch) + len(flatten_items) > self.batch_size:
                yield self.padding_batch_train(batch)
                batch = flatten_items
            else:
                batch.extend(flatten_items)
        if batch:
            yield self.padding_batch_train(batch)

    def iter_inference(self):
        chunk_gen = get_chunk(read_jsonline_lazy(self.filename), self.batch_size)
        for item_chunk in chunk_gen:
            item_chunk = [self.loader.collate_fn(item) for item in item_chunk]
            yield self.padding_batch_inference(item_chunk)

    def padding_batch_train(self, batch):
        name2max_len = {'tokens': self.loader.max_src_len,
                        'tokens_with_oov': self.loader.max_src_len,
                        'target': self.loader.max_target_len + 1}
        result = {}
        for key in batch[0].keys():
            data = [b[key] for b in batch]
            if key in name2max_len:
                pad_data, pad_data_len = self.__padding(data, name2max_len[key])
                if key == 'tokens':
                    src_tensor = torch.tensor(pad_data_len, dtype=torch.int64)
                    result['tokens_len'] = src_tensor
            else:
                pad_data = torch.tensor(data, dtype=torch.int64)
            result[key] = pad_data

        return result

    def padding_batch_inference(self, batch):
        name2max_len = {'tokens': self.loader.max_src_len,
                        'tokens_with_oov': self.loader.max_src_len}
        padding_key = {'tokens', 'tokens_with_oov','oov_count'}
        result = {}
        for key in batch[0].keys():
            data = [b[key] for b in batch]
            if key in padding_key:
                if key in name2max_len:
                    pad_data, pad_data_len = self.__padding(data, name2max_len[key])
                    if key == 'tokens':
                        src_tensor = torch.tensor(pad_data_len, dtype=torch.int64)
                        result['tokens_len'] = src_tensor
                else:
                    pad_data = torch.tensor(data, dtype=torch.int64)
                result[key] = pad_data
            else:
                result[key] = data
        return result

    def __padding(self, x_raw, max_len):
        x_raw = np.asarray(x_raw)
        x_len_list = []
        for x_ in x_raw:
            x_len = len(x_)
            if x_len > max_len:
                x_len_list.append(max_len)
            else:
                x_len_list.append(x_len)

        pad_id = self.loader.vocab2id[PAD_WORD]
        x = np.array([np.concatenate((x_[:max_len], [pad_id] * (max_len - len(x_)))) for x_ in x_raw])
        x = torch.tensor(x, dtype=torch.int64)
        return x, x_len_list
