# -*- coding: UTF-8 -*-
import random
import time
import traceback
import sys
import threading
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from pysenal import read_jsonline_lazy, get_chunk
from deep_keyphrase.utils.constants import *

TOKENS = 'tokens'
TOKENS_LENS = 'tokens_len'
TOKENS_OOV = 'tokens_with_oov'
OOV_COUNT = 'oov_count'
OOV_LIST = 'oov_list'
TARGET_LIST = 'targets'
TARGET = 'target'
RAW_BATCH = 'raw'

TRAIN_MODE = 'train'
EVAL_MODE = 'eval'
INFERENCE_MODE = 'inference'


class ExceptionWrapper(object):
    """
    Wraps an exception plus traceback to communicate across threads
    """

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


class KeyphraseDataLoader(object):
    def __init__(self, data_source, vocab2id, batch_size, max_src_len,
                 max_oov_count, max_target_len, mode,
                 pre_fetch=False,
                 shuffle_in_batch=False,
                 token_field='tokens',
                 keyphrase_field='keyphrases'):
        self.data_source = data_source
        self.vocab2id = vocab2id
        self.vocab_size = len(self.vocab2id)
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_oov_count = max_oov_count
        self.max_target_len = max_target_len
        self.mode = mode
        self.pre_fetch = pre_fetch
        self.shuffle_in_batch = shuffle_in_batch
        self.token_field = token_field
        self.keyphrases_field = keyphrase_field

    def __iter__(self):
        return iter(KeyphraseDataIterator(self))

    def collate_fn(self, item, is_inference=False):
        tokens = item[self.token_field]
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

        final_item = {TOKENS: token_ids,
                      TOKENS_OOV: token_ids_with_oov,
                      OOV_COUNT: len(oov_list),
                      OOV_LIST: oov_list}

        if is_inference:
            final_item[RAW_BATCH] = item
        else:
            keyphrase = item['phrase']
            target_ids = [self.vocab2id[BOS_WORD]]
            for token in keyphrase:
                target_ids.append(self.vocab2id.get(token, self.vocab2id[UNK_WORD]))
            target_ids.append(self.vocab2id[EOS_WORD])
            final_item[TARGET] = target_ids
        return final_item


class KeyphraseDataIterator(object):
    def __init__(self, loader):
        self.loader = loader
        self.data_source = loader.data_source
        self.batch_size = loader.batch_size
        self.num_workers = 8
        if self.loader.mode == TRAIN_MODE:
            self.chunk_size = self.num_workers * 50
        else:
            self.chunk_size = self.batch_size
        self._data = self.load_data(self.chunk_size)
        self._batch_count_in_output_queue = 0
        self._redundant_batch = []
        self.workers = []
        self.worker_shutdown = False

        if self.loader.mode in {TRAIN_MODE, EVAL_MODE}:
            self.input_queue = multiprocessing.Queue(1000 * self.num_workers)
            self.output_queue = multiprocessing.Queue(1000 * self.num_workers)
            self.done_event = threading.Event()

            for _ in range(self.num_workers):
                worker = multiprocessing.Process(
                    target=self._train_data_worker_loop
                )
                self.workers.append(worker)
            for worker in self.workers:
                worker.daemon = True
                worker.start()

        if self.loader.mode in {TRAIN_MODE, EVAL_MODE} or self.loader.pre_fetch:
            self.__prefetch()

    def __iter__(self):
        if self.loader.mode == TRAIN_MODE:
            yield from self.iter_train()
        else:
            yield from self.iter_inference()

    def load_data(self, chunk_size):
        if isinstance(self.data_source, str):
            data = read_jsonline_lazy(self.data_source)
        elif isinstance(self.data_source, list):
            data = iter(self.data_source)
        else:
            raise TypeError('input filename type is error')
        return get_chunk(data, chunk_size)

    def _train_data_worker_loop(self):
        while True:
            if self.done_event.is_set():
                return
            raw_batch = self.input_queue.get()
            # exit signal
            if raw_batch is None:
                break
            try:
                batch = []
                for item in raw_batch:
                    batch.append(self.loader.collate_fn(item))
                if self.loader.shuffle_in_batch:
                    random.shuffle(batch)
                batch = self.padding_batch_train(batch)
                self.output_queue.put(batch)
            except Exception as e:
                self.output_queue.put(ExceptionWrapper(sys.exc_info()))

    def __prefetch(self):
        item_chunk = next(self._data)
        batches, redundant_batch = self.get_batches(item_chunk, [])
        self._redundant_batch = redundant_batch
        for batch in batches:
            self.input_queue.put(batch)
            self._batch_count_in_output_queue += 1

    def iter_train(self):
        redundant_batch = self._redundant_batch
        for item_chunk in self._data:
            batches, redundant_batch = self.get_batches(item_chunk, redundant_batch)
            batch_idx = 0
            for idx in range(self._batch_count_in_output_queue):
                if batch_idx < len(batches):
                    self.input_queue.put(batches[batch_idx])
                    batch_idx += 1
                yield self.output_queue.get()

            if batch_idx < len(batches):
                for batch in batches[batch_idx:]:
                    self.input_queue.put(batch)

            self._batch_count_in_output_queue = len(batches)
        if redundant_batch:
            self.input_queue.put(redundant_batch)
            yield self.output_queue.get()

    def get_batches(self, item_chunk, batch):
        batches = []

        for item in item_chunk:
            if batch and len(batch) > self.batch_size:
                for sliced_batch in get_chunk(batch, self.batch_size):
                    batches.append(sliced_batch)
                batch = []
            flatten_items = self.flatten_raw_item(item)
            if batch and len(batch) + len(flatten_items) > self.batch_size:
                batches.append(batch)
                batch = flatten_items
            else:
                batch.extend(flatten_items)
        batches = self.reorder_batch_list(batches)
        return batches, batch

    def reorder_batch(self, batch):
        seq_idx_and_len = [(idx, len(item[TOKENS])) for idx, item in enumerate(batch)]
        seq_idx_and_len = sorted(seq_idx_and_len, key=lambda i: i[1], reverse=True)
        batch = [batch[idx] for idx, _ in seq_idx_and_len]
        return batch

    def reorder_batch_list(self, batches):
        new_batches = []
        for batch in batches:
            new_batches.append(self.reorder_batch(batch))
        return new_batches

    def flatten_raw_item(self, item):
        flatten_items = []
        for phrase in item[self.loader.keyphrases_field]:
            flatten_items.append({'tokens': item['tokens'], 'phrase': phrase})
        return flatten_items

    def iter_inference(self):
        for item_chunk in self._data:
            # item_chunk is same as a batch
            item_chunk = [self.loader.collate_fn(item, is_inference=True) for item in item_chunk]
            if len(item_chunk) > 1:
                item_chunk = self.reorder_batch(item_chunk)
            yield self.padding_batch_inference(item_chunk)

    def padding_batch_train(self, batch):
        name2max_len = {TOKENS: self.loader.max_src_len,
                        TOKENS_OOV: self.loader.max_src_len,
                        TARGET: self.loader.max_target_len + 1}
        result = {}
        for key in batch[0].keys():
            data = [b[key] for b in batch]
            if key in name2max_len:
                pad_data, pad_data_len = self.__padding(data, name2max_len[key])
                if key == TOKENS:
                    src_tensor = torch.tensor(pad_data_len, dtype=torch.int64)
                    result[TOKENS_LENS] = src_tensor
            else:
                pad_data = torch.tensor(data, dtype=torch.int64)
            result[key] = pad_data

        return result

    def padding_batch_inference(self, batch):
        name2max_len = {TOKENS: self.loader.max_src_len,
                        TOKENS_OOV: self.loader.max_src_len}
        padding_key = {TOKENS, TOKENS_OOV, OOV_COUNT}
        result = {}
        for key in batch[0].keys():
            data = [b[key] for b in batch]
            if key in padding_key:
                if key in name2max_len:
                    pad_data, pad_data_len = self.__padding(data, name2max_len[key])
                    if key == TOKENS:
                        src_tensor = torch.tensor(pad_data_len, dtype=torch.int64)
                        result[TOKENS_LENS] = src_tensor
                else:
                    pad_data = torch.tensor(data, dtype=torch.int64)
                result[key] = pad_data
            else:
                result[key] = data
        result[RAW_BATCH] = [i[RAW_BATCH] for i in batch]
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

    def _shutdown_workers(self):
        # print('shutdown workers')
        if not self.worker_shutdown and self.loader.mode in {'train', 'eval'}:
            self.worker_shutdown = True
            self.done_event.set()
            for _ in self.workers:
                self.input_queue.put(None)
                time.sleep(1)

            for worker in self.workers:
                worker.terminate()

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()
