# -*- coding: UTF-8 -*-
from pysenal import read_lines_lazy
from .constants import *


def load_vocab(src_filename, vocab_size=None):
    vocab2id = {}
    for word in read_lines_lazy(src_filename):
        if word not in vocab2id:
            vocab2id[word] = len(vocab2id)

        if vocab_size and len(vocab2id) >= vocab_size:
            break
    if PAD_WORD not in vocab2id:
        raise ValueError('padding char is not in vocab')
    if UNK_WORD not in vocab2id:
        raise ValueError('unk char is not in vocab')
    if BOS_WORD not in vocab2id:
        raise ValueError('begin of sentence char is not in vocab')
    if EOS_WORD not in vocab2id:
        raise ValueError('end of sentence char is not in vocab')
    # if DIGIT_WORD not in vocab2id:
    #     raise ValueError('digit char is not in vocab')
    # if SEP_WORD not in vocab2id:
    #     raise ValueError('separator char is not in vocab')
    return vocab2id
