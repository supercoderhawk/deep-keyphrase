# -*- coding: UTF-8 -*-
from pysenal import read_lines_lazy
from .constants import *


def load_vocab(src_filename, vocab_size=None):
    vocab2id = {PAD_WORD: 0,
                UNK_WORD: 1,
                BOS_WORD: 2,
                EOS_WORD: 3,
                DIGIT_WORD: 4,
                SEP_WORD: 5}
    for word in read_lines_lazy(src_filename):
        if word not in vocab2id:
            vocab2id[word] = len(vocab2id)

        if vocab_size and len(vocab2id) >= vocab_size:
            break

    return vocab2id
