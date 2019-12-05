# -*- coding: UTF-8 -*-
from deep_keyphrase.utils.tokenizer import token_char_tokenize


def test_token_char_tokenize():
    tokens = token_char_tokenize('1.在10~11个之间。')
    assert tokens == ['<digit>', '.', '在', '<digit>', '~', '<digit>', '个', '之', '间', '。']
