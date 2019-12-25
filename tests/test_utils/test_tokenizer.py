# -*- coding: UTF-8 -*-
from deep_keyphrase.utils.tokenizer import token_char_tokenize


def test_token_char_tokenize():
    tokens = token_char_tokenize('1.1在10~11个之间。')
    assert tokens == ['<digit>', '在', '<digit>', '~', '<digit>', '个', '之', '间', '。']

    tokens = token_char_tokenize('1.发明内容11.11-11.12')
    assert tokens == ['<digit>', '.', '发', '明', '内', '容', '<digit>', '-', '<digit>']
