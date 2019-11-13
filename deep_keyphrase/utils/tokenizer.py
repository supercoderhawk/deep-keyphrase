# -*- coding: UTF-8 -*-
import re
from .constants import DIGIT_WORD

num_regex = re.compile(r'\d+([.]\d+)?')

char_regex = re.compile(r'[_\-—<>{,(?\\.\'%]|\d+([.]\d+)?', re.IGNORECASE)


def token_char_tokenize(text):
    text = char_regex.sub(r' \g<0> ', text)
    tokens = num_regex.sub(DIGIT_WORD, text).split()
    chars = []
    for token in tokens:
        if token == DIGIT_WORD:
            chars.append(token)
        else:
            chars.extend(list(token))
    return chars
