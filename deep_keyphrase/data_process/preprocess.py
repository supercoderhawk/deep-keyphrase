# -*- coding: UTF-8 -*-
from multiprocessing import Pool
import en_core_web_sm
from pysenal import get_chunk, read_jsonline_lazy


class Kp20kPreprocessor(object):
    nlp = en_core_web_sm.load(disable=['ner', 'parser', 'textcat'])

    def __init__(self, src_filename, dest_filename, parallel_count=10):
        self.src_filename = src_filename
        self.dest_filename = dest_filename
        self.pool = Pool(parallel_count)

    def process(self):
        chunk_size = 100
        for item_chunk in get_chunk(read_jsonline_lazy(self.src_filename), chunk_size):
            self.pool.map(self.tokenize_record, item_chunk)

    def tokenize_record(self, record):
        abstract_tokens = self.tokenize(record['abstract'])
        title_tokens = self.tokenize(record['title'])
        keyword_token_list = []
        for keyword in record['keyword'].split(';'):
            keyword_token_list.append(self.tokenize(keyword))
        result = {'title_tokens': title_tokens, 'abstract_tokens': abstract_tokens,
                  'keyword_tokens': keyword_token_list}
        return result

    def tokenize(self, text, lower=True, stem=False):
        tokens = []
        for token in self.nlp(text):
            if lower:
                token_text = token.lower()
            else:
                token_text = token.text

            tokens.append(token_text)
        return tokens
