# -*- coding: UTF-8 -*-
import os
import argparse
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pysenal import get_chunk, read_jsonline_lazy, append_jsonlines


class Kp20kPreprocessor(object):

    def __init__(self, args):
        self.src_filename = args.src_filename
        self.dest_filename = args.dest_filename
        self.pool = Pool(args.parallel_count)
        self.is_src_lower = args.is_src_lower
        self.is_src_stem = args.is_src_stem
        self.is_tgt_lower = args.is_tgt_lower
        self.is_tgt_stem = args.is_tgt_stem
        self.stemmer = PorterStemmer()
        if os.path.exists(self.dest_filename):
            print('destination file existed, will be deleted!!!')
            os.remove(self.dest_filename)

    def process(self):
        chunk_size = 100
        for item_chunk in get_chunk(read_jsonline_lazy(self.src_filename), chunk_size):
            processed_records = self.pool.map(self.tokenize_record, item_chunk)
            append_jsonlines(self.dest_filename, processed_records)

    def tokenize_record(self, record):
        abstract_tokens = self.tokenize(record['abstract'], self.is_src_lower, self.is_src_stem)
        title_tokens = self.tokenize(record['title'], self.is_src_lower, self.is_src_stem)
        keyword_token_list = []
        for keyword in record['keyword'].split(';'):
            keyword_token_list.append(self.tokenize(keyword, self.is_tgt_lower, self.is_tgt_stem))
        result = {'title_tokens': title_tokens, 'abstract_tokens': abstract_tokens,
                  'keyword_tokens': keyword_token_list}
        return result

    def tokenize(self, text, is_lower, is_stem):
        tokens = word_tokenize(text)
        if is_lower:
            tokens = [token.lower() for token in tokens]
        if is_stem:
            tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_filename', type=str)
    parser.add_argument('-dest_filename', type=str)
    args = parser.parse_args()
    processor = Kp20kPreprocessor(args)
    processor.process()


if __name__ == '__main__':
    main()
