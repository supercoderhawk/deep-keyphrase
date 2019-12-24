# -*- coding: UTF-8 -*-
import os
import re
import argparse
import string
from collections import Counter
from itertools import chain
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pysenal import get_chunk, read_jsonline_lazy, append_jsonlines, write_lines
from deep_keyphrase.utils.constants import (PAD_WORD, UNK_WORD, DIGIT_WORD,
                                            BOS_WORD, EOS_WORD, SEP_WORD)


class Kp20kPreprocessor(object):
    """
    kp20k data preprocessor, build the data and vocab for training.

    """
    num_and_punc_regex = re.compile(r'[_\-—<>{,(?\\.\'%]|\d+([.]\d+)?', re.IGNORECASE)
    num_regex = re.compile(r'\d+([.]\d+)?')

    def __init__(self, args):
        self.src_filename = args.src_filename
        self.dest_filename = args.dest_filename
        self.dest_vocab_path = args.dest_vocab_path
        self.vocab_size = args.vocab_size
        self.parallel_count = args.parallel_count
        self.is_src_lower = args.src_lower
        self.is_src_stem = args.src_stem
        self.is_target_lower = args.target_lower
        self.is_target_stem = args.target_stem
        self.stemmer = PorterStemmer()
        if os.path.exists(self.dest_filename):
            print('destination file existed, will be deleted!!!')
            os.remove(self.dest_filename)

    def build_vocab(self, tokens):
        vocab = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD, DIGIT_WORD, SEP_WORD]
        vocab.extend(list(string.digits))

        token_counter = Counter(tokens).most_common(self.vocab_size)
        for token, count in token_counter:
            vocab.append(token)
            if len(vocab) >= self.vocab_size:
                break
        return vocab

    def process(self):
        pool = Pool(self.parallel_count)
        tokens = []
        chunk_size = 100
        for item_chunk in get_chunk(read_jsonline_lazy(self.src_filename), chunk_size):
            processed_records = pool.map(self.tokenize_record, item_chunk)
            if self.dest_vocab_path:
                for record in processed_records:
                    tokens.extend(record['title_and_abstract_tokens'] + record['flatten_keyword_tokens'])
            for record in processed_records:
                record.pop('flatten_keyword_tokens')
            append_jsonlines(self.dest_filename, processed_records)
        if self.dest_vocab_path:
            vocab = self.build_vocab(tokens)
            write_lines(self.dest_vocab_path, vocab)

    def tokenize_record(self, record):
        abstract_tokens = self.tokenize(record['abstract'], self.is_src_lower, self.is_src_stem)
        title_tokens = self.tokenize(record['title'], self.is_src_lower, self.is_src_stem)
        keyword_token_list = []
        for keyword in record['keyword'].split(';'):
            keyword_token_list.append(self.tokenize(keyword, self.is_target_lower, self.is_target_stem))
        result = {
            # 'title_tokens': title_tokens, 'abstract_tokens': abstract_tokens,
            'title_and_abstract_tokens': title_tokens + abstract_tokens,
            'keyword_tokens': keyword_token_list,
            'flatten_keyword_tokens': list(chain(*keyword_token_list))
        }
        return result

    def tokenize(self, text, is_lower, is_stem):
        text = self.num_and_punc_regex.sub(r' \g<0> ', text)
        tokens = word_tokenize(text)
        if is_lower:
            tokens = [token.lower() for token in tokens]
        if is_stem:
            tokens = [self.stemmer.stem(token) for token in tokens]
        for idx, token in enumerate(tokens):
            token = tokens[idx]
            if self.num_regex.fullmatch(token):
                tokens[idx] = DIGIT_WORD
        return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_filename', type=str, required=True,
                        help='input source kp20k file path')
    parser.add_argument('-dest_filename', type=str, required=True,
                        help='destination of processed file path')
    parser.add_argument('-dest_vocab_path', type=str,
                        help='')
    parser.add_argument('-vocab_size', type=int, default=50000,
                        help='')
    parser.add_argument('-parallel_count', type=int, default=10)
    parser.add_argument('-src_lower', action='store_true')
    parser.add_argument('-src_stem', action='store_true')
    parser.add_argument('-target_lower', action='store_true')
    parser.add_argument('-target_stem', action='store_true')

    args = parser.parse_args()
    processor = Kp20kPreprocessor(args)
    processor.process()


if __name__ == '__main__':
    main()
