# -*- coding: UTF-8 -*-
import copy
from operator import itemgetter
from collections import OrderedDict
from pysenal import read_jsonline, read_json


class KeyphraseEvaluator(object):
    def __init__(self, top_n,
                 metrics_mode,
                 token_field='tokens',
                 true_keyphrase_field='keyphrases',
                 pred_keyphrase_field='pred_keyphrases'):
        self.top_n = top_n
        self.metrics_mode = metrics_mode
        self.token_field = token_field
        self.true_keyphrase_field = true_keyphrase_field
        self.pred_keyphrase_field = pred_keyphrase_field

    def __load_data(self, input_data):
        if isinstance(input_data, str):
            if input_data.endswith('.json'):
                data_source = read_json(input_data)
            elif input_data.endswith('.jsonl'):
                data_source = read_jsonline(input_data)
            else:
                raise ValueError('input file type is not supported, only support .json and .jsonl')
        elif isinstance(input_data, list):
            data_source = copy.deepcopy(input_data)
        else:
            raise TypeError('input data type error. only accept str (path) and  list.')
        return data_source

    def evaluate(self, input_data, eval_mode='all', ):
        data_source = self.__load_data(input_data)
        if self.metrics_mode == 'micro':
            result = self.evaluate_micro_average(data_source, eval_mode)
        elif self.metrics_mode == 'macro':
            result = self.evaluate_macro_average(data_source, eval_mode)
        else:
            raise ValueError('evaluation mode is error.')
        return result

    def evaluate_micro_average(self, data_source, eval_mode):
        self.__check_eval_mode(eval_mode)
        top_counter = OrderedDict()
        for n in self.top_n:
            true_positive_count = 0
            pred_count = 0
            true_count = 0
            for record in data_source:
                tokens = record[self.token_field]
                true_positive_phrase_list = []
                pred_phrase_list_topn = self.filter_phrase(record[self.pred_keyphrase_field],
                                                           eval_mode,
                                                           tokens,
                                                           n)
                true_phrase_list = self.filter_phrase(record[self.true_keyphrase_field], eval_mode, tokens)
                for predict_phrase in pred_phrase_list_topn:
                    if predict_phrase in true_phrase_list:
                        true_positive_phrase_list.append(predict_phrase)
                true_positive_count += len(true_positive_phrase_list)
                pred_count += len(pred_phrase_list_topn)
                true_count += len(true_phrase_list)
            if not pred_count:
                prec = 0
            else:
                prec = true_positive_count / pred_count
            if not true_count:
                recall = 0
            else:
                recall = true_positive_count / true_count
            if prec + recall == 0:
                f1 = 0
            else:
                f1 = 2 * prec * recall / (prec + recall)
            top_counter[n] = {'precision': prec,
                              'recall': recall,
                              'f1': f1}
        return top_counter

    def evaluate_macro_average(self, data_source, eval_mode):
        self.__check_eval_mode(eval_mode)
        top_counter = OrderedDict()
        for n in self.top_n:
            counter = []
            for record in data_source:
                tokens = record[self.token_field]
                true_positive_topn_phrase_list = []
                pred_phrase_list_topn = self.filter_phrase(record[self.pred_keyphrase_field],
                                                           eval_mode,
                                                           record[self.token_field],
                                                           n)
                true_phrase_list = self.filter_phrase(record[self.true_keyphrase_field], eval_mode, tokens)
                for predict_phrase in pred_phrase_list_topn:
                    if predict_phrase in true_phrase_list:
                        true_positive_topn_phrase_list.append(predict_phrase)
                if not pred_phrase_list_topn:
                    p = 0
                else:
                    p = len(true_positive_topn_phrase_list) / len(pred_phrase_list_topn)

                if not true_phrase_list:
                    continue
                else:
                    r = len(true_positive_topn_phrase_list) / len(true_phrase_list)

                if p + r == 0:
                    f1 = 0
                else:
                    f1 = 2 * p * r / (p + r)
                counter.append({'true_positive': len(true_positive_topn_phrase_list),
                                'pred_count': len(pred_phrase_list_topn),
                                'true_count': len(true_phrase_list),
                                'precision': p,
                                'recall': r,
                                'f1': f1})
            top_counter[n] = {'precision': sum(map(itemgetter('precision'), counter)) / len(counter),
                              'recall': sum(map(itemgetter('recall'), counter)) / len(counter),
                              'f1': sum(map(itemgetter('f1'), counter)) / len(counter)}
        return top_counter

    def filter_phrase(self, phrase_list, mode, input_tokens, top_n=None):
        input_text = ' '.join(input_tokens)
        filtered_phrase_list = []

        for phrase in phrase_list:
            phrase_text = ' '.join(phrase)
            if mode == 'all':
                filtered_phrase_list.append(phrase)
            elif mode == 'present' and phrase_text in input_text:
                filtered_phrase_list.append(phrase)
            elif mode == 'absent' and phrase_text not in input_text:
                filtered_phrase_list.append(phrase)

        if top_n is not None:
            filtered_phrase_list = filtered_phrase_list[:top_n]
        return filtered_phrase_list

    def __check_eval_mode(self, eval_mode):
        if eval_mode not in {'all', 'present', 'absent'}:
            raise ValueError('evaluation mode must be in `all`, `present` and `absent`')
