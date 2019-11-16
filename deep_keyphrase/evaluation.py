# -*- coding: UTF-8 -*-
from operator import itemgetter
from collections import OrderedDict
from pysenal import read_jsonline_lazy


class KeyphraseEvaluator(object):
    def __init__(self, top_n, metrics_mode):
        self.top_n = top_n
        self.metrics_mode = metrics_mode

    def evaluate(self, src_filename, eval_mode='all'):
        if self.metrics_mode == 'micro':
            result = self.evaluate_micro_average(src_filename, eval_mode)
        elif self.metrics_mode == 'macro':
            result = self.evaluate_macro_average(src_filename, eval_mode)
        else:
            raise ValueError('evaluation mode is error.')
        return result

    def evaluate_micro_average(self, src_filename, eval_mode):
        self.__check_eval_mode(eval_mode)
        top_counter = OrderedDict()
        for n in self.top_n:
            true_positive_count = 0
            pred_count = 0
            true_count = 0
            for record in read_jsonline_lazy(src_filename):
                tokens = record['tokens']
                true_positive_phrase_list = []
                pred_phrase_list_topn = self.filter_phrase(record['pred_keyphrases'],
                                                           eval_mode,
                                                           record['tokens'],
                                                           n)
                true_phrase_list = self.filter_phrase(record['keyphrases'], eval_mode, tokens)
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

    def evaluate_macro_average(self, src_filename, eval_mode):
        self.__check_eval_mode(eval_mode)
        top_counter = OrderedDict()
        for n in self.top_n:
            counter = []
            for record in read_jsonline_lazy(src_filename):
                tokens = record['tokens']
                true_positive_topn_phrase_list = []
                pred_phrase_list_topn = self.filter_phrase(record['pred_keyphrases'],
                                                           eval_mode,
                                                           record['tokens'],
                                                           n)
                true_phrase_list = self.filter_phrase(record['keyphrases'], eval_mode, tokens)
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
