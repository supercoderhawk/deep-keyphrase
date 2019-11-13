# -*- coding: UTF-8 -*-
from operator import itemgetter
from collections import OrderedDict
from pysenal import read_jsonline_lazy


class KeyphraseEvaluator(object):
    def __init__(self, top_n, eval_mode):
        self.top_n = top_n
        self.eval_mode = eval_mode

    def evaluate(self, src_filename):
        if self.eval_mode == 'micro':
            result = self.evaluate_micro_average(src_filename)
        elif self.eval_mode == 'macro':
            result = self.evaluate_macro_average(src_filename)
        else:
            raise ValueError('evaluation mode is error.')
        return result

    def evaluate_micro_average(self, src_filename):
        top_counter = OrderedDict()
        for n in self.top_n:
            true_positive_count = 0
            pred_count = 0
            true_count = 0
            for record in read_jsonline_lazy(src_filename):
                true_positive_phrase_list = []
                pred_phrase_list_topn = record['pred_keyphrases'][:n]
                true_phrase_list = record['keyphrases']
                for predict_phrase in pred_phrase_list_topn:
                    if predict_phrase in true_phrase_list:
                        true_positive_phrase_list.append(predict_phrase)
                true_positive_count += len(true_positive_phrase_list)
                pred_count += len(pred_phrase_list_topn)
                true_count += len(true_phrase_list)
            prec = true_positive_count / pred_count
            recall = true_positive_count / true_count
            if prec + recall == 0:
                f1 = 0
            else:
                f1 = 2 * prec * recall / (prec + recall)
            top_counter[n] = {'precision': prec,
                              'recall': recall,
                              'f1': f1}
        return top_counter

    def evaluate_macro_average(self, src_filename):
        top_counter = OrderedDict()
        for n in self.top_n:
            counter = []
            for record in read_jsonline_lazy(src_filename):
                true_positive_topn_phrase_list = []
                pred_phrase_list_topn = record['pred_keyphrases'][:n]
                true_phrase_list = record['keyphrases']
                for predict_phrase in pred_phrase_list_topn:
                    if predict_phrase in true_phrase_list:
                        true_positive_topn_phrase_list.append(predict_phrase)
                p = len(true_positive_topn_phrase_list) / len(pred_phrase_list_topn)
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
