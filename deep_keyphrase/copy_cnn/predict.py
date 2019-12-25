# -*- coding: UTF-8 -*-
from deep_keyphrase.base_predictor import BasePredictor


class CopyCnnPredictor(BasePredictor):
    def __init__(self, model_info):
        super().__init__(model_info)

    def predict(self, input_list, batch_size, delimiter=''):
        pass

    def eval_predict(self):
        pass
