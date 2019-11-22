# -*- coding: UTF-8 -*-
import os
import json
import torch
from collections import namedtuple, OrderedDict
from pysenal import read_file


class BasePredictor(object):
    def __init__(self, model_info):
        self.config = self.load_config(model_info)

    def load_config(self, model_info):
        if 'config' not in model_info:
            if isinstance(model_info['model'], str):
                config_path = os.path.splitext(model_info['model'])[0] + '.json'
            else:
                raise ValueError('config path is not assigned')
        else:
            config_info = model_info['config']
            if isinstance(config_info, str):
                config_path = config_info
            else:
                return config_info
        # json to object
        config = json.loads(read_file(config_path),
                            object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        return config

    def load_model(self, model_info, model):
        if isinstance(model_info['model'], torch.nn.Module):
            return model_info['model']

        model_path = model_info['model']
        if not isinstance(model_path, str):
            raise TypeError('model path should be str')
        # model = load_model_func()
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = OrderedDict()
        # avoid error when load parallel trained model
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                k = k[7:]
            state_dict[k] = v
        model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def predict(self, input_list, batch_size, delimiter=''):
        raise NotImplementedError('predict method is not implemented')
