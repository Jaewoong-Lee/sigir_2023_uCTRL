# -*- coding: utf-8 -*-
# @Time   : 2022/3/24
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import importlib
import torch

from recbole_debias.utils.enum_type import ModelType


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = [
        'debiased_recommender'
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['recbole_debias.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        model_name (str): model name

    Returns:
        Trainer: trainer class
    """
    try:
        if 'DICE' in model_name:
            return getattr(importlib.import_module('recbole_debias.trainer'), 'DICE' + 'Trainer')
        return getattr(importlib.import_module('recbole_debias.trainer'), model_name + 'Trainer')
    except AttributeError:
        return getattr(importlib.import_module('recbole_debias.trainer'), 'DebiasTrainer')


def load_pretrained_model(model, model_path):

    checkpoint = torch.load(model_path)
    model.load_state_dict(state_dict=checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    message_output = 'Loading pretrained model from {}'.format(model_path)

    return model,message_output