==============
deep-keyphrase
==============


Implement some keyphrase generation algorithm

.. image:: https://img.shields.io/github/workflow/status/supercoderhawk/deep-keyphrase/ci   :alt: GitHub Workflow Status



Description
===========
Implemented Paper
>>>>>>>>>>>>>>>>>>>>>

CopyRNN

`Deep Keyphrase Generation (Meng et al., 2017)`__

.. __: https://arxiv.org/abs/1704.06879


CopyCNN

CopyTransformer


Usage
============

required files (4 files in total)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

1. vocab_file: word line by line (don't with index!!!!)

2. training, valid and test file

data format for training, valid and test
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
json line format, every line is a dict



Notes
=============================
1. compared with the original :code:`seq2seq-keyphrase-pytorch`
    1. fix the implementation error:
        1. copy mechanism
        2. train and inference are not correspond (training doesn\'t have input feeding and inference has input feeding)
    2. easy data preparing
    3. tensorboard support