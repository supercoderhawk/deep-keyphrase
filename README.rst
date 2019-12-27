==============
deep-keyphrase
==============


Implement some keyphrase generation algorithm

.. image:: https://img.shields.io/github/workflow/status/supercoderhawk/deep-keyphrase/ci.svg

.. image:: https://img.shields.io/pypi/v/deep-keyphrase.svg
    :target: https://pypi.org/project/deep-keyphrase

.. image:: https://img.shields.io/pypi/dm/deep-keyphrase.svg
    :target: https://pypi.org/project/pysenal


Description
===========
Implemented Paper
>>>>>>>>>>>>>>>>>>>>>

CopyRNN

`Deep Keyphrase Generation (Meng et al., 2017)`__

.. __: https://arxiv.org/abs/1704.06879


ToDo List
>>>>>>>>>>>>>>>

CopyCNN

CopyTransformer


Usage
============

required files (4 files in total)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

1. vocab_file: word line by line (don't with index!!!!) ::

    this
    paper
    proposes

2. training, valid and test file

data format for training, valid and test
""""""""""""""""""""""""""""""""""""""""""""""""""
json line format, every line is a dict::

    {'tokens': ['this', 'paper', 'proposes', 'using', 'virtual', 'reality', 'to', 'enhance', 'the', 'perception', 'of', 'actions', 'by', 'distant', 'users', 'on', 'a', 'shared', 'application', '.', 'here', ',', 'distance', 'may', 'refer', 'either', 'to', 'space', '(', 'e.g.', 'in', 'a', 'remote', 'synchronous', 'collaboration', ')', 'or', 'time', '(', 'e.g.', 'during', 'playback', 'of', 'recorded', 'actions', ')', '.', 'our', 'approach', 'consists', 'in', 'immersing', 'the', 'application', 'in', 'a', 'virtual', 'inhabited', '3d', 'space', 'and', 'mimicking', 'user', 'actions', 'by', 'animating', 'avatars', '.', 'we', 'illustrate', 'this', 'approach', 'with', 'two', 'applications', ',', 'the', 'one', 'for', 'remote', 'collaboration', 'on', 'a', 'shared', 'application', 'and', 'the', 'other', 'to', 'playback', 'recorded', 'sequences', 'of', 'user', 'actions', '.', 'we', 'suggest', 'this', 'could', 'be', 'a', 'low', 'cost', 'enhancement', 'for', 'telepresence', '.'] ,
    'keyphrases': [['telepresence'], ['animation'], ['avatars'], ['application', 'sharing'], ['collaborative', 'virtual', 'environments']]}


Training
>>>>>>>>>>>>>>>
download the kp20k_

.. _kp20k: https://drive.google.com/uc?id=1ZTQEGZSq06kzlPlOv4yGjbUpoDrNxebR&export=download

::

    mkdir data
    mkdir data/raw
    mkdir data/raw/kp20k_new
    # !! please unzip kp20k data put the files into above folder manually
    python -m nltk.downloader punkt
    bash scripts/prepare_kp20k.sh
    bash scripts/train_copyrnn_kp20k.sh

    # start tensorboard
    # enter the experiment result dir, suffix is time that experiment starts
    cd data/kp20k/copyrnn_kp20k_basic-20191212-080000
    # start tensorboard services
    tenosrboard --bind_all --logdir logs --port 6006

Notes
=============================
1. compared with the original :code:`seq2seq-keyphrase-pytorch`
    1. fix the implementation error:
        1. copy mechanism
        2. train and inference are not correspond (training doesn\'t have input feeding and inference has input feeding)
    2. easy data preparing
    3. tensorboard support
    4. **faster beam search (6x faster used cpu and more than 10x faster used gpu)**