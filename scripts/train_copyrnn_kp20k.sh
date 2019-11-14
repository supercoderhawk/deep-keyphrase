#!/bin/bash
ROOT_PATH=${PWD}
DATA_DIR=$ROOT_PATH/data
SRC_FILENAME=$DATA_DIR/kp20k_train.jsonl
VOCAB_PATH=$DATA_DIR/vocab_kp20k.txt
DEST_DIR=$DATA_DIR/kp20k/
#export CUDA_VISIBLE_DEVICES=1

python3 deep_keyphrase/copy_rnn/train.py -src_filename $SRC_FILENAME \
  -vocab_path $VOCAB_PATH -dest_dir $DEST_DIR
