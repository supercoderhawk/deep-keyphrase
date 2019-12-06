#!/bin/bash
ROOT_PATH=${PWD}
DATA_DIR=$ROOT_PATH/data
TRAIN_FILENAME=$DATA_DIR/kp20k.train.jsonl
VALID_FILENAME=$DATA_DIR/kp20k.valid.jsonl
TEST_FILENAME=$DATA_DIR/kp20k.test.jsonl
VOCAB_PATH=$DATA_DIR/vocab_kp20k.txt
DEST_DIR=$DATA_DIR/kp20k/
EXP_NAME=copyrnn_kp20k_basic

#export CUDA_VISIBLE_DEVICES=1

python3 deep_keyphrase/copy_rnn/train.py -exp_name $EXP_NAME \
  -train_filename $TRAIN_FILENAME \
  -valid_filename $VALID_FILENAME -test_filename $TEST_FILENAME \
  -vocab_path $VOCAB_PATH -dest_dir $DEST_DIR \
  -bidirectional -teacher_forcing -copy_net
