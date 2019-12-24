#!/bin/bash
ROOT_PATH=${PWD}
DATA_DIR=$ROOT_PATH/data
SRC_TRAIN=$DATA_DIR/raw/kp20k_new/kp20k_training.json
SRC_VALID=$DATA_DIR/raw/kp20k_new/kp20k_validation.json
SRC_TEST=$DATA_DIR/raw/kp20k_new/kp20k_testing.json
DEST_TRAIN=$DATA_DIR/kp20k.train.jsonl
DEST_VALID=$DATA_DIR/kp20k.valid.jsonl
DEST_TEST=$DATA_DIR/kp20k.test.jsonl
DEST_VOCAB=$DATA_DIR/kp20k_vocab.txt

python3 deep_keyphrase/data_process/preprocess.py -src_filename $SRC_TRAIN \
        -dest_filename $DEST_TRAIN -dest_vocab_path $DEST_VOCAB -src_lower -target_lower
python3 deep_keyphrase/data_process/preprocess.py -src_filename $SRC_VALID \
        -dest_filename $DEST_VALID -src_lower -target_lower
python3 deep_keyphrase/data_process/preprocess.py -src_filename $SRC_TEST \
        -dest_filename $DEST_TEST -src_lower -target_lower
