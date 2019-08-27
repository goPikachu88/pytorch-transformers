#!/usr/bin/env bash

set -eux
export CUDA_VISIBLE_DEVICES=1
export TASK_NAME=tacred
export OUTPUT_FOLDER=max_seq_256

python run_relation_classifier.py \
    --model_type roberta \
    --model_name_or_path output/$TASK_NAME/$OUTPUT_FOLDER \
    --task_name $TASK_NAME \
    --do_test \
    --do_lower_case \
    --data_dir dataset/$TASK_NAME \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 16   \
    --output_dir output/$TASK_NAME/$OUTPUT_FOLDER
