#!/usr/bin/env bash

set -eux
export CUDA_VISIBLE_DEVICES=1
export TASK_NAME=tacred

python ./run_relation_classifier.py \
    --model_type roberta \
    --model_name_or_path dataset/roberta.base \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir dataset/$TASK_NAME \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 16   \
    --per_gpu_train_batch_size 16   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir output/$TASK_NAME/max_seq_256 \
    --save_steps 500
