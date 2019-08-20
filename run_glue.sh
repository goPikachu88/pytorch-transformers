#!/usr/bin/env bash

set -eux
export CUDA_VISIBLE_DEVICES=7
export GLUE_DIR=dataset/glue
export TASK_NAME=MRPC

python ./examples/run_glue.py \
    --model_type roberta \
    --model_name_or_path dataset/roberta.base \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir output/$TASK_NAME/