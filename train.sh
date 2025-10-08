#!/bin/bash
task_name="y1_place_and_place_0828"
gpu_id=0

export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 train.py \
    --task_name ${task_name} \
    --ckpt_dir ./output/act_ckpt/act-${task_name}/ \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 30 \
    --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 10000 \
    --lr 4e-5 \
    --seed 0