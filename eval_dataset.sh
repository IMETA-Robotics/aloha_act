#!/bin/bash
task_name="y1_place_and_place_0828"
gpu_id=0

export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 eval_dataset.py \
    --task_name ${task_name} \
    --ckpt_dir output/act_ckpt/act-${task_name}/ \
    --ckpt_name policy_best.ckpt \
    --policy_class ACT \
    --chunk_size 25 \
    --control_rate 25 \
    --seed 0 \
    --temporal_agg \