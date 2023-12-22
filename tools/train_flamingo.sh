#!/bin/bash
deepspeed --include=localhost:1,2,3 train_flamingo.py \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --learning_rate_pretraining_components 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 100 \
    --seed 1234 \
    --local_rank -1 \
    --gradient_checkpointing \
    --zero_stage 2 \
    --precision bf16 \
    --work_dir ../work_dir \
    # --enable_tensorboard