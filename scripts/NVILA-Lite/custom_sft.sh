#!/bin/bash

DEFAULT_RUN_NAME="NVILA-Lite-2B-finetune-efficient"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=1
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

STAGE_PATH=${1:-"/home/workspace/NVILA-Lite-2B"}
DATA_MIXTURE=${2:-"JDRB"}
OUTPUT_DIR=${3:-"/home/workspace/VILA/runs/train/$DEFAULT_RUN_NAME"}

source /home/workspace/VILA/scripts/setups/custom_train.sh

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    /home/workspace/VILA/llava/train/train_mem_ln.py \
        --deepspeed /home/workspace/VILA/scripts/zero3.json \
        --model_name_or_path $STAGE_PATH \
        --data_mixture $DATA_MIXTURE \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample_3x3_fix \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --tune_language_model False \
        --lora_enable True \
        --lora_r 4 \
        --lora_alpha 4 \
        --lora_dropout 0.05 \
        --lora_llm True \
        --lora_vt False \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio dynamic \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --vision_tower_lr 4e-7 \
        --mm_projector_lr 1e-6 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --report_to wandb
