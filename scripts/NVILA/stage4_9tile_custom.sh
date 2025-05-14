#!/bin/bash

DEFAULT_RUN_NAME="NVILA-8B-finetune"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=1
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1
STAGE2_PATH="/home/workspace/NVILA-8B/"
DATA_MIXTURE="JRDB"
OUTPUT_DIR="/home/workspace/VILA/runs/train/$DEFAULT_RUN_NAME"

source /home/workspace/VILA/scripts/setups/custom_train.sh

export NCCL_IB_TIMEOUT=31

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    /home/workspace/VILA/llava/train/train_mem_ln.py \
        --deepspeed /home/workspace/VILA/scripts/zero3_gradient_clipping.json \
        --model_name_or_path $STAGE2_PATH \
        --data_mixture $DATA_MIXTURE  \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --dynamic_s2 True \
        --s2_scales "448,896,1344" \
        --s2_max_split_size 448 \
        --s2_resize_output_to_scale_idx -1 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample \
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
        --image_aspect_ratio dynamic_s2 \
        --max_grad_norm 5.0 \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 20 \
        --save_total_limit 1 \
        --learning_rate 1.5e-5 \
        --vision_tower_lr 3e-7 \
        --mm_projector_lr 6e-7 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 8192 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --report_to wandb
