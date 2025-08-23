#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/run_pretraining.log"

# exec > "$LOG_FILE" 2>&1

echo cuda_visible_devices: $CUDA_VISIBLE_DEVICES

module load Python/3.11.5-GCCcore-13.2.0

# Change to project directory
cd /home/mg873uh/Fomo25/baseline-codebase

# Launch training with torchrun (for PyTorch DistributedDataParallel)
python src/pretrain.py \
    --save_dir=/home/mg873uh/Fomo25/baseline-codebase/_models \
    --pretrain_data_dir=/projects/p1170-25-1/data/pretrain_preproc/FOMO60k_2.667mm_float16 \
    --model_name=unet_b_lw_dec \
    --patch_size=96 \
    --batch_size=4 \
    --epochs=100 \
    --warmup_epochs=5 \
    --num_workers=64 \
    --augmentation_preset=all

