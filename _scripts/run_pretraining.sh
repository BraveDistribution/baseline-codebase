#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/run_pretraining.log"

# exec > "$LOG_FILE" 2>&1

echo cuda_visible_devices: $CUDA_VISIBLE_DEVICES

# Activate environment
source /home/mg873uh/Projects_kb/.venv_fomo/bin/activate

# Change to project directory
cd /home/mg873uh/Projects_kb/baseline-codebase

# Launch training with torchrun (for PyTorch DistributedDataParallel)
python src/pretrain.py \
    --save_dir=/home/mg873uh/Projects_kb/baseline-codebase/models/mato \
    --pretrain_data_dir=/home/mg873uh/Projects_kb/data/pretrain_preproc/FOMO60k_2.667mm_float16 \
    --model_name=unet_b_lw_dec \
    --patch_size 96 96 96 \
    --batch_size=8 \
    --epochs=100 \
    --warmup_epochs=5 \
    --num_workers=16 \
    --augmentation_preset=all \
    --model=contrastive \
    --accumulate_grad_batches=40 \
    --new_version