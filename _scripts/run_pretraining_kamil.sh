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
python src/pretrain_mae_kamil.py \
    --save_dir=/home/mg873uh/Projects_kb/baseline-codebase/models/mato \
    --pretrain_data_dir=/home/mg873uh/Projects_kb/data/pretrain_preproc/FOMO60k \
    --patch_size=16 \
    --batch_size=8 \
    --epochs=100 \
    --num_workers=16 \
    --augmentation_preset=all \
    --accumulate_grad_batches=40
#   --new_version