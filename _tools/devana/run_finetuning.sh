#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ask user for task ID
read -p "Enter task ID (1, 2, or 3): " TASKID

# Validate input
if [[ ! "$TASKID" =~ ^[1-3]$ ]]; then
    echo "Error: Invalid task ID. Please enter 1, 2, or 3."
    exit 1
fi

LOG_FILE="$SCRIPT_DIR/run_finetuning_task${TASKID}.log"

#exec > "$LOG_FILE" 2>&1

echo cuda_visible_devices: $CUDA_VISIBLE_DEVICES

module load Python/3.11.5-GCCcore-13.2.0

# Change to project directory
cd /home/mg873uh/Fomo25/baseline-codebase

python src/finetune.py \
    --data_dir=/projects/p1170-25-1/data/finetuning_preproc \
    --save_dir=/home/mg873uh/Fomo25/baseline-codebase/_models/finetune \
    --pretrained_weights_path=/home/mg873uh/Fomo25/baseline-codebase/_models/models/FOMO60k_2.667mm_float16/unet_b_lw_dec/versions/version_0/last.ckpt \
    --model_name=unet_b \
    --patch_size=96 \
    --taskid=${TASKID} \
    --batch_size=4 \
    --epochs=500 \
    --train_batches_per_epoch=100 \
    --augmentation_preset=basic