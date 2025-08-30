#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ask user for task ID
read -p "Enter task ID (1, 2, or 3): " TASKID

# Validate input
if [[ ! "$TASKID" =~ ^[1-3]$ ]]; then
    echo "Error: Invalid task ID. Please enter 1, 2, or 3."
    exit 1
fi

echo cuda_visible_devices: $CUDA_VISIBLE_DEVICES

# Activate environment
source /home/mg873uh/Projects_kb/.venv_fomo/bin/activate

# Change to project directory
cd /home/mg873uh/Projects_kb/baseline-codebase

echo "Running hierarchical finetuning for task ${TASKID}..."

python src/finetune_hierarchical.py \
    --task_id ${TASKID} \
    --model_type regression \
    --global_encoder unet_b_lw_dec \
    --local_data_dir /home/mg873uh/Projects_kb/data/finetuning_preproc \
    --global_data_dir /home/mg873uh/Projects_kb/data/finetuning_preproc/Unified_2.6667mm_float16 \
    --save_dir /home/mg873uh/Projects_kb/baseline-codebase/_models/hierarchical \
    --local_checkpoint /home/mg873uh/Projects_kb/baseline-codebase/_scripts/26_8.ckpt \
    --global_checkpoint /home/mg873uh/Projects_kb/baseline-codebase/_models/models/FOMO60k_2.667mm_float16/unet_b_lw_dec/versions/version_0/last.ckpt \
    --augmentation_preset basic \
    --use_balanced_dataset \
    --split_param 0.001 \
    --train_batches_per_epoch 100 \
    --patch_size 96 \
    --epochs 500 \
    --batch_size 4 \
    --experiment_name Hierarchical_finetune \
    --learning_rate 1e-4 \
    --precision bf16-mixed
