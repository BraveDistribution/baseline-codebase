#!/bin/bash

# Activate environment
source /home/mg873uh/Projects_kb/.venv_fomo/bin/activate

# Change to project directory
cd /home/mg873uh/Projects_kb/baseline-codebase

# Launch training with torchrun (for PyTorch DistributedDataParallel)
python src/pretrain.py \
    --save_dir=/home/mg873uh/Projects_kb/baseline-codebase/_models \
    --pretrain_data_dir=/home/mg873uh/Projects_kb/data/pretrain_preproc/FOMO60k \
    --model_name=unet_b_lw_dec \
    --patch_size=96 \
    --batch_size=2 \
    --epochs=100 \
    --warmup_epochs=5 \
    --num_workers=64 \
    --augmentation_preset=all