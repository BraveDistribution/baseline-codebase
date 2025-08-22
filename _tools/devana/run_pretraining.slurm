#!/bin/bash

#SBATCH --job-name=FOMO2025_noCrop_pretrining       # Job name
#SBATCH --account=p1170-25-1                        # Account name
#SBATCH --output=stdout.%J.out                      # Stdout
#SBATCH --error=stderr.%J.err                       # Stderr
#SBATCH --partition=gpu                             # Adjust to your partition
#SBATCH --nodes=1                                   # One node
#SBATCH --ntasks=1                                  # One task
#SBATCH --gpus=2                                    # Request GPUs
#SBATCH --cpus-per-task=16                          # Total CPUs (8 per GPU recommended)
#SBATCH --mem=128G                                  # Adjust as needed
#SBATCH --time=00:02:00                             # Adjust wall time


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
    --augmentation_preset=all_noCrop