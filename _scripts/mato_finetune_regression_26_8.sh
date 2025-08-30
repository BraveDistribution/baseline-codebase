#!/bin/bash
#SBATCH --job-name=finetune_regression
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=82G
#SBATCH --time=24:00:00
#SBATCH --partition=dgx

set -euo pipefail

# Create logs & checkpoint dirs if they don't exist
mkdir -p logs

# Activate venv
source /home/mg873uh/Projects_kb/.venv_fomo/bin/activate

# PyTorch distributed env (works fine for single-node too)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

# NCCL tuning
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Training parameters (NEW)
DATA_DIR="/home/mg873uh/Projects_kb/data/finetuning_preproc/Task003_FOMO3"
CHECKPOINT_DIR="/home/mg873uh/Projects_kb/finetuning_regression_29_8_final_model"
MODEL_CKPT="27_8.ckpt"
EXPERIMENT_NAME="finetuning_regression_29_8_bias_correction_final_model"

mkdir -p $CHECKPOINT_DIR

echo "SLURM allocated GPUs. CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

srun python /home/mg873uh/Projects_kb/baseline-codebase/src/finetune_classification_transformer.py \
    --data_dir="$DATA_DIR" \
    --save_checkpoint_dir="$CHECKPOINT_DIR" \
    --model_checkpoint="$MODEL_CKPT" \
    --experiment_name="$EXPERIMENT_NAME" \
    --num_epochs=300 \
    --batch_size=6 \
    --task_type=regression \
    --split_param 0.001

echo "Training completed!"

