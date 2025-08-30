#!/bin/bash
#SBATCH --job-name=finetune_regression
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

# Create logs & checkpoint dirs if they don't exist
mkdir -p logs

# Activate venv
source /home/mg873uh/Projects_kb/.venv_fomo/bin/activate

# Training parameters (NEW)
DATA_DIR="/home/mg873uh/Projects_kb/data/finetuning_preproc/Task003_FOMO3"
MODEL_CKPT="/home/mg873uh/Projects_kb/baseline-codebase/_scripts/27_8.ckpt"
FOLD=4
CHECKPOINT_DIR="run_regression_FOLD_$FOLD"
EXPERIMENT_NAME="run_regression_FOLD_$FOLD"

mkdir -p $CHECKPOINT_DIR
cd /home/mg873uh/Projects_kb/baseline-codebase
echo "SLURM allocated GPUs. CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

srun python src/finetune_classification_transformer.py \
    --data_dir="$DATA_DIR" \
    --save_checkpoint_dir="$CHECKPOINT_DIR" \
    --model_checkpoint="$MODEL_CKPT" \
    --experiment_name="$EXPERIMENT_NAME" \
    --num_epochs=300 \
    --batch_size=6 \
    --task_type=regression \
    --split_method=kfold \
    --split_idx=$FOLD \
    --n_splits=5 \
    --aug_setup=basic

echo "Training completed!"
