#!/bin/bash
#SBATCH --job-name=finetune_classification
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=dgx

# Create logs directory if it doesn't exist
mkdir -p logs

source /home/mg873uh/Projects_kb/.venv_fomo/bin/activate

# Set PyTorch distributed training environment variables
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

# NCCL settings for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Training parameters
DATA_DIR="/home/mg873uh/Projects_kb/data/finetuning_preproc/Task003_FOMO3"
CHECKPOINT_DIR="/home/mg873uh/Projects_kb/checkpoints_finetune2"
CHECKPOINT_SAVED="/home/mg873uh/Projects_kb/checkpoints/contrastive_2gpu_1131/last2.ckpt"
EXPERIMENT_NAME="regression_${SLURM_JOB_ID}"
echo "SLURM allocated GPUs. CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"
srun python /home/mg873uh/Projects_kb/baseline-codebase/src/finetune_classification_transformer.py \
    --data_dir=$DATA_DIR \
    --save_checkpoint_dir=$CHECKPOINT_DIR \
    --model_checkpoint=$CHECKPOINT_SAVED \
    --experiment_name=$EXPERIMENT_NAME \
    --num_epochs=100 \
    --batch_size=10 \
    --patch_size=96 \
    --task_type=regression

echo "Training completed!"