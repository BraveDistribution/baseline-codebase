#!/bin/bash
#SBATCH --job-name=contrastive_pretrain
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=3
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
DATA_DIR="/home/mg873uh/Projects_kb/data/pretrain_preproc/FOMO60k"
CHECKPOINT_DIR="/home/mg873uh/Projects_kb/checkpoints"
EXPERIMENT_NAME="contrastive_2gpu_${SLURM_JOB_ID}"

echo "SLURM allocated GPUs. CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# Run training script
srun python /home/mg873uh/Projects_kb/baseline-codebase/src/pretrain_transformer.py \
    --data_dir=$DATA_DIR \
    --model_checkpoint_dir=$CHECKPOINT_DIR \
    --experiment_name=$EXPERIMENT_NAME \
    --epochs=100 \
    --batch_size=10 \
    --learning_rate=1e-4 \
    --mae_mask_ratio=0.6 \
    --mask_patch_size=4 \
    --accumulate_grad_batches=5 \
    --checkpoint_every_n_epoch=5

echo "Training completed!"