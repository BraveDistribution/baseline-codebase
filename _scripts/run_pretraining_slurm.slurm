#!/bin/bash
#SBATCH --job-name=swin_pretrain
#SBATCH --output=/home/mg873uh/Projects_kb/baseline-codebase/logs/pretrain_%j.out
#SBATCH --error=/home/mg873uh/Projects_kb/baseline-codebase/logs/pretrain_%j.err
#SBATCH --nodes=1               # Start with 1 node, increase as needed
#SBATCH --ntasks-per-node=2      # Number of GPUs per node (adjust based on your cluster)
#SBATCH --gres=gpu:2             # Request 4 GPUs
#SBATCH --cpus-per-task=8        # CPUs per GPU (8 * 4 = 32 CPUs total)
#SBATCH --mem=128GB               # Total memory for the job
#SBATCH --time=24:00:00          # Maximum runtime (adjust as needed)
#SBATCH --partition=dgx          # GPU partition (check your cluster's partitions)

# Create log directory if it doesn't exist
mkdir -p /home/mg873uh/Projects_kb/baseline-codebase/logs

# Log some useful information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Activate environment
source /home/mg873uh/Projects_kb/.venv_fomo/bin/activate

# Change to project directory
cd /home/mg873uh/Projects_kb/baseline-codebase

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# PyTorch/NCCL settings for better performance
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


GPUS_PER_NODE=$SLURM_NTASKS_PER_NODE
BATCH_SIZE_PER_GPU=12
ACCUMULATE_GRAD_BATCHES=10  # 12 * 10 * 4 GPUs = 480 effective batch size

echo "Training with $GPUS_PER_NODE GPUs"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Gradient accumulation: $ACCUMULATE_GRAD_BATCHES"
echo "Effective batch size: $((BATCH_SIZE_PER_GPU * ACCUMULATE_GRAD_BATCHES * GPUS_PER_NODE))"

# Run training with srun for proper SLURM task distribution
srun python src/pretrain.py \
    --save_dir=/home/mg873uh/Projects_kb/baseline-codebase/models/mato \
    --pretrain_data_dir=/home/mg873uh/Projects_kb/data/pretrain_preproc/FOMO60k \
    --model_name=unet_b_lw_dec \
    --patch_size 96 96 96 \
    --batch_size=$BATCH_SIZE_PER_GPU \
    --epochs=100 \
    --warmup_epochs=5 \
    --num_workers=8 \
    --augmentation_preset=all \
    --model=contrastive \
    --accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \

echo "End time: $(date)"
echo "Job completed!"