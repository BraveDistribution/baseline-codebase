#!/bin/bash
#SBATCH --job-name=pretrain_unet
#SBATCH --output=/home/mg873uh/Projects_kb/baseline-codebase/slurm-%j.out
#SBATCH --error=/home/mg873uh/Projects_kb/baseline-codebase/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# Log script directory and setup logging
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/run_pretraining_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load necessary modules (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.9

# Activate virtual environment
source /home/mg873uh/Projects_kb/.venv_fomo/bin/activate

# Change to project directory
cd /home/mg873uh/Projects_kb/baseline-codebase

# Launch training
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

echo "Job finished at: $(date)"
