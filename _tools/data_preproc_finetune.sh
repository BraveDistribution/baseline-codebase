#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../../.venv_fomo"
SRC_DIR="$SCRIPT_DIR/../src"

# Ask user for task ID
read -p "Enter task ID (1, 2, or 3): " TASKID

# Validate input
if [[ ! "$TASKID" =~ ^[1-3]$ ]]; then
    echo "Error: Invalid task ID. Please enter 1, 2, or 3."
    exit 1
fi

LOG_FILE="$SCRIPT_DIR/data_preproc_finetune_task${TASKID}.log"

# Redirect all output to log file
exec > "$LOG_FILE" 2>&1

# Check if we are in the correct virtual environment
if [[ "$VIRTUAL_ENV" != *".venv_fomo"* ]]; then
    echo "Not in .venv_fomo virtual environment. Activating..."

    # Check if .venv_fomo exists
    if [ ! -d "$VENV_DIR" ]; then
        echo "Error: .venv_fomo directory not found!"
        echo "Please create the virtual environment first with: uv venv .venv_fomo"
        exit 1
    fi

    # Activate the virtual environment
    source $VENV_DIR/bin/activate
    echo "Activated .venv_fomo virtual environment"
else
    echo "Already in .venv_fomo virtual environment"
fi

# Set source path based on task ID
SOURCE_PATH="/home/mg873uh/Projects_kb/data/finetuning/fomo-task${TASKID}"
OUTPUT_PATH="/home/mg873uh/Projects_kb/data/finetuning_preproc"

# Run the preprocessing script
echo "Running preprocessing script for task ${TASKID}..."
python $SRC_DIR/data/preprocess/run_preprocessing.py --taskid=${TASKID} --source_path=${SOURCE_PATH} --output_path=${OUTPUT_PATH}