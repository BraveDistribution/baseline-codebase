#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../../.venv_fomo"
SRC_DIR="$SCRIPT_DIR/../src"
LOG_FILE="$SCRIPT_DIR/data_preproc_pretrain.log"

# Redirect all output to log file
# exec > "$LOG_FILE" 2>&1

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

# Run the preprocessing script
echo "Running unifying script..."
python $SRC_DIR/data/fomo-60k/unify_shapes.py \
    --in_path=/home/mg873uh/Projects_kb/data/finetuning_preproc/Task003_FOMO3/ \
    --out_path=/home/mg873uh/Projects_kb/data/finetuning_preproc/Task003_FOMO3_2.667mm_float16/ \
    --target_element_type=float16 \
    --target_spacing=2.6667 \
    --target_shape 96 96 96 \
#    --do_conversion