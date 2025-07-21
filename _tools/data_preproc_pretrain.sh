#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../../.venv_fomo"
SRC_DIR="$SCRIPT_DIR/../src"

# Check if we are in the correct virtual environment
if [[ "$VIRTUAL_ENV" != *".venv_fomo"* ]]; then
    echo "Not in .venv_fomo virtual environment. Activating..."

    # Check if .venv_fomo exists
    if [ ! -d "$VENV_DIR/.venv_fomo" ]; then
        echo "Error: .venv_fomo directory not found!"
        echo "Please create the virtual environment first with: uv venv .venv_fomo"
        exit 1
    fi

    # Activate the virtual environment
    source $VENV_DIR/.venv_fomo/bin/activate
    echo "Activated .venv_fomo virtual environment"
else
    echo "Already in .venv_fomo virtual environment"
fi

# Run the preprocessing script
echo "Running preprocessing script..."
python $SRC_DIR/data/fomo-60k/preprocess.py --in_path=~/.cache/huggingface/hub/datasets--FOMO25--FOMO-MRI/snapshots/cd7f40948e99e6c562cacf1c5255305f923480c2/fomo-60k --out_path=~/Projects_kb/data/pretrain_preproc