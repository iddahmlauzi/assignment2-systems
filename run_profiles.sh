#!/bin/bash

# Configuration for small model
SMALL_CONTEXTS=(512 2048 4096)

# Configuration for XL model
XL_PARAMS="--d-model 2560 --d-ff 10240 --num-layers 32 --num-heads 32"
XL_CONTEXTS=(256 512 1024)

# Modes to iterate through
MODES=("f")

# Exit immediately if a command fails
set -e

for mode in "${MODES[@]}"; do
    echo "========================================"
    echo "Starting runs for Mode: $mode"
    echo "========================================"

    # --- SMALL MODEL RUNS ---
    for ctx in "${SMALL_CONTEXTS[@]}"; do
        echo "--> Running Small Model | Mode: $mode | Context: $ctx"
        
        # Ensure the environment is synced
        uv pip install -e ./cs336-basics
        
        # Execute Modal run
        uv run modal run scripts/benchmark.py --profile --mode "$mode" --context-length "$ctx"
    done

    # --- XL MODEL RUNS ---
    for ctx in "${XL_CONTEXTS[@]}"; do
        echo "--> Running XL Model | Mode: $mode | Context: $ctx"
        
        # Ensure the environment is synced
        uv pip install -e ./cs336-basics
        
        # Execute Modal run with extra params
        uv run modal run scripts/benchmark.py --profile --mode "$mode" $XL_PARAMS --context-length "$ctx"
    done
done

echo "Done! All profiles have been launched."