#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting model training..."

python3 src/training.py --config configs/config.yaml

echo "Model training completed."