#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting model evaluation..."

python3 src/evaluation.py --config configs/config.yaml

echo "Model evaluation completed."
