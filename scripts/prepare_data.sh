#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting data preparation..."

python3 src/data_preparation.py --config configs/config.yaml

echo "Data preparation completed."