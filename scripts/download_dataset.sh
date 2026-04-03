#!/usr/bin/env bash
set -euo pipefail

# Replace with your dataset slug (from Kaggle URL):
DATASET="owner/dataset-name"

mkdir -p data/raw
kaggle datasets download -d "$DATASET" -p data/raw --unzip
