#!/bin/bash
set -e

echo "Starting automated retraining pipeline..."

echo "[1/3] Running make_dataset.py..."
python src/data/make_dataset.py

echo "[2/3] Running build_features.py..."
python src/features/build_features.py

echo "[3/3] Running train_model.py..."
python src/models/train_model.py

echo "Automated retraining pipeline completed successfully."
