#!/usr/bin/env python3
"""
Quick test script to train a minimal GBLM model for testing purposes.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import lightgbm as lgb
from pathlib import Path
import json

# Load tokenizer to get vocab size
artifacts_dir = Path("artifacts")
with open(artifacts_dir / "tokenizer.json", "r") as f:
    tokenizer_data = json.load(f)
    num_classes = len(tokenizer_data["itos"])

print(f"Number of classes: {num_classes}")

# Load data
npz = np.load(artifacts_dir / "gblm_data.npz")
X = npz["X"][:1000]  # Use only first 1000 samples for testing
y = npz["y"][:1000]

print(f"Using {X.shape[0]} samples for quick test")

# Create dataset
dtrain = lgb.Dataset(
    X,
    label=y,
    categorical_feature=list(range(X.shape[1])),
)

# Train with minimal parameters
params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "learning_rate": 0.3,
    "num_leaves": 4,  # Very small
    "min_data_in_leaf": 50,
    "verbose": -1,
    "metric": "multi_logloss",
    "num_threads": 1,  # Single thread to avoid complexity
}

print("Training with minimal parameters...")
booster = lgb.train(
    params=params,
    train_set=dtrain,
    num_boost_round=2,  # Only 2 rounds for testing
)

# Save model
model_path = artifacts_dir / "gblm_model.txt"
booster.save_model(str(model_path))
print(f"Model saved to {model_path}")

# Save training metrics for compatibility
metrics = {
    "train_accuracy": 0.1,  # Placeholder values
    "valid_accuracy": 0.1,
    "train_logloss": 8.0,
    "valid_logloss": 8.0,
    "train_perplexity": 3000.0,
    "valid_perplexity": 3000.0,
    "best_iteration": 2,
    "num_classes": num_classes,
    "context_length": X.shape[1],
    "note": "This is a minimal test model for pipeline testing only"
}

metrics_path = artifacts_dir / "gblm_train_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {metrics_path}")

print("\nQuick test training complete!")
print("Note: This is a minimal model for testing the pipeline only.")
print("For real training, use scripts/train_gblm.py with appropriate parameters.")