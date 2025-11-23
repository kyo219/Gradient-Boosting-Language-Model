#!/usr/bin/env python3
"""
Script to train GBLM using LightGBM.

Usage:
    python scripts/train_gblm.py
"""

from pathlib import Path
import json
import argparse

from gblm_model.config import (
    GBLMTrainConfig,
    PathsConfig,
    TrainSplitConfig,
    LightGBMConfig,
)
from gblm_model.train import train_gblm


def parse_args():
    parser = argparse.ArgumentParser(description="Train GBLM with LightGBM")

    # Paths
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing artifacts (tokenizer, data)",
    )
    parser.add_argument(
        "--tokenizer-json",
        type=str,
        default="tokenizer.json",
        help="Tokenizer JSON filename",
    )
    parser.add_argument(
        "--data-npz",
        type=str,
        default="gblm_data.npz",
        help="Training data NPZ filename",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="gblm_model.txt",
        help="Output model filename",
    )

    # Train/validation split
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.1,
        help="Validation set size ratio (0.0-1.0)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle data before splitting",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for data splitting",
    )

    # LightGBM parameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for gradient boosting",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=64,
        help="Number of leaves in each tree",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=-1,
        help="Maximum tree depth (-1 for no limit)",
    )
    parser.add_argument(
        "--min-data-in-leaf",
        type=int,
        default=20,
        help="Minimum data points in leaf",
    )
    parser.add_argument(
        "--feature-fraction",
        type=float,
        default=1.0,
        help="Feature fraction for bagging",
    )
    parser.add_argument(
        "--bagging-fraction",
        type=float,
        default=1.0,
        help="Data fraction for bagging",
    )
    parser.add_argument(
        "--bagging-freq",
        type=int,
        default=0,
        help="Bagging frequency (0 = disable)",
    )
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=0.0,
        help="L1 regularization",
    )
    parser.add_argument(
        "--lambda-l2",
        type=float,
        default=0.0,
        help="L2 regularization",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=500,
        help="Number of boosting rounds",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=20,
        help="Early stopping rounds (0 to disable)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel threads (-1 for all)",
    )

    # Other options
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        help="Load configuration from JSON file (overrides other arguments)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Build configuration
    if args.config_json and args.config_json.exists():
        # Load from JSON file
        print(f"Loading configuration from {args.config_json}")
        with open(args.config_json, "r") as f:
            config_dict = json.load(f)
        cfg = GBLMTrainConfig.from_dict(config_dict)
    else:
        # Build from command line arguments
        cfg = GBLMTrainConfig(
            paths=PathsConfig(
                artifacts_dir=args.artifacts_dir,
                tokenizer_json=args.tokenizer_json,
                data_npz=args.data_npz,
                model_file=args.model_file,
            ),
            split=TrainSplitConfig(
                valid_size=args.valid_size,
                shuffle=not args.no_shuffle,
                random_seed=args.random_seed,
            ),
            lgbm=LightGBMConfig(
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                max_depth=args.max_depth,
                min_data_in_leaf=args.min_data_in_leaf,
                feature_fraction=args.feature_fraction,
                bagging_fraction=args.bagging_fraction,
                bagging_freq=args.bagging_freq,
                lambda_l1=args.lambda_l1,
                lambda_l2=args.lambda_l2,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping_rounds,
                n_jobs=args.n_jobs,
                verbose=-1 if args.quiet else 1,
            ),
        )

    # Train model
    print("=== Starting GBLM Training ===")
    print(f"Configuration:")
    print(json.dumps(cfg.to_dict(), indent=2))
    print()

    booster, metrics = train_gblm(cfg, verbose=not args.quiet)

    # Display results
    print("\n=== Training Complete ===")
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Save configuration for reproducibility
    config_path = cfg.paths.artifacts_dir / "gblm_train_config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"\nConfiguration saved to {config_path}")

    print(f"\nModel saved to {cfg.paths.artifacts_dir / cfg.paths.model_file}")
    print(f"Metrics saved to {cfg.paths.artifacts_dir / 'gblm_train_metrics.json'}")


if __name__ == "__main__":
    main()