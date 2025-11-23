#!/usr/bin/env python3
"""Train GBLM with feature engineering.

Example script showing how to use the new feature engineering capabilities.
"""

import argparse
from pathlib import Path

from src.gblm_model.config import GBLMTrainConfig
from src.gblm_model.features import FeatureConfig
from src.gblm_model.train import train_gblm


def main():
    parser = argparse.ArgumentParser(
        description="Train GBLM with feature engineering"
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Use pre-trained embeddings for feature engineering"
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path("artifacts/embedding_matrix.npy"),
        help="Path to embedding matrix"
    )
    parser.add_argument(
        "--add-stats",
        action="store_true",
        default=True,
        help="Add context statistics features"
    )
    parser.add_argument(
        "--add-mean-pooled",
        action="store_true",
        help="Add mean-pooled embedding features"
    )
    parser.add_argument(
        "--add-prototypes",
        action="store_true",
        help="Add prototype similarity features"
    )
    parser.add_argument(
        "--n-prototypes",
        type=int,
        default=50,
        help="Number of prototype clusters"
    )
    parser.add_argument(
        "--embedding-dim-reduction",
        type=int,
        help="Reduce embedding dimensions to this value"
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=300,
        help="Number of boosting rounds"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for LightGBM"
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=128,
        help="Number of leaves in LightGBM"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print training progress"
    )

    args = parser.parse_args()

    # Configure training
    train_cfg = GBLMTrainConfig()
    train_cfg.lgbm.num_boost_round = args.num_boost_round
    train_cfg.lgbm.learning_rate = args.learning_rate
    train_cfg.lgbm.num_leaves = args.num_leaves

    # Configure features
    feature_cfg = FeatureConfig(
        use_token_ids=True,  # Always use token IDs as base features
        add_context_length=args.add_stats,
        add_unk_count=args.add_stats,
        add_type_token_ratio=args.add_stats,
        use_embeddings=args.use_embeddings,
        embedding_path=args.embedding_path if args.use_embeddings else None,
        add_mean_pooled_embedding=args.add_mean_pooled,
        embedding_dim_reduction=args.embedding_dim_reduction,
        add_prototype_similarities=args.add_prototypes,
        n_prototypes=args.n_prototypes,
    )

    # Print configuration
    if args.verbose:
        print("=" * 60)
        print("GBLM Training with Feature Engineering")
        print("=" * 60)
        print("\nTraining Configuration:")
        print(f"  Num boost rounds: {train_cfg.lgbm.num_boost_round}")
        print(f"  Learning rate: {train_cfg.lgbm.learning_rate}")
        print(f"  Num leaves: {train_cfg.lgbm.num_leaves}")

        print("\nFeature Configuration:")
        print(f"  Use token IDs: {feature_cfg.use_token_ids}")
        print(f"  Add context statistics: {args.add_stats}")
        if args.use_embeddings:
            print(f"  Use embeddings: {feature_cfg.use_embeddings}")
            print(f"  Embedding path: {feature_cfg.embedding_path}")
            print(f"  Add mean-pooled: {feature_cfg.add_mean_pooled_embedding}")
            print(f"  Add prototypes: {feature_cfg.add_prototype_similarities}")
            if feature_cfg.add_prototype_similarities:
                print(f"  Num prototypes: {feature_cfg.n_prototypes}")
            if feature_cfg.embedding_dim_reduction:
                print(f"  Embedding dim reduction: {feature_cfg.embedding_dim_reduction}")
        print("=" * 60)
        print()

    # Train model
    try:
        booster, metrics = train_gblm(
            cfg=train_cfg,
            feature_cfg=feature_cfg if (args.use_embeddings or args.add_stats) else None,
            verbose=args.verbose
        )

        # Print results
        if args.verbose:
            print("\n" + "=" * 60)
            print("Training Complete!")
            print("=" * 60)
            print("\nFinal Metrics:")
            print(f"  Train accuracy: {metrics['train_accuracy']:.4f}")
            print(f"  Valid accuracy: {metrics['valid_accuracy']:.4f}")
            print(f"  Train perplexity: {metrics['train_perplexity']:.2f}")
            print(f"  Valid perplexity: {metrics['valid_perplexity']:.2f}")
            print(f"  Best iteration: {metrics['best_iteration']}")
            print(f"  Total features: {metrics['num_features']}")
            print(f"  Categorical features: {metrics['num_categorical_features']}")
            print("=" * 60)

    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()