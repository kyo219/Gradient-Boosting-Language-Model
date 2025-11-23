#!/usr/bin/env python3
"""Generate GBLM training dataset from corpus using tokenizer."""

import sys
from pathlib import Path
import argparse
import json
import time

# Add parent directory to path to import gblm_data module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gblm_data.config import GBLMConfig
from src.gblm_data.corpus import load_corpus_texts
from src.gblm_data.tokenizer import Tokenizer
from src.gblm_data.dataset import (
    make_gblm_training_data,
    save_dataset,
    create_train_val_split,
    analyze_dataset
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate GBLM training dataset from corpus"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to corpus file (text or CSV)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer JSON file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=16,
        help="Context window size (L)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to generate"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to use from corpus"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle the generated samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio for train/val split"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Create train/validation split"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    args = parser.parse_args()

    # Load configuration if provided
    if args.config:
        print(f"Loading config from {args.config}")
        cfg = GBLMConfig.from_json(args.config)
    else:
        print("Using command-line arguments for configuration")
        from src.gblm_data.config import create_default_config
        cfg = create_default_config(args.corpus)

        # Override with command-line arguments
        cfg.dataset.context_length = args.context_length
        cfg.dataset.max_samples = args.max_samples
        cfg.dataset.shuffle = args.shuffle
        cfg.dataset.random_seed = args.seed
        cfg.vocab.max_docs = args.max_docs
        cfg.paths.artifacts_dir = Path(args.output_dir)

    print("\n=== Configuration ===")
    print(f"Corpus: {cfg.paths.corpus_file}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Output directory: {cfg.paths.artifacts_dir}")
    print(f"Context length: {cfg.dataset.context_length}")
    print(f"Max samples: {cfg.dataset.max_samples}")
    print(f"Max documents: {cfg.vocab.max_docs}")
    print(f"Shuffle: {cfg.dataset.shuffle}")
    print(f"Random seed: {cfg.dataset.random_seed}")

    # Step 1: Load tokenizer
    print("\n=== Loading Tokenizer ===")
    tokenizer = Tokenizer.load(args.tokenizer)
    print(f"Loaded tokenizer: {tokenizer}")

    # Step 2: Load corpus texts
    print("\n=== Loading Corpus ===")
    texts = load_corpus_texts(
        file_path=cfg.paths.corpus_file,
        text_column=cfg.paths.text_column,
        max_docs=cfg.vocab.max_docs,
        is_csv=cfg.paths.is_csv
    )

    # Step 3: Generate GBLM training data
    print("\n=== Generating GBLM Dataset ===")
    start_time = time.time()

    X, y = make_gblm_training_data(
        texts=texts,
        tokenizer=tokenizer,
        context_length=cfg.dataset.context_length,
        max_samples=cfg.dataset.max_samples,
        shuffle=cfg.dataset.shuffle,
        random_seed=cfg.dataset.random_seed,
        verbose=args.verbose
    )

    generation_time = time.time() - start_time
    print(f"Dataset generation took {generation_time:.2f} seconds")

    # Step 4: Analyze dataset
    print("\n=== Dataset Analysis ===")
    stats = analyze_dataset(X, y, tokenizer)
    print(f"Number of samples: {stats['n_samples']:,}")
    print(f"Context length: {stats['context_length']}")
    print(f"Vocabulary size: {stats['vocab_size']}")
    print(f"Unique contexts: {stats['n_unique_contexts']:,}")
    print(f"Unique targets: {stats['n_unique_targets']}")
    print(f"Padding ratio: {stats['padding_ratio']:.2%}")
    print(f"Unknown token ratio (X): {stats['unk_ratio_X']:.2%}")
    print(f"Unknown token ratio (y): {stats['unk_ratio_y']:.2%}")

    print("\nTop 10 most frequent target tokens:")
    for i, target_info in enumerate(stats['top_targets'], 1):
        print(f"  {i:2}. {target_info['token']:15} "
              f"({target_info['count']:7,} occurrences, "
              f"{target_info['frequency']:.2%})")

    # Step 5: Create train/validation split if requested
    if args.split:
        print(f"\n=== Creating Train/Validation Split ===")
        print(f"Validation ratio: {args.val_ratio:.1%}")

        X_train, y_train, X_val, y_val = create_train_val_split(
            X, y,
            val_ratio=args.val_ratio,
            random_seed=cfg.dataset.random_seed
        )

        # Save train and validation sets separately
        save_dataset(X_train, y_train, cfg.paths.artifacts_dir, prefix="train")
        save_dataset(X_val, y_val, cfg.paths.artifacts_dir, prefix="val")

    # Step 6: Save full dataset
    print("\n=== Saving Dataset ===")
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(X, y, cfg.paths.artifacts_dir, prefix="gblm_data")

    # Save configuration used for dataset generation
    dataset_config_path = cfg.paths.artifacts_dir / "dataset_config.json"
    cfg.save_json(dataset_config_path)
    print(f"Dataset configuration saved to {dataset_config_path}")

    # Save extended statistics
    extended_stats = {
        **stats,
        "corpus_file": str(cfg.paths.corpus_file),
        "tokenizer_file": str(args.tokenizer),
        "n_documents": len(texts),
        "generation_time_seconds": generation_time,
        "config": cfg.to_dict()
    }

    stats_path = cfg.paths.artifacts_dir / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(extended_stats, f, indent=2)
    print(f"Dataset statistics saved to {stats_path}")

    print("\n=== Dataset Generation Complete ===")
    print(f"All files saved to {cfg.paths.artifacts_dir}")
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Ready for LightGBM training!")

    # Print example usage for LightGBM
    print("\n=== Example LightGBM Usage ===")
    print("```python")
    print("import numpy as np")
    print("import lightgbm as lgb")
    print("from src.gblm_data.dataset import load_dataset, to_lgb_dataset")
    print()
    print(f"# Load dataset")
    print(f"X, y = load_dataset('{cfg.paths.artifacts_dir}')")
    print()
    print("# Create LightGBM dataset")
    print("dtrain = to_lgb_dataset(X, y)")
    print()
    print("# Train model")
    print("params = {")
    print("    'objective': 'multiclass',")
    print(f"    'num_class': {tokenizer.vocab_size},")
    print("    'metric': 'multi_logloss',")
    print("    'boosting_type': 'gbdt',")
    print("    'num_leaves': 31,")
    print("    'learning_rate': 0.05,")
    print("    'feature_fraction': 0.9")
    print("}")
    print()
    print("model = lgb.train(params, dtrain, num_boost_round=100)")
    print("```")


if __name__ == "__main__":
    main()