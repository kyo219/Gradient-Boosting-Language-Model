#!/usr/bin/env python3
"""Build vocabulary and tokenizer from corpus."""

import sys
from pathlib import Path
import argparse
import json

# Add parent directory to path to import gblm_data module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gblm_data.config import GBLMConfig, create_default_config
from src.gblm_data.corpus import load_corpus_texts
from src.gblm_data.vocab import count_vocab, build_vocab, save_vocab, analyze_oov_rate
from src.gblm_data.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Build vocabulary and tokenizer from corpus"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to corpus file (text or CSV)"
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
        help="Output directory for vocabulary and tokenizer"
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=5,
        help="Minimum word frequency for vocabulary"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5000,
        help="Maximum vocabulary size (top-k most frequent words)"
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        default=True,
        help="Convert text to lowercase"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process"
    )
    parser.add_argument(
        "--is-csv",
        action="store_true",
        help="Indicate that the corpus is a CSV file"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name for text in CSV file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    args = parser.parse_args()

    # Create or load configuration
    if args.config:
        print(f"Loading config from {args.config}")
        cfg = GBLMConfig.from_json(args.config)
    else:
        print("Creating default configuration")
        cfg = create_default_config(args.corpus)

        # Override with command-line arguments
        cfg.vocab.min_freq = args.min_freq
        cfg.vocab.top_k = args.top_k
        cfg.vocab.lowercase = args.lowercase
        cfg.vocab.max_docs = args.max_docs
        cfg.paths.artifacts_dir = Path(args.output_dir)
        cfg.paths.is_csv = args.is_csv
        cfg.paths.text_column = args.text_column if args.is_csv else None

    print("\n=== Configuration ===")
    print(f"Corpus: {cfg.paths.corpus_file}")
    print(f"Output directory: {cfg.paths.artifacts_dir}")
    print(f"Min frequency: {cfg.vocab.min_freq}")
    print(f"Top-K vocab size: {cfg.vocab.top_k}")
    print(f"Lowercase: {cfg.vocab.lowercase}")
    print(f"Max documents: {cfg.vocab.max_docs}")

    # Step 1: Load corpus texts
    print("\n=== Loading Corpus ===")
    texts = load_corpus_texts(
        file_path=cfg.paths.corpus_file,
        text_column=cfg.paths.text_column,
        max_docs=cfg.vocab.max_docs,
        is_csv=cfg.paths.is_csv
    )

    # Step 2: Count vocabulary
    print("\n=== Counting Vocabulary ===")
    counter = count_vocab(
        texts=texts,
        lowercase=cfg.vocab.lowercase,
        verbose=args.verbose
    )

    # Step 3: Build vocabulary
    print("\n=== Building Vocabulary ===")
    vocab_tokens = build_vocab(
        counter=counter,
        min_freq=cfg.vocab.min_freq,
        top_k=cfg.vocab.top_k,
        verbose=True
    )

    # Step 4: Create tokenizer
    print("\n=== Creating Tokenizer ===")
    tokenizer = Tokenizer.from_vocab(
        vocab_tokens=vocab_tokens,
        lowercase=cfg.vocab.lowercase
    )
    print(f"Tokenizer created: {tokenizer}")

    # Step 5: Analyze OOV rate on a sample
    print("\n=== Analyzing OOV Rate ===")
    oov_stats = analyze_oov_rate(
        texts=texts[:min(100, len(texts))],  # Sample first 100 texts
        vocab_tokens=vocab_tokens,
        lowercase=cfg.vocab.lowercase
    )
    print(f"OOV rate on sample: {oov_stats['oov_rate']:.2f}%")
    print(f"OOV tokens: {oov_stats['oov_tokens']}/{oov_stats['total_tokens']}")
    if oov_stats['sample_oov_words']:
        print(f"Sample OOV words: {oov_stats['sample_oov_words'][:10]}")

    # Step 6: Save artifacts
    print("\n=== Saving Artifacts ===")
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save vocabulary
    vocab_path = cfg.paths.artifacts_dir / "vocab.json"
    save_vocab(vocab_tokens, vocab_path, save_frequencies=counter)

    # Save tokenizer
    tokenizer_path = cfg.paths.artifacts_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    # Save configuration
    config_path = cfg.paths.artifacts_dir / "config.json"
    cfg.save_json(config_path)
    print(f"Configuration saved to {config_path}")

    # Save statistics
    stats = {
        "corpus_file": str(cfg.paths.corpus_file),
        "n_documents": len(texts),
        "vocab_size": len(vocab_tokens),
        "special_tokens": tokenizer.itos[:4],
        "total_tokens": sum(counter.values()),
        "unique_tokens_before_filtering": len(counter),
        "min_freq": cfg.vocab.min_freq,
        "top_k": cfg.vocab.top_k,
        "oov_stats": oov_stats
    }

    stats_path = cfg.paths.artifacts_dir / "build_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Build statistics saved to {stats_path}")

    print("\n=== Build Complete ===")
    print(f"All artifacts saved to {cfg.paths.artifacts_dir}")
    print(f"Vocabulary size: {tokenizer.vocab_size} tokens")


if __name__ == "__main__":
    main()