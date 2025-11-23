#!/usr/bin/env python3
"""
Analyze vocabulary coverage and build vocabulary based on coverage percentage.
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Tuple, List
import sys
sys.path.insert(0, '.')

from gblm_data.corpus import load_corpus_texts
from gblm_data.vocab import build_vocab, count_vocab


def analyze_coverage(frequencies: dict, coverage_percent: float = 95.0) -> Tuple[int, int, List[Tuple[str, int]]]:
    """
    Analyze vocabulary coverage and find cutoff for given coverage percentage.

    Returns:
        vocab_size: Size of vocabulary for given coverage
        min_freq: Minimum frequency for the coverage
        selected_tokens: List of (token, freq) tuples
    """
    # Sort by frequency
    sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

    # Calculate total tokens
    total_count = sum(f for _, f in sorted_freq)
    target_coverage = total_count * (coverage_percent / 100.0)

    cumulative = 0
    selected_tokens = []
    min_freq = 1

    for token, freq in sorted_freq:
        cumulative += freq
        selected_tokens.append((token, freq))
        min_freq = freq

        if cumulative >= target_coverage:
            break

    actual_coverage = (cumulative / total_count) * 100

    print(f"\n=== Coverage Analysis ===")
    print(f"Target coverage: {coverage_percent:.1f}%")
    print(f"Actual coverage: {actual_coverage:.2f}%")
    print(f"Vocabulary size: {len(selected_tokens)}")
    print(f"Minimum frequency: {min_freq}")
    print(f"Total unique tokens: {len(frequencies)}")
    print(f"Total token count: {total_count:,}")

    # Show frequency distribution
    freq_dist = Counter()
    for _, f in sorted_freq:
        freq_dist[f] += 1

    print(f"\n=== Frequency Distribution ===")
    common_freqs = sorted(freq_dist.items(), key=lambda x: x[0], reverse=True)[:20]
    for freq, count in common_freqs:
        coverage_by_freq = (freq * count / total_count) * 100
        print(f"Freq {freq:6d}: {count:6d} tokens, coverage {coverage_by_freq:6.2f}%")

    return len(selected_tokens), min_freq, selected_tokens


def build_coverage_vocab(corpus_path: Path, coverage_percent: float = 95.0,
                         lowercase: bool = True, max_docs: int = None):
    """
    Build vocabulary based on coverage percentage.
    """
    print(f"Loading corpus from {corpus_path}...")
    docs = load_corpus_texts(
        file_path=corpus_path,
        is_csv=False,
        text_column=None,
        max_docs=max_docs
    )

    print(f"Building frequency counts from {len(docs)} documents...")
    # Count all tokens first
    counter = count_vocab(docs, lowercase=lowercase, verbose=False)

    # Build full vocabulary first
    vocab_tokens = build_vocab(
        counter=counter,
        min_freq=1,  # Get all tokens
        top_k=None,  # No limit
        verbose=False
    )

    # Create frequency dict
    frequencies = dict(counter)

    # Analyze coverage
    vocab_size, min_freq, selected_tokens = analyze_coverage(frequencies, coverage_percent)

    # Build new vocabulary with calculated min_freq
    # Round up min_freq to ensure we don't exceed coverage
    print(f"\n=== Building final vocabulary ===")
    print(f"Using minimum frequency: {min_freq}")

    final_vocab_tokens = build_vocab(
        counter=counter,
        min_freq=min_freq,
        top_k=None,
        verbose=False
    )

    # Create final frequencies dict
    final_frequencies = {token: counter[token] for token in final_vocab_tokens}

    # Double check coverage
    total_selected = sum(final_frequencies.values())
    total_all = sum(frequencies.values())
    final_coverage = (total_selected / total_all) * 100

    print(f"\n=== Final Vocabulary Stats ===")
    print(f"Vocabulary size: {len(final_vocab_tokens)}")
    print(f"Coverage: {final_coverage:.2f}%")
    print(f"Min frequency used: {min_freq}")

    return final_vocab_tokens, final_frequencies, min_freq


def main():
    parser = argparse.ArgumentParser(description="Analyze vocabulary coverage")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("cleaned_merged_fairy_tales_without_eos.txt"),
        help="Path to corpus file",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=95.0,
        help="Target coverage percentage (default: 95.0)",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        default=True,
        help="Convert to lowercase",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        help="Maximum number of documents to process",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the coverage-based vocabulary",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for vocabulary files",
    )

    args = parser.parse_args()

    # Build coverage-based vocabulary
    vocab_tokens, frequencies, min_freq = build_coverage_vocab(
        corpus_path=args.corpus,
        coverage_percent=args.coverage,
        lowercase=args.lowercase,
        max_docs=args.max_docs
    )

    if args.save:
        # Save vocabulary
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save vocabulary in the same format as the original vocab.json
        vocab_data = {
            "tokens": vocab_tokens,
            "size": len(vocab_tokens),
            "special_tokens": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            "frequencies": frequencies
        }

        vocab_path = output_dir / f"vocab_coverage{int(args.coverage)}.json"
        with open(vocab_path, "w") as f:
            json.dump(vocab_data, f, indent=2)
        print(f"\nVocabulary saved to {vocab_path}")

        # Save stats
        stats = {
            "coverage_percent": args.coverage,
            "vocabulary_size": len(vocab_tokens),
            "min_frequency": min_freq,
            "special_tokens": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        }
        stats_path = output_dir / f"vocab_coverage{int(args.coverage)}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()