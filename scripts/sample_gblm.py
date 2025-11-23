#!/usr/bin/env python3
"""
Script to generate text using trained GBLM model.

Usage:
    python scripts/sample_gblm.py --prompt "Once upon a time"
"""

from pathlib import Path
import argparse
import json

from src.gblm_model.config import PathsConfig
from src.gblm_model.inference import (
    load_booster,
    generate_text,
    batch_generate,
)
from src.gblm_model.train import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using trained GBLM model"
    )

    # Paths
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing model and tokenizer",
    )
    parser.add_argument(
        "--tokenizer-json",
        type=str,
        default="tokenizer.json",
        help="Tokenizer JSON filename",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="gblm_model.txt",
        help="Model filename",
    )

    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Starting text for generation",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="File containing multiple prompts (one per line)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=16,
        help="Context window size",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate",
    )

    # Sampling parameters
    parser.add_argument(
        "--sampling",
        type=str,
        default="greedy",
        choices=["greedy", "top_k", "top_p", "temperature"],
        help="Sampling method",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top tokens for top-k sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Cumulative probability for top-p sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax (lower = more deterministic)",
    )
    parser.add_argument(
        "--no-stop-at-eos",
        action="store_true",
        help="Continue generating past EOS token",
    )

    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print generation progress",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Save generated text to file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate per prompt",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup paths
    paths = PathsConfig(
        artifacts_dir=args.artifacts_dir,
        tokenizer_json=args.tokenizer_json,
        data_npz="gblm_data.npz",  # Not used for inference
        model_file=args.model_file,
    )

    artifacts_dir = paths.artifacts_dir
    tokenizer_path = artifacts_dir / paths.tokenizer_json
    model_path = artifacts_dir / paths.model_file

    # Load model and tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"Vocabulary size: {len(tokenizer.itos)}")

    print(f"Loading model from {model_path}...")
    booster = load_booster(model_path)
    print("Model loaded successfully")

    # Prepare prompts
    prompts = []
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = [args.prompt]

    # Generation parameters
    gen_kwargs = {
        "context_length": args.context_length,
        "max_new_tokens": args.max_new_tokens,
        "sampling": args.sampling,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "stop_at_eos": not args.no_stop_at_eos,
        "verbose": args.verbose,
    }

    print(f"\n=== Generation Settings ===")
    print(f"Context length: {args.context_length}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Sampling: {args.sampling}")
    if args.sampling == "top_k":
        print(f"Top-k: {args.top_k}")
    elif args.sampling == "top_p":
        print(f"Top-p: {args.top_p}")
    if args.sampling in ["top_k", "top_p", "temperature"]:
        print(f"Temperature: {args.temperature}")
    print(f"Stop at EOS: {not args.no_stop_at_eos}")

    # Generate text
    all_results = []
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n=== Prompt {prompt_idx + 1}/{len(prompts)} ===")
        print(f"Prompt: {prompt}")
        print("-" * 50)

        for sample_idx in range(args.num_samples):
            if args.num_samples > 1:
                print(f"\nSample {sample_idx + 1}/{args.num_samples}:")

            text = generate_text(
                booster=booster,
                tokenizer=tokenizer,
                prompt=prompt,
                **gen_kwargs,
            )

            print("\nGenerated text:")
            print(text)
            print()

            all_results.append({
                "prompt": prompt,
                "sample_idx": sample_idx,
                "generated": text,
            })

    # Save results if requested
    if args.output_file:
        output_data = {
            "settings": {
                "context_length": args.context_length,
                "max_new_tokens": args.max_new_tokens,
                "sampling": args.sampling,
                "top_k": args.top_k if args.sampling == "top_k" else None,
                "top_p": args.top_p if args.sampling == "top_p" else None,
                "temperature": args.temperature,
                "stop_at_eos": not args.no_stop_at_eos,
            },
            "results": all_results,
        }

        if args.output_file.suffix == ".json":
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        else:
            # Plain text output
            with open(args.output_file, "w", encoding="utf-8") as f:
                for result in all_results:
                    f.write(f"Prompt: {result['prompt']}\n")
                    f.write(f"Generated: {result['generated']}\n")
                    f.write("-" * 80 + "\n")

        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()