#!/usr/bin/env python3
"""
Interactive CLI chat interface for GBLM.

Usage:
    python scripts/chat_gblm.py --context-length 64 --max-new-tokens 32
"""

from pathlib import Path
import argparse
import sys
import signal
from typing import List, Optional
import numpy as np

from src.gblm_model.config import PathsConfig
from src.gblm_model.inference import (
    load_booster,
    load_feature_config,
    predict_next_token_proba,
    sample_from_proba_greedy,
    sample_from_proba_top_k,
    sample_from_proba_top_p,
)
from src.gblm_model.train import load_tokenizer
from src.gblm_data.tokenizer import Tokenizer
import lightgbm as lgb


class ChatGBLM:
    """
    Interactive chat interface for GBLM.
    Manages conversation state and rolling context window.
    """

    def __init__(
        self,
        booster: lgb.Booster,
        tokenizer: Tokenizer,
        context_length: int,
        max_new_tokens: int = 32,
        sampling: str = "greedy",
        top_k: int = 10,
        top_p: float = 0.9,
        temperature: float = 1.0,
        feature_cfg: Optional = None,
        embedding_matrix: Optional[np.ndarray] = None,
    ):
        """
        Initialize chat interface.

        Args:
            booster: Trained LightGBM model
            tokenizer: Tokenizer instance
            context_length: Model's context window size
            max_new_tokens: Maximum tokens to generate per turn
            sampling: Sampling method ("greedy", "top_k", "top_p")
            top_k: k value for top-k sampling
            top_p: p value for top-p sampling
            temperature: Temperature for sampling
            feature_cfg: Feature configuration
            embedding_matrix: Pre-loaded embedding matrix
        """
        self.booster = booster
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.sampling = sampling
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.feature_cfg = feature_cfg
        self.embedding_matrix = embedding_matrix

        # Initialize conversation with BOS token
        self.conversation_ids: List[int] = [tokenizer.bos_id]

    def get_context_ids(self) -> List[int]:
        """
        Get the last context_length tokens from conversation.
        This implements the rolling context window.

        Returns:
            List of token IDs for the current context
        """
        if len(self.conversation_ids) <= self.context_length:
            return self.conversation_ids.copy()
        else:
            # Take last context_length tokens (rolling window)
            return self.conversation_ids[-self.context_length:]

    def sample_next_token(self, proba: np.ndarray) -> int:
        """
        Sample next token based on the configured sampling method.

        Args:
            proba: Probability distribution over vocabulary

        Returns:
            Sampled token ID
        """
        if self.sampling == "greedy":
            return sample_from_proba_greedy(proba)
        elif self.sampling == "top_k":
            return sample_from_proba_top_k(
                proba, k=self.top_k, temperature=self.temperature
            )
        elif self.sampling == "top_p":
            return sample_from_proba_top_p(
                proba, p=self.top_p, temperature=self.temperature
            )
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")

    def generate_response(self, user_text: str) -> str:
        """
        Generate model response to user input.

        Args:
            user_text: User's input text

        Returns:
            Model's generated response
        """
        # Encode user input and append to conversation
        user_ids = self.tokenizer.encode(user_text, add_bos_eos=False)
        self.conversation_ids.extend(user_ids)

        # Generate model response
        generated_ids = []
        for _ in range(self.max_new_tokens):
            # Get context for prediction
            context_ids = self.get_context_ids()

            # Predict next token probabilities
            proba = predict_next_token_proba(
                booster=self.booster,
                tokenizer=self.tokenizer,
                token_ids=context_ids,
                context_length=self.context_length,
                feature_cfg=self.feature_cfg,
                embedding_matrix=self.embedding_matrix,
            )

            # Sample next token
            next_id = self.sample_next_token(proba)

            # Append to conversation and generated sequence
            self.conversation_ids.append(next_id)
            generated_ids.append(next_id)

            # Stop at EOS token
            if next_id == self.tokenizer.eos_id:
                break

        # Decode only the generated part (skip special tokens)
        model_text = self.tokenizer.decode(generated_ids, skip_special=True)
        return model_text

    def reset_conversation(self):
        """Reset conversation to initial state."""
        self.conversation_ids = [self.tokenizer.bos_id]

    def get_conversation_stats(self) -> dict:
        """
        Get statistics about the current conversation.

        Returns:
            Dictionary with conversation statistics
        """
        return {
            "total_tokens": len(self.conversation_ids),
            "context_window": self.context_length,
            "effective_context": min(len(self.conversation_ids), self.context_length),
            "truncated": len(self.conversation_ids) > self.context_length,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat interface for GBLM"
    )

    # Model and tokenizer paths
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
        "--context-length",
        type=int,
        default=16,
        help="Context window size (must match training context length)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate per turn",
    )

    # Sampling parameters
    parser.add_argument(
        "--sampling",
        type=str,
        default="top_k",
        choices=["greedy", "top_k", "top_p"],
        help="Sampling method",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="k value for top-k sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="p value for top-p sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )

    # Other options
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show conversation statistics after each turn",
    )

    return parser.parse_args()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nGoodbye!")
    sys.exit(0)


def main():
    args = parse_args()

    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

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
    feature_config_path = artifacts_dir / "feature_config.json"
    embedding_matrix_path = artifacts_dir / "embedding_matrix.npy"

    # Load model and tokenizer
    print("=" * 60)
    print("GBLM Interactive Chat")
    print("=" * 60)
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"Vocabulary size: {len(tokenizer.itos)}")

    print(f"Loading model from {model_path}...")
    booster = load_booster(model_path)
    print("Model loaded successfully")

    # Load feature configuration if available
    feature_cfg = None
    embedding_matrix = None

    if feature_config_path.exists():
        print(f"Loading feature configuration...")
        feature_cfg = load_feature_config(feature_config_path)

        # Load embedding matrix if needed
        if feature_cfg and feature_cfg.use_embeddings and embedding_matrix_path.exists():
            print(f"Loading embedding matrix...")
            embedding_matrix = np.load(embedding_matrix_path)
            print(f"Embedding matrix shape: {embedding_matrix.shape}")

    # Display chat settings
    print("\n" + "=" * 60)
    print("Chat Settings:")
    print(f"  Context length: {args.context_length}")
    print(f"  Max tokens per turn: {args.max_new_tokens}")
    print(f"  Sampling: {args.sampling}")
    if args.sampling == "top_k":
        print(f"  Top-k: {args.top_k}")
    elif args.sampling == "top_p":
        print(f"  Top-p: {args.top_p}")
    if args.sampling in ["top_k", "top_p"]:
        print(f"  Temperature: {args.temperature}")
    print("=" * 60)

    # Initialize chat interface
    chat = ChatGBLM(
        booster=booster,
        tokenizer=tokenizer,
        context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        sampling=args.sampling,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        feature_cfg=feature_cfg,
        embedding_matrix=embedding_matrix,
    )

    # Print instructions
    print("\nType your message (or 'exit'/'quit'/:q to quit, 'reset' to clear context)")
    print("=" * 60 + "\n")

    # Main chat loop
    while True:
        try:
            # Get user input
            user_text = input("You> ").strip()

            # Check for exit commands
            if user_text.lower() in ["exit", "quit", ":q"]:
                print("\nGoodbye!")
                break

            # Check for reset command
            if user_text.lower() == "reset":
                chat.reset_conversation()
                print("Conversation reset.\n")
                continue

            # Skip empty input
            if not user_text:
                continue

            # Generate and display response
            print("Model> ", end="", flush=True)
            response = chat.generate_response(user_text)
            print(response)

            # Show statistics if requested
            if args.show_stats:
                stats = chat.get_conversation_stats()
                print(f"\n[Stats: {stats['total_tokens']} tokens, "
                      f"context: {stats['effective_context']}/{stats['context_window']}"
                      f"{', TRUNCATED' if stats['truncated'] else ''}]")

            print()  # Empty line for readability

        except EOFError:
            # Handle Ctrl+D
            print("\n\nGoodbye!")
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C (backup, should be caught by signal handler)
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Continuing chat session...\n")


if __name__ == "__main__":
    main()