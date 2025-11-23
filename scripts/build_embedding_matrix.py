#!/usr/bin/env python3
"""Build embedding matrix aligned with GBLM tokenizer vocabulary.

This script loads pre-trained embeddings and aligns them with the GBLM tokenizer's
vocabulary to create an embedding matrix that can be used for feature engineering.
"""

import argparse
import json
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

from src.gblm_data.tokenizer import Tokenizer


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def download_glove_embeddings(
    output_dir: Path,
    dimension: int = 50
) -> Path:
    """Download GloVe embeddings if not already present.

    Args:
        output_dir: Directory to save embeddings
        dimension: Embedding dimension (50, 100, 200, or 300)

    Returns:
        Path to the extracted embedding file
    """
    valid_dims = {50, 100, 200, 300}
    if dimension not in valid_dims:
        raise ValueError(f"Dimension must be one of {valid_dims}")

    glove_dir = output_dir / "glove"
    glove_dir.mkdir(exist_ok=True, parents=True)

    glove_file = glove_dir / f"glove.6B.{dimension}d.txt"

    if glove_file.exists():
        logger.info(f"GloVe embeddings already downloaded at {glove_file}")
        return glove_file

    # Download GloVe embeddings
    logger.info(f"Downloading GloVe embeddings (dimension={dimension})...")
    zip_url = "http://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = glove_dir / "glove.6B.zip"

    if not zip_path.exists():
        logger.info("Downloading GloVe zip file (862MB)...")
        urllib.request.urlretrieve(zip_url, zip_path)
        logger.info("Download complete")

    # Extract specific file
    logger.info(f"Extracting glove.6B.{dimension}d.txt...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(f"glove.6B.{dimension}d.txt", glove_dir)

    logger.info(f"GloVe embeddings saved to {glove_file}")
    return glove_file


def load_pretrained_embeddings(
    embedding_path: Path,
    max_vocab: Optional[int] = None
) -> dict[str, np.ndarray]:
    """Load pre-trained embeddings from text file.

    Args:
        embedding_path: Path to embedding file
        max_vocab: Maximum vocabulary size to load (for memory efficiency)

    Returns:
        Dictionary mapping words to embedding vectors
    """
    logger.info(f"Loading embeddings from {embedding_path}...")
    emb_dict = {}

    with open(embedding_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_vocab and i >= max_vocab:
                break

            parts = line.strip().split()
            if len(parts) > 2:  # word + vector
                word = parts[0]
                try:
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    emb_dict[word] = vector
                except ValueError:
                    logger.warning(f"Skipping invalid line {i+1}: {line[:50]}...")

            if (i + 1) % 100000 == 0:
                logger.info(f"Loaded {i+1} embeddings...")

    logger.info(f"Loaded {len(emb_dict)} embeddings")
    return emb_dict


def align_embeddings_with_vocab(
    emb_dict: dict[str, np.ndarray],
    tokenizer: Tokenizer,
    unk_init: str = "random"
) -> np.ndarray:
    """Align pre-trained embeddings with tokenizer vocabulary.

    Args:
        emb_dict: Dictionary of pre-trained embeddings
        tokenizer: GBLM tokenizer
        unk_init: Initialization for OOV tokens ("random", "zero", or "mean")

    Returns:
        Embedding matrix aligned with tokenizer vocabulary
    """
    # Get embedding dimension
    first_vec = next(iter(emb_dict.values()))
    embedding_dim = len(first_vec)

    vocab_size = len(tokenizer.itos)
    logger.info(f"Creating embedding matrix: {vocab_size} x {embedding_dim}")

    # Initialize embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    # Initialize OOV vector
    if unk_init == "random":
        np.random.seed(42)
        unk_vector = np.random.randn(embedding_dim).astype(np.float32) * 0.1
    elif unk_init == "zero":
        unk_vector = np.zeros(embedding_dim, dtype=np.float32)
    elif unk_init == "mean":
        # Use mean of all embeddings
        all_vecs = np.array(list(emb_dict.values()))
        unk_vector = all_vecs.mean(axis=0)
    else:
        raise ValueError(f"Unknown unk_init method: {unk_init}")

    # Align embeddings
    n_found = 0
    for idx, token in enumerate(tokenizer.itos):
        if token in emb_dict:
            embedding_matrix[idx] = emb_dict[token]
            n_found += 1
        else:
            # Use OOV initialization
            if token == tokenizer.pad_token:
                # Padding token gets zero embedding
                embedding_matrix[idx] = np.zeros(embedding_dim, dtype=np.float32)
            else:
                embedding_matrix[idx] = unk_vector.copy()

    coverage = n_found / vocab_size * 100
    logger.info(f"Vocabulary coverage: {n_found}/{vocab_size} ({coverage:.2f}%)")

    # Special token handling
    special_tokens = {
        tokenizer.pad_token: "zero",
        tokenizer.unk_token: "unk",
        tokenizer.eos_token: "unk"
    }

    for token, init_type in special_tokens.items():
        if token in tokenizer.stoi:
            idx = tokenizer.stoi[token]
            if init_type == "zero":
                embedding_matrix[idx] = np.zeros(embedding_dim, dtype=np.float32)
            elif init_type == "unk":
                embedding_matrix[idx] = unk_vector.copy()

    return embedding_matrix


def save_embedding_matrix(
    embedding_matrix: np.ndarray,
    output_path: Path,
    metadata: dict
) -> None:
    """Save embedding matrix and metadata.

    Args:
        embedding_matrix: Numpy array of embeddings
        output_path: Path to save the matrix
        metadata: Metadata about the embeddings
    """
    # Save numpy matrix
    np.save(output_path, embedding_matrix)
    logger.info(f"Saved embedding matrix to {output_path}")

    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build embedding matrix aligned with GBLM tokenizer"
    )
    parser.add_argument(
        "--embedding-source",
        type=str,
        choices=["glove", "custom"],
        default="glove",
        help="Source of embeddings"
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        help="Path to custom embedding file (if using custom source)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=50,
        help="Embedding dimension for GloVe (50, 100, 200, or 300)"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("artifacts/tokenizer.json"),
        help="Path to tokenizer file"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/embedding_matrix.npy"),
        help="Path to save embedding matrix"
    )
    parser.add_argument(
        "--unk-init",
        type=str,
        choices=["random", "zero", "mean"],
        default="random",
        help="Initialization method for OOV tokens"
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        help="Maximum vocabulary size to load from embeddings (for efficiency)"
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/embeddings"),
        help="Directory for downloading embeddings"
    )

    args = parser.parse_args()

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    logger.info(f"Tokenizer loaded: vocabulary size = {len(tokenizer.itos)}")

    # Get embedding file path
    if args.embedding_source == "glove":
        embedding_path = download_glove_embeddings(
            args.download_dir,
            args.embedding_dim
        )
    else:
        if not args.embedding_path:
            raise ValueError("--embedding-path required when using custom source")
        embedding_path = args.embedding_path
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    # Load embeddings
    emb_dict = load_pretrained_embeddings(embedding_path, args.max_vocab)

    # Align with vocabulary
    embedding_matrix = align_embeddings_with_vocab(
        emb_dict,
        tokenizer,
        args.unk_init
    )

    # Prepare metadata
    metadata = {
        "source": args.embedding_source,
        "source_file": str(embedding_path),
        "embedding_dim": embedding_matrix.shape[1],
        "vocab_size": embedding_matrix.shape[0],
        "unk_init": args.unk_init,
        "tokenizer_path": str(args.tokenizer_path),
        "n_embeddings_loaded": len(emb_dict),
    }

    # Save
    args.output_path.parent.mkdir(exist_ok=True, parents=True)
    save_embedding_matrix(embedding_matrix, args.output_path, metadata)

    logger.info("Done! Embedding matrix ready for use.")


if __name__ == "__main__":
    main()