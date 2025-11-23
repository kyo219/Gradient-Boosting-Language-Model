"""Module for generating GBLM training datasets."""

from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

from .tokenizer import Tokenizer


def make_gblm_training_data(
    texts: List[str],
    tokenizer: Tokenizer,
    context_length: int,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    random_seed: int = 42,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate GBLM (LightGBM) training data from texts.

    For each text:
      1. Encode: token_ids = [BOS] + encode(text) + [EOS]
      2. For each position i from 1 to len(token_ids)-1:
         - Target: y = token_ids[i]
         - Context: previous L tokens (left-padded if needed)
      3. Combine all samples from all texts

    Args:
        texts: List of document texts.
        tokenizer: Pre-built Tokenizer instance.
        context_length: Context window size (L).
        max_samples: Maximum number of samples to generate (None = all).
        shuffle: Whether to shuffle the samples.
        random_seed: Random seed for shuffling.
        verbose: Whether to print progress information.

    Returns:
        X: Feature matrix of shape (N, L) with context token IDs.
        y: Target array of shape (N,) with next token IDs.
    """
    if verbose:
        print(f"Generating GBLM training data from {len(texts)} texts...")
        print(f"Context length: {context_length}")

    contexts = []
    targets = []

    for text_idx, text in enumerate(texts):
        # Encode text with BOS and EOS tokens
        token_ids = tokenizer.encode(text, add_bos_eos=True)

        # Skip very short sequences
        if len(token_ids) < 2:
            continue

        # Generate samples for each position
        for i in range(1, len(token_ids)):
            # Target is the token at position i
            target = token_ids[i]

            # Context is the previous L tokens
            context_start = max(0, i - context_length)
            context_raw = token_ids[context_start:i]

            # Left-pad context to length L if needed
            if len(context_raw) < context_length:
                padding_length = context_length - len(context_raw)
                context = [tokenizer.pad_id] * padding_length + context_raw
            else:
                context = context_raw

            contexts.append(context)
            targets.append(target)

        if verbose and (text_idx + 1) % 100 == 0:
            print(f"Processed {text_idx + 1}/{len(texts)} texts...")

    # Convert to numpy arrays
    X = np.array(contexts, dtype=np.int32)
    y = np.array(targets, dtype=np.int32)

    if verbose:
        print(f"Generated {len(X)} samples before filtering")

    # Shuffle if requested
    if shuffle:
        np.random.seed(random_seed)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

    # Apply max_samples limit
    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]
        if verbose:
            print(f"Truncated to {max_samples} samples")

    if verbose:
        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Unique targets: {len(np.unique(y))}")

    return X, y


def to_lgb_dataset(
    X: np.ndarray,
    y: np.ndarray,
    categorical: bool = True,
    feature_names: Optional[List[str]] = None
):
    """
    Convert GBLM data to LightGBM Dataset format.

    Args:
        X: Feature matrix of shape (N, L).
        y: Target array of shape (N,).
        categorical: Whether to treat all features as categorical.
        feature_names: Optional names for features.

    Returns:
        lgb.Dataset: LightGBM dataset object.

    Note:
        This function requires lightgbm to be installed.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "LightGBM is required for this function. "
            "Install it with: pip install lightgbm"
        )

    n_features = X.shape[1]

    # Set categorical features if requested
    cat_features = list(range(n_features)) if categorical else []

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"token_t-{n_features-i}" for i in range(n_features)]

    # Create LightGBM dataset
    dataset = lgb.Dataset(
        X, label=y,
        categorical_feature=cat_features,
        feature_name=feature_names,
        free_raw_data=False
    )

    return dataset


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    save_dir: Path,
    prefix: str = "gblm_data"
) -> None:
    """
    Save GBLM dataset to disk.

    Args:
        X: Feature matrix.
        y: Target array.
        save_dir: Directory to save the data.
        prefix: Prefix for the saved files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save as compressed numpy arrays
    np.savez_compressed(
        save_dir / f"{prefix}.npz",
        X=X,
        y=y
    )

    print(f"Dataset saved to {save_dir / f'{prefix}.npz'}")

    # Also save metadata
    metadata = {
        "n_samples": len(X),
        "context_length": X.shape[1],
        "n_unique_targets": len(np.unique(y)),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype)
    }

    import json
    with open(save_dir / f"{prefix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {save_dir / f'{prefix}_metadata.json'}")


def load_dataset(
    load_dir: Path,
    prefix: str = "gblm_data"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load GBLM dataset from disk.

    Args:
        load_dir: Directory containing the saved data.
        prefix: Prefix of the saved files.

    Returns:
        X: Feature matrix.
        y: Target array.
    """
    load_dir = Path(load_dir)
    data_file = load_dir / f"{prefix}.npz"

    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    data = np.load(data_file)
    X = data['X']
    y = data['y']

    print(f"Loaded dataset: X={X.shape}, y={y.shape}")

    return X, y


def create_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and validation sets.

    Args:
        X: Feature matrix.
        y: Target array.
        val_ratio: Proportion of data to use for validation.
        random_seed: Random seed for splitting.

    Returns:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
    """
    n_samples = len(X)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val

    # Shuffle indices
    np.random.seed(random_seed)
    indices = np.random.permutation(n_samples)

    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create splits
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")

    return X_train, y_train, X_val, y_val


def analyze_dataset(X: np.ndarray, y: np.ndarray, tokenizer: Tokenizer) -> dict:
    """
    Analyze dataset statistics.

    Args:
        X: Feature matrix.
        y: Target array.
        tokenizer: Tokenizer for decoding.

    Returns:
        stats: Dictionary with dataset statistics.
    """
    stats = {
        "n_samples": len(X),
        "context_length": X.shape[1],
        "vocab_size": tokenizer.vocab_size,
        "n_unique_contexts": len(np.unique(X, axis=0)),
        "n_unique_targets": len(np.unique(y)),
    }

    # Target distribution
    target_counts = np.bincount(y)
    top_targets_idx = np.argsort(target_counts)[::-1][:10]

    stats["top_targets"] = [
        {
            "token": tokenizer.itos[idx] if idx < len(tokenizer.itos) else f"ID_{idx}",
            "count": int(target_counts[idx]),
            "frequency": float(target_counts[idx] / len(y))
        }
        for idx in top_targets_idx
    ]

    # Padding statistics
    pad_count = np.sum(X == tokenizer.pad_id)
    stats["padding_ratio"] = float(pad_count / X.size)

    # Unknown token statistics
    unk_count_X = np.sum(X == tokenizer.unk_id)
    unk_count_y = np.sum(y == tokenizer.unk_id)
    stats["unk_ratio_X"] = float(unk_count_X / X.size)
    stats["unk_ratio_y"] = float(unk_count_y / len(y))

    return stats