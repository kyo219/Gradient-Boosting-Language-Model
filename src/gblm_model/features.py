"""Feature engineering module for GBLM model.

This module provides a pluggable feature engineering layer for the GBLM model,
allowing flexible addition of features beyond raw token IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from pathlib import Path

from ..gblm_data.tokenizer import Tokenizer


@dataclass
class FeatureConfig:
    """Configuration for feature engineering.

    Controls which features are enabled and their parameters.
    """
    # Basic features
    use_token_ids: bool = True  # Keep the original token ID features

    # Context-level statistics
    add_context_length: bool = True
    add_unk_count: bool = True
    add_type_token_ratio: bool = True

    # Embedding-based features
    use_embeddings: bool = False
    embedding_path: Optional[Path] = None
    add_mean_pooled_embedding: bool = False
    embedding_dim_reduction: Optional[int] = None  # If set, reduce embedding dims using PCA

    # Prototype/cluster-based features
    add_prototype_similarities: bool = False
    n_prototypes: int = 50  # Number of prototype clusters

    # Additional embedding-based similarity features
    add_last_token_similarities: bool = False  # Similarity of last token to vocab
    top_k_similar: int = 20  # Number of top similar tokens to consider
    add_position_weighted_embedding: bool = False  # Position-weighted context embedding
    add_max_pooled_embedding: bool = False  # Max-pooled embedding features

    # Advanced features (future extensions)
    add_position_correlations: bool = False
    add_ngram_features: bool = False

    def validate(self):
        """Validate the configuration."""
        if self.use_embeddings:
            if self.embedding_path is None:
                raise ValueError("embedding_path must be provided when use_embeddings is True")
            if not self.embedding_path.exists():
                raise ValueError(f"Embedding file not found: {self.embedding_path}")

        if self.add_mean_pooled_embedding and not self.use_embeddings:
            raise ValueError("add_mean_pooled_embedding requires use_embeddings to be True")

        if self.add_prototype_similarities and not self.use_embeddings:
            raise ValueError("add_prototype_similarities requires use_embeddings to be True")

        if self.add_last_token_similarities and not self.use_embeddings:
            raise ValueError("add_last_token_similarities requires use_embeddings to be True")

        if self.add_position_weighted_embedding and not self.use_embeddings:
            raise ValueError("add_position_weighted_embedding requires use_embeddings to be True")

        if self.add_max_pooled_embedding and not self.use_embeddings:
            raise ValueError("add_max_pooled_embedding requires use_embeddings to be True")


def build_features_for_lightgbm(
    X_ids: np.ndarray,
    tokenizer: Tokenizer,
    cfg: FeatureConfig,
    embedding_matrix: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list[int]]:
    """Transform raw context token IDs into final features for LightGBM.

    Args:
        X_ids: Context token IDs of shape (N, L) where N is number of samples
               and L is context length
        tokenizer: The tokenizer used for encoding/decoding
        cfg: Feature configuration specifying which features to add
        embedding_matrix: Optional pre-trained embeddings of shape (V, D) where
                         V is vocabulary size and D is embedding dimension

    Returns:
        X: Final feature matrix for LightGBM
        categorical_feature_indices: Indices of categorical features
    """
    cfg.validate()

    N, L = X_ids.shape
    features = []
    categorical_indices = []
    current_idx = 0

    # 1. Original token ID features (categorical)
    if cfg.use_token_ids:
        features.append(X_ids)
        categorical_indices.extend(range(current_idx, current_idx + L))
        current_idx += L

    # 2. Context-level statistics (numeric)
    stat_features = []

    if cfg.add_context_length:
        # Number of non-PAD tokens
        context_lengths = np.sum(X_ids != tokenizer.pad_id, axis=1, keepdims=True)
        stat_features.append(context_lengths)

    if cfg.add_unk_count:
        # Number of UNK tokens
        unk_counts = np.sum(X_ids == tokenizer.unk_id, axis=1, keepdims=True)
        stat_features.append(unk_counts)

    if cfg.add_type_token_ratio:
        # Type-token ratio (unique tokens / total non-PAD tokens)
        ttr_values = np.zeros((N, 1), dtype=np.float32)
        for i in range(N):
            non_pad_mask = X_ids[i] != tokenizer.pad_id
            non_pad_tokens = X_ids[i][non_pad_mask]
            if len(non_pad_tokens) > 0:
                unique_tokens = np.unique(non_pad_tokens)
                ttr_values[i] = len(unique_tokens) / len(non_pad_tokens)
        stat_features.append(ttr_values)

    if stat_features:
        stat_features = np.hstack(stat_features)
        features.append(stat_features)
        current_idx += stat_features.shape[1]

    # 3. Embedding-based features (numeric)
    if cfg.use_embeddings and embedding_matrix is not None:
        emb_features = []

        if cfg.add_mean_pooled_embedding:
            # Compute mean-pooled context embeddings
            context_emb = compute_mean_pooled_embeddings(
                X_ids, embedding_matrix, tokenizer.pad_id
            )

            # Optionally reduce dimensionality
            if cfg.embedding_dim_reduction is not None:
                context_emb = reduce_embedding_dims(
                    context_emb, cfg.embedding_dim_reduction
                )

            emb_features.append(context_emb)

        if cfg.add_prototype_similarities:
            # Compute similarities to prototype embeddings
            similarities = compute_prototype_similarities(
                X_ids, embedding_matrix, tokenizer.pad_id, cfg.n_prototypes
            )
            emb_features.append(similarities)

        if cfg.add_last_token_similarities:
            # Compute similarities between last token and vocabulary
            last_token_sims = compute_last_token_similarities(
                X_ids, embedding_matrix, tokenizer.pad_id, cfg.top_k_similar
            )
            emb_features.append(last_token_sims)

        if cfg.add_position_weighted_embedding:
            # Compute position-weighted context embedding
            pos_weighted_emb = compute_position_weighted_embeddings(
                X_ids, embedding_matrix, tokenizer.pad_id
            )
            if cfg.embedding_dim_reduction is not None:
                pos_weighted_emb = reduce_embedding_dims(
                    pos_weighted_emb, cfg.embedding_dim_reduction
                )
            emb_features.append(pos_weighted_emb)

        if cfg.add_max_pooled_embedding:
            # Compute max-pooled context embedding
            max_pooled_emb = compute_max_pooled_embeddings(
                X_ids, embedding_matrix, tokenizer.pad_id
            )
            if cfg.embedding_dim_reduction is not None:
                max_pooled_emb = reduce_embedding_dims(
                    max_pooled_emb, cfg.embedding_dim_reduction
                )
            emb_features.append(max_pooled_emb)

        if emb_features:
            emb_features = np.hstack(emb_features)
            features.append(emb_features)
            current_idx += emb_features.shape[1]

    # Combine all features
    if not features:
        raise ValueError("No features were generated. Check your FeatureConfig.")

    X = np.hstack(features) if len(features) > 1 else features[0]

    return X, categorical_indices


def compute_mean_pooled_embeddings(
    X_ids: np.ndarray,
    embedding_matrix: np.ndarray,
    pad_id: int,
) -> np.ndarray:
    """Compute mean-pooled embeddings for contexts.

    Args:
        X_ids: Token IDs of shape (N, L)
        embedding_matrix: Embeddings of shape (V, D)
        pad_id: ID of padding token

    Returns:
        Mean-pooled embeddings of shape (N, D)
    """
    N, L = X_ids.shape
    V, D = embedding_matrix.shape

    # Get embeddings for all tokens
    emb_seq = embedding_matrix[X_ids]  # (N, L, D)

    # Mask out padding tokens
    mask = (X_ids != pad_id)[..., None].astype(np.float32)  # (N, L, 1)

    # Compute mean pooling
    emb_sum = (emb_seq * mask).sum(axis=1)  # (N, D)
    lengths = mask.sum(axis=1).clip(min=1.0)  # (N, 1)
    context_emb = emb_sum / lengths  # (N, D)

    return context_emb


def reduce_embedding_dims(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
    """Reduce embedding dimensions using PCA.

    Args:
        embeddings: Embeddings of shape (N, D)
        target_dim: Target dimension

    Returns:
        Reduced embeddings of shape (N, target_dim)
    """
    from sklearn.decomposition import PCA

    if target_dim >= embeddings.shape[1]:
        return embeddings

    pca = PCA(n_components=target_dim, random_state=42)
    reduced = pca.fit_transform(embeddings)

    return reduced


def compute_prototype_similarities(
    X_ids: np.ndarray,
    embedding_matrix: np.ndarray,
    pad_id: int,
    n_prototypes: int,
) -> np.ndarray:
    """Compute cosine similarities to prototype embeddings.

    Args:
        X_ids: Token IDs of shape (N, L)
        embedding_matrix: Embeddings of shape (V, D)
        pad_id: ID of padding token
        n_prototypes: Number of prototype clusters

    Returns:
        Similarities of shape (N, n_prototypes)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity

    # Get context embeddings
    context_emb = compute_mean_pooled_embeddings(X_ids, embedding_matrix, pad_id)

    # Cluster vocabulary embeddings to get prototypes
    # Note: In production, we'd cache these prototypes
    kmeans = KMeans(n_clusters=n_prototypes, random_state=42, n_init=10)
    kmeans.fit(embedding_matrix)
    prototypes = kmeans.cluster_centers_  # (n_prototypes, D)

    # Compute cosine similarities
    similarities = cosine_similarity(context_emb, prototypes)  # (N, n_prototypes)

    return similarities


def compute_last_token_similarities(
    X_ids: np.ndarray,
    embedding_matrix: np.ndarray,
    pad_id: int,
    top_k: int = 20,
) -> np.ndarray:
    """Compute cosine similarities between last non-pad token and top-k most similar vocab tokens.

    This helps the model understand which tokens are semantically similar to the
    last token in the context, potentially improving next token prediction.

    Args:
        X_ids: Token IDs of shape (N, L)
        embedding_matrix: Embeddings of shape (V, D)
        pad_id: ID of padding token
        top_k: Number of top similar tokens to track

    Returns:
        Similarity features of shape (N, top_k)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    N, L = X_ids.shape
    V, D = embedding_matrix.shape

    # Find the last non-pad token for each sample
    last_token_features = np.zeros((N, top_k), dtype=np.float32)

    for i in range(N):
        # Find last non-pad token
        non_pad_indices = np.where(X_ids[i] != pad_id)[0]
        if len(non_pad_indices) > 0:
            last_token_id = X_ids[i, non_pad_indices[-1]]
            last_token_emb = embedding_matrix[last_token_id].reshape(1, -1)

            # Compute similarities to all vocabulary
            similarities = cosine_similarity(last_token_emb, embedding_matrix)[0]

            # Get top-k similarities (excluding self)
            # Sort in descending order and take top-k+1 (to exclude self)
            top_indices = np.argsort(-similarities)[:top_k+1]
            top_sims = similarities[top_indices]

            # Remove self-similarity if it's in top-k
            if last_token_id in top_indices[:top_k]:
                # Use top_k+1 similarities, excluding self
                mask = top_indices[:top_k+1] != last_token_id
                last_token_features[i] = top_sims[mask][:top_k]
            else:
                last_token_features[i] = top_sims[:top_k]

    return last_token_features


def compute_position_weighted_embeddings(
    X_ids: np.ndarray,
    embedding_matrix: np.ndarray,
    pad_id: int,
) -> np.ndarray:
    """Compute position-weighted mean embeddings, giving more weight to recent tokens.

    Args:
        X_ids: Token IDs of shape (N, L)
        embedding_matrix: Embeddings of shape (V, D)
        pad_id: ID of padding token

    Returns:
        Position-weighted embeddings of shape (N, D)
    """
    N, L = X_ids.shape
    V, D = embedding_matrix.shape

    # Get embeddings for all tokens
    emb_seq = embedding_matrix[X_ids]  # (N, L, D)

    # Create position weights (exponentially increasing towards end)
    position_weights = np.exp(np.arange(L) / L)  # Shape: (L,)
    position_weights = position_weights / position_weights.sum()  # Normalize

    # Expand for broadcasting
    position_weights = position_weights.reshape(1, L, 1)  # (1, L, 1)

    # Mask out padding tokens
    mask = (X_ids != pad_id)[..., None].astype(np.float32)  # (N, L, 1)

    # Apply position weights and mask
    weighted_emb = emb_seq * mask * position_weights  # (N, L, D)

    # Sum and normalize
    weighted_sum = weighted_emb.sum(axis=1)  # (N, D)
    weight_totals = (mask * position_weights).sum(axis=1).clip(min=1e-10)  # (N, 1)
    pos_weighted_emb = weighted_sum / weight_totals  # (N, D)

    return pos_weighted_emb


def compute_max_pooled_embeddings(
    X_ids: np.ndarray,
    embedding_matrix: np.ndarray,
    pad_id: int,
) -> np.ndarray:
    """Compute max-pooled embeddings for contexts.

    Args:
        X_ids: Token IDs of shape (N, L)
        embedding_matrix: Embeddings of shape (V, D)
        pad_id: ID of padding token

    Returns:
        Max-pooled embeddings of shape (N, D)
    """
    N, L = X_ids.shape
    V, D = embedding_matrix.shape

    # Get embeddings for all tokens
    emb_seq = embedding_matrix[X_ids]  # (N, L, D)

    # Mask out padding tokens (set to -inf for max pooling)
    mask = (X_ids != pad_id)[..., None].astype(np.float32)  # (N, L, 1)

    # Apply mask: set padded positions to very negative value
    masked_emb = emb_seq * mask + (1 - mask) * (-1e10)

    # Max pooling along sequence dimension
    max_pooled_emb = masked_emb.max(axis=1)  # (N, D)

    return max_pooled_emb


def load_embedding_matrix(
    embedding_path: Path,
    tokenizer: Tokenizer,
    embedding_dim: Optional[int] = None,
) -> np.ndarray:
    """Load and align pre-trained embeddings with tokenizer vocabulary.

    Args:
        embedding_path: Path to embedding file (numpy .npy or text format)
        tokenizer: Tokenizer with vocabulary
        embedding_dim: Expected embedding dimension (for validation)

    Returns:
        Embedding matrix of shape (V, D) aligned with tokenizer vocab
    """
    if embedding_path.suffix == '.npy':
        # Load pre-aligned numpy matrix
        embedding_matrix = np.load(embedding_path)
        if embedding_matrix.shape[0] != len(tokenizer.itos):
            raise ValueError(
                f"Embedding matrix size {embedding_matrix.shape[0]} doesn't match "
                f"vocabulary size {len(tokenizer.itos)}"
            )
    else:
        # Load text format embeddings and align
        embedding_matrix = align_text_embeddings(
            embedding_path, tokenizer, embedding_dim
        )

    return embedding_matrix


def align_text_embeddings(
    embedding_path: Path,
    tokenizer: Tokenizer,
    embedding_dim: Optional[int] = None,
) -> np.ndarray:
    """Align text format embeddings with tokenizer vocabulary.

    Args:
        embedding_path: Path to text embedding file
        tokenizer: Tokenizer with vocabulary
        embedding_dim: Expected embedding dimension

    Returns:
        Aligned embedding matrix
    """
    # Load embeddings from text file
    emb_dict = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 2:  # word + vector
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                emb_dict[word] = vector

    if not emb_dict:
        raise ValueError(f"No embeddings loaded from {embedding_path}")

    # Determine embedding dimension
    first_vec = next(iter(emb_dict.values()))
    actual_dim = len(first_vec)
    if embedding_dim is not None and actual_dim != embedding_dim:
        raise ValueError(
            f"Expected embedding dimension {embedding_dim}, got {actual_dim}"
        )

    # Create aligned embedding matrix
    V = len(tokenizer.itos)
    embedding_matrix = np.zeros((V, actual_dim), dtype=np.float32)

    # Initialize with small random values for OOV tokens
    np.random.seed(42)
    unk_vector = np.random.randn(actual_dim).astype(np.float32) * 0.1

    for idx, token in enumerate(tokenizer.itos):
        if token in emb_dict:
            embedding_matrix[idx] = emb_dict[token]
        else:
            # Use random initialization for OOV tokens
            embedding_matrix[idx] = unk_vector.copy()

    return embedding_matrix