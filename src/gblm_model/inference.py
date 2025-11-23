"""
Inference and text generation utilities for GBLM.
"""

from pathlib import Path
from typing import List, Literal, Optional
import numpy as np
import lightgbm as lgb

from src.gblm_data.tokenizer import Tokenizer
from src.gblm_model.train import load_tokenizer  # Reuse


SamplingMethod = Literal["greedy", "top_k", "top_p", "temperature"]


def load_booster(model_path: Path) -> lgb.Booster:
    """
    Load LightGBM Booster from saved model file.

    Args:
        model_path: Path to the saved model file.

    Returns:
        Loaded LightGBM Booster.
    """
    booster = lgb.Booster(model_file=str(model_path))
    return booster


def prepare_context_ids(
    token_ids: List[int],
    context_length: int,
    pad_id: int,
) -> np.ndarray:
    """
    Extract last context_length tokens and apply left padding.

    Args:
        token_ids: Current token ID sequence.
        context_length: Context length L.
        pad_id: PAD token ID.

    Returns:
        X: shape (1, L) int32 ndarray.
    """
    # Take last context_length tokens
    context = token_ids[-context_length:]

    # Left padding if needed
    if len(context) < context_length:
        context = [pad_id] * (context_length - len(context)) + context

    X = np.asarray(context, dtype=np.int32).reshape(1, -1)
    return X


def predict_next_token_proba(
    booster: lgb.Booster,
    tokenizer: Tokenizer,
    token_ids: List[int],
    context_length: int,
) -> np.ndarray:
    """
    Predict next token probability distribution from current token_ids.

    Args:
        booster: Trained LightGBM Booster.
        tokenizer: Tokenizer instance.
        token_ids: Current token ID sequence.
        context_length: Context length.

    Returns:
        proba: shape (V,) ndarray. Probability for each token ID.
    """
    X = prepare_context_ids(
        token_ids=token_ids,
        context_length=context_length,
        pad_id=tokenizer.pad_id,
    )

    # Get prediction
    best_iter = booster.best_iteration if booster.best_iteration else booster.current_iteration()
    proba = booster.predict(X, num_iteration=best_iter)[0]

    # Safety: normalize probabilities
    proba = np.asarray(proba, dtype=np.float64)
    s = proba.sum()
    if s <= 0:
        # Fallback to uniform distribution
        proba = np.ones_like(proba) / len(proba)
    else:
        proba = proba / s

    return proba


def sample_from_proba_greedy(proba: np.ndarray) -> int:
    """
    Greedy sampling - select the token with maximum probability.

    Args:
        proba: Probability distribution over vocabulary.

    Returns:
        Selected token ID.
    """
    return int(np.argmax(proba))


def sample_from_proba_top_k(
    proba: np.ndarray,
    k: int = 10,
    temperature: float = 1.0,
) -> int:
    """
    Top-k sampling with optional temperature.

    Args:
        proba: Probability distribution over vocabulary.
        k: Number of top tokens to consider.
        temperature: Temperature for softmax (1.0 = no change).

    Returns:
        Sampled token ID.
    """
    V = proba.shape[0]
    k = min(k, V)

    # Apply temperature
    if temperature != 1.0 and temperature > 0:
        log_proba = np.log(proba + 1e-10)
        log_proba = log_proba / temperature
        proba = np.exp(log_proba - np.max(log_proba))
        proba = proba / proba.sum()

    # Get top k indices
    if k < V:
        # Use argpartition for efficiency
        topk_idx = np.argpartition(-proba, k-1)[:k]
    else:
        topk_idx = np.arange(V)

    topk_proba = proba[topk_idx]
    topk_proba = topk_proba / topk_proba.sum()

    sampled_idx = np.random.choice(topk_idx, p=topk_proba)
    return int(sampled_idx)


def sample_from_proba_top_p(
    proba: np.ndarray,
    p: float = 0.9,
    temperature: float = 1.0,
) -> int:
    """
    Top-p (nucleus) sampling with optional temperature.

    Args:
        proba: Probability distribution over vocabulary.
        p: Cumulative probability threshold.
        temperature: Temperature for softmax.

    Returns:
        Sampled token ID.
    """
    # Apply temperature
    if temperature != 1.0 and temperature > 0:
        log_proba = np.log(proba + 1e-10)
        log_proba = log_proba / temperature
        proba = np.exp(log_proba - np.max(log_proba))
        proba = proba / proba.sum()

    # Sort in descending order
    sorted_idx = np.argsort(-proba)
    sorted_proba = proba[sorted_idx]

    # Find cutoff where cumulative probability exceeds p
    cumsum = np.cumsum(sorted_proba)
    cutoff_idx = np.searchsorted(cumsum, p, side='right')
    cutoff_idx = max(1, cutoff_idx)  # At least one token

    # Keep only top tokens
    nucleus_idx = sorted_idx[:cutoff_idx]
    nucleus_proba = proba[nucleus_idx]
    nucleus_proba = nucleus_proba / nucleus_proba.sum()

    sampled_idx = np.random.choice(nucleus_idx, p=nucleus_proba)
    return int(sampled_idx)


def generate_text(
    booster: lgb.Booster,
    tokenizer: Tokenizer,
    prompt: str,
    context_length: int,
    max_new_tokens: int = 50,
    sampling: SamplingMethod = "greedy",
    top_k: int = 10,
    top_p: float = 0.9,
    temperature: float = 1.0,
    stop_at_eos: bool = True,
    verbose: bool = False,
) -> str:
    """
    Generate text using trained GBLM.

    Args:
        booster: Trained LightGBM Booster.
        tokenizer: Tokenizer instance.
        prompt: Starting text for generation.
        context_length: Context length L.
        max_new_tokens: Maximum number of tokens to generate.
        sampling: Sampling method ("greedy", "top_k", "top_p", "temperature").
        top_k: k value for top-k sampling.
        top_p: p value for top-p sampling.
        temperature: Temperature for softmax.
        stop_at_eos: Stop generation at EOS token.
        verbose: Print generation progress.

    Returns:
        Generated text (including prompt).
    """
    # Initial token sequence
    # Add BOS as sentence start marker
    token_ids = [tokenizer.bos_id] + tokenizer.encode(prompt, add_bos_eos=False)

    if verbose:
        print(f"Starting generation with prompt: {prompt}")
        print(f"Initial tokens ({len(token_ids)}): {token_ids[:20]}...")

    for i in range(max_new_tokens):
        # Predict next token probabilities
        proba = predict_next_token_proba(
            booster=booster,
            tokenizer=tokenizer,
            token_ids=token_ids,
            context_length=context_length,
        )

        # Sample next token
        if sampling == "greedy":
            next_id = sample_from_proba_greedy(proba)
        elif sampling == "top_k":
            next_id = sample_from_proba_top_k(proba, k=top_k, temperature=temperature)
        elif sampling == "top_p":
            next_id = sample_from_proba_top_p(proba, p=top_p, temperature=temperature)
        elif sampling == "temperature":
            # Temperature sampling without top-k or top-p
            next_id = sample_from_proba_top_k(
                proba, k=len(proba), temperature=temperature
            )
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")

        token_ids.append(next_id)

        if verbose and (i + 1) % 10 == 0:
            print(f"Generated {i + 1} tokens...")

        # Stop at EOS if requested
        if stop_at_eos and next_id == tokenizer.eos_id:
            if verbose:
                print("Reached EOS token, stopping generation.")
            break

    # Decode skipping special tokens (BOS/EOS)
    generated_text = tokenizer.decode(token_ids, skip_special=True)

    if verbose:
        print(f"Generation complete. Total tokens: {len(token_ids)}")

    return generated_text


def batch_generate(
    booster: lgb.Booster,
    tokenizer: Tokenizer,
    prompts: List[str],
    context_length: int,
    **kwargs,
) -> List[str]:
    """
    Generate text for multiple prompts.

    Args:
        booster: Trained LightGBM Booster.
        tokenizer: Tokenizer instance.
        prompts: List of starting texts.
        context_length: Context length.
        **kwargs: Additional arguments passed to generate_text.

    Returns:
        List of generated texts.
    """
    results = []
    for prompt in prompts:
        text = generate_text(
            booster=booster,
            tokenizer=tokenizer,
            prompt=prompt,
            context_length=context_length,
            **kwargs,
        )
        results.append(text)
    return results