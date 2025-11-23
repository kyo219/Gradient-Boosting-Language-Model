"""Module for vocabulary building and word tokenization."""

import re
from collections import Counter
from typing import Iterable, List, Optional
import json
from pathlib import Path


# Regular expression for simple word tokenization
WORD_REGEX = re.compile(r"[^a-z0-9']+")

# Special tokens used in the vocabulary
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


def simple_word_tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Simple word-level tokenizer for English text.

    Specification:
      - Optionally converts to lowercase
      - Replaces non-alphanumeric characters (except apostrophes) with spaces
      - Splits on whitespace

    Args:
        text: Input text to tokenize.
        lowercase: If True, convert text to lowercase.

    Returns:
        tokens: List of token strings.
    """
    if lowercase:
        text = text.lower()

    # Replace non-word characters with spaces
    text = WORD_REGEX.sub(" ", text)

    # Split on whitespace and filter empty strings
    tokens = text.split()

    return tokens


def count_vocab(
    texts: Iterable[str],
    lowercase: bool = True,
    tokenizer_fn: Optional[callable] = None,
    verbose: bool = False
) -> Counter:
    """
    Count word frequencies from a collection of texts.

    Args:
        texts: Iterable of text documents.
        lowercase: Whether to lowercase text before tokenization.
        tokenizer_fn: Optional custom tokenizer function.
                     If None, uses simple_word_tokenize.
        verbose: If True, print progress information.

    Returns:
        counter: Counter object with word frequencies.
    """
    if tokenizer_fn is None:
        tokenizer_fn = lambda text: simple_word_tokenize(text, lowercase)

    counter = Counter()

    for i, text in enumerate(texts):
        tokens = tokenizer_fn(text)
        counter.update(tokens)

        if verbose and (i + 1) % 100 == 0:
            print(f"Processed {i + 1} documents...")

    if verbose:
        print(f"Total unique tokens: {len(counter)}")
        print(f"Total token count: {sum(counter.values())}")

    return counter


def build_vocab(
    counter: Counter,
    min_freq: int = 5,
    top_k: Optional[int] = 5000,
    verbose: bool = False
) -> List[str]:
    """
    Build vocabulary from word frequency counter.

    Specification:
      - Filters words with frequency < min_freq
      - Sorts remaining words by frequency (descending)
      - If top_k is specified, keeps only top_k most frequent words
      - Returns list of regular tokens (special tokens not included)

    Args:
        counter: Counter object with word frequencies.
        min_freq: Minimum frequency threshold for inclusion.
        top_k: Maximum vocabulary size (None = no limit).
        verbose: If True, print vocabulary statistics.

    Returns:
        vocab_tokens: List of vocabulary tokens (without special tokens).
    """
    # Filter by minimum frequency
    filtered_words = [
        (word, count) for word, count in counter.items()
        if count >= min_freq
    ]

    # Sort by frequency (descending), then alphabetically for stability
    filtered_words.sort(key=lambda x: (-x[1], x[0]))

    # Apply top_k limit if specified
    if top_k is not None and len(filtered_words) > top_k:
        filtered_words = filtered_words[:top_k]

    # Extract just the words
    vocab_tokens = [word for word, _ in filtered_words]

    if verbose:
        print(f"Vocabulary size: {len(vocab_tokens)}")
        if vocab_tokens:
            print(f"Most common words: {vocab_tokens[:10]}")
            print(f"Least common words: {vocab_tokens[-10:]}")

    return vocab_tokens


def save_vocab(
    vocab_tokens: List[str],
    file_path: Path,
    save_frequencies: Optional[Counter] = None
) -> None:
    """
    Save vocabulary to a JSON file.

    Args:
        vocab_tokens: List of vocabulary tokens.
        file_path: Path to save the vocabulary.
        save_frequencies: Optional counter to save word frequencies.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_data = {
        "tokens": vocab_tokens,
        "size": len(vocab_tokens),
        "special_tokens": SPECIAL_TOKENS
    }

    if save_frequencies is not None:
        # Save frequencies for the vocabulary tokens only
        freq_dict = {
            token: save_frequencies[token]
            for token in vocab_tokens
            if token in save_frequencies
        }
        vocab_data["frequencies"] = freq_dict

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)

    print(f"Vocabulary saved to {file_path}")


def load_vocab(file_path: Path) -> List[str]:
    """
    Load vocabulary from a JSON file.

    Args:
        file_path: Path to the vocabulary file.

    Returns:
        vocab_tokens: List of vocabulary tokens.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    if isinstance(vocab_data, list):
        # Old format: just a list of tokens
        return vocab_data
    elif isinstance(vocab_data, dict):
        # New format: dictionary with metadata
        return vocab_data.get("tokens", [])
    else:
        raise ValueError(f"Invalid vocabulary format in {file_path}")


def analyze_oov_rate(
    texts: List[str],
    vocab_tokens: List[str],
    lowercase: bool = True,
    sample_size: Optional[int] = None
) -> dict:
    """
    Analyze out-of-vocabulary (OOV) rate for given texts and vocabulary.

    Args:
        texts: List of text documents.
        vocab_tokens: List of vocabulary tokens.
        lowercase: Whether to lowercase text.
        sample_size: Number of texts to sample for analysis (None = all).

    Returns:
        stats: Dictionary with OOV statistics.
    """
    vocab_set = set(vocab_tokens)

    if sample_size is not None and len(texts) > sample_size:
        import random
        texts = random.sample(texts, sample_size)

    total_tokens = 0
    oov_tokens = 0
    oov_types = set()

    for text in texts:
        tokens = simple_word_tokenize(text, lowercase)
        for token in tokens:
            total_tokens += 1
            if token not in vocab_set:
                oov_tokens += 1
                oov_types.add(token)

    oov_rate = (oov_tokens / total_tokens * 100) if total_tokens > 0 else 0

    return {
        "total_tokens": total_tokens,
        "oov_tokens": oov_tokens,
        "oov_rate": oov_rate,
        "oov_types": len(oov_types),
        "sample_oov_words": list(oov_types)[:20]
    }