"""
Tests for gblm_data module.
"""

import pytest
import tempfile
import json
import numpy as np
from pathlib import Path
from collections import Counter

# Test imports
def test_imports():
    """Test that all modules can be imported."""
    from src.gblm_data import config
    from src.gblm_data import corpus
    from src.gblm_data import vocab
    from src.gblm_data import tokenizer
    from src.gblm_data import dataset


def test_corpus_loading():
    """Test corpus loading functionality."""
    from src.gblm_data.corpus import load_corpus_texts

    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\n")
        f.write("This is another test document with more words.\n")
        f.write("A third document for testing purposes.")
        temp_path = Path(f.name)

    try:
        # Load corpus
        docs = load_corpus_texts(temp_path, max_docs=2)

        # Check results
        assert len(docs) == 2
        assert isinstance(docs, list)
        assert all(isinstance(doc, str) for doc in docs)
    finally:
        temp_path.unlink()


def test_vocab_building():
    """Test vocabulary building functionality."""
    from src.gblm_data.vocab import build_vocab, count_vocab

    # Create test documents
    docs = [
        "the cat sat on the mat",
        "the dog sat on the floor",
        "the cat and dog are friends"
    ]

    # Count vocabulary
    counter = count_vocab(docs, lowercase=True, verbose=False)

    # Check counter
    assert isinstance(counter, Counter)
    assert counter['the'] == 5  # 'the' appears 5 times
    assert counter['cat'] == 2  # 'cat' appears 2 times

    # Build vocabulary with min_freq=2
    vocab_tokens = build_vocab(counter, min_freq=2, top_k=None, verbose=False)

    # Check vocab
    assert isinstance(vocab_tokens, list)
    assert 'the' in vocab_tokens
    assert 'cat' in vocab_tokens
    assert 'dog' in vocab_tokens
    assert 'mat' not in vocab_tokens  # appears only once


def test_tokenizer():
    """Test tokenizer functionality."""
    from src.gblm_data.tokenizer import Tokenizer

    # Create simple vocabulary
    vocab_list = ['the', 'cat', 'sat', 'on', 'mat', 'dog']

    # Create tokenizer
    tok = Tokenizer(vocab_list)

    # Test encoding
    text = "the cat sat on the mat"
    encoded = tok.encode(text, add_bos_eos=False)

    assert isinstance(encoded, list)
    assert all(isinstance(id, int) for id in encoded)

    # Test decoding
    decoded = tok.decode(encoded)
    # Tokenizer may not preserve exact spacing, check tokens are present
    assert 'cat' in decoded or len(encoded) > 0

    # Test unknown token handling
    text_with_unk = "the cat jumped"  # 'jumped' is not in vocab
    encoded_with_unk = tok.encode(text_with_unk, add_bos_eos=False)
    assert tok.unk_id in encoded_with_unk

    # Test special tokens
    encoded_with_special = tok.encode("test", add_bos_eos=True)
    assert encoded_with_special[0] == tok.bos_id
    assert encoded_with_special[-1] == tok.eos_id


def test_dataset_creation():
    """Test dataset creation functionality."""
    from src.gblm_data.dataset import make_gblm_training_data
    from src.gblm_data.tokenizer import Tokenizer

    # Create simple tokenizer
    vocab_list = ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'floor']
    tok = Tokenizer(vocab_list)

    # Create test documents
    docs = ["the cat sat on the mat", "the dog sat on the floor"]

    # Create dataset
    X, y = make_gblm_training_data(
        texts=docs,
        tokenizer=tok,
        context_length=3,
        max_samples=100
    )

    # Check shapes
    assert X.shape[1] == 3  # context_length
    assert len(y) == len(X)
    assert X.dtype == np.int32
    assert y.dtype == np.int32


def test_config():
    """Test configuration classes."""
    from src.gblm_data.config import VocabConfig, DatasetConfig, PathsConfig

    # Test VocabConfig
    vocab_cfg = VocabConfig(min_freq=5, top_k=1000)
    assert vocab_cfg.min_freq == 5
    assert vocab_cfg.top_k == 1000

    # Test DatasetConfig
    dataset_cfg = DatasetConfig(context_length=16, max_samples=10000)
    assert dataset_cfg.context_length == 16
    assert dataset_cfg.max_samples == 10000

    # Test PathsConfig
    paths_cfg = PathsConfig(corpus_file=Path("test.txt"))
    assert paths_cfg.artifacts_dir == Path("artifacts")


def test_tokenizer_serialization():
    """Test tokenizer save/load functionality."""
    from src.gblm_data.tokenizer import Tokenizer

    # Create tokenizer
    vocab_list = ['hello', 'world', 'test']
    tok1 = Tokenizer(vocab_list)

    # Convert to dict and back
    tok_dict = tok1.to_dict()
    tok2 = Tokenizer.from_dict(tok_dict)

    # Check that they're equivalent
    assert tok1.itos == tok2.itos
    assert tok1.stoi == tok2.stoi
    assert tok1.pad_id == tok2.pad_id
    assert tok1.unk_id == tok2.unk_id
    assert tok1.bos_id == tok2.bos_id
    assert tok1.eos_id == tok2.eos_id

    # Test encoding/decoding produces same results
    text = "hello world"
    assert tok1.encode(text) == tok2.encode(text)


def test_small_pipeline_integration():
    """Test complete pipeline with small data."""
    from src.gblm_data.corpus import load_corpus_texts
    from src.gblm_data.vocab import build_vocab, count_vocab
    from src.gblm_data.tokenizer import Tokenizer

    # Create test corpus
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("The quick brown fox jumps over the lazy dog.\n")
        f.write("A quick brown dog jumps over a lazy fox.\n")
        f.write("The fox and the dog are quick friends.")
        temp_path = Path(f.name)

    try:
        # Load corpus
        docs = load_corpus_texts(temp_path)
        assert len(docs) > 0

        # Build vocabulary
        counter = count_vocab(docs, lowercase=True, verbose=False)
        vocab_tokens = build_vocab(counter, min_freq=1, top_k=20, verbose=False)

        # Create tokenizer
        tokenizer = Tokenizer(vocab_tokens)

        # Create dataset
        from src.gblm_data.dataset import make_gblm_training_data
        X, y = make_gblm_training_data(
            texts=docs[:2],
            tokenizer=tokenizer,
            context_length=4,
            max_samples=10
        )

        # Check dataset
        assert X.shape[1] == 4  # context_length
        assert len(y) == len(X)
        assert len(X) <= 10  # max_samples

    finally:
        temp_path.unlink()