"""
Tests for gblm_model module.
"""

import pytest
import tempfile
import json
import numpy as np
import lightgbm as lgb
from pathlib import Path

# Test imports
def test_imports():
    """Test that all modules can be imported."""
    from src.gblm_model import config
    from src.gblm_model import metrics
    from src.gblm_model import train
    from src.gblm_model import inference


def test_config_classes():
    """Test model configuration classes."""
    from src.gblm_model.config import (
        TrainSplitConfig,
        LightGBMConfig,
        PathsConfig,
        GBLMTrainConfig
    )

    # Test TrainSplitConfig
    split_cfg = TrainSplitConfig(valid_size=0.2, shuffle=True, random_seed=123)
    assert split_cfg.valid_size == 0.2
    assert split_cfg.shuffle == True
    assert split_cfg.random_seed == 123

    # Test LightGBMConfig
    lgbm_cfg = LightGBMConfig(
        learning_rate=0.05,
        num_leaves=32,
        num_boost_round=10
    )
    assert lgbm_cfg.learning_rate == 0.05
    assert lgbm_cfg.num_leaves == 32

    # Test params conversion
    params = lgbm_cfg.to_lgbm_params(num_class=100)
    assert params['num_class'] == 100
    assert params['learning_rate'] == 0.05
    assert 'metric' in params

    # Test PathsConfig
    paths_cfg = PathsConfig()
    assert paths_cfg.artifacts_dir == Path("artifacts")

    # Test GBLMTrainConfig
    train_cfg = GBLMTrainConfig(
        paths=paths_cfg,
        split=split_cfg,
        lgbm=lgbm_cfg
    )

    # Test serialization
    config_dict = train_cfg.to_dict()
    assert 'paths' in config_dict
    assert 'split' in config_dict
    assert 'lgbm' in config_dict

    # Test deserialization
    train_cfg2 = GBLMTrainConfig.from_dict(config_dict)
    assert train_cfg2.lgbm.learning_rate == 0.05


def test_metrics():
    """Test evaluation metrics."""
    from src.gblm_model.metrics import (
        compute_accuracy,
        compute_multi_logloss,
        compute_perplexity
    )

    # Create test data
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 2, 0])  # One mistake

    # Test accuracy
    acc = compute_accuracy(y_true, y_pred)
    assert acc == 0.8  # 4 out of 5 correct

    # Create probability predictions
    n_samples = len(y_true)
    n_classes = 3
    proba = np.zeros((n_samples, n_classes))

    # Set high probability for correct classes
    for i, true_class in enumerate(y_true):
        proba[i, :] = 0.1  # Base probability
        proba[i, true_class] = 0.8  # High prob for true class

    # Test logloss
    logloss = compute_multi_logloss(y_true, proba)
    assert logloss > 0  # Should be positive
    assert logloss < 1  # Should be reasonable for good predictions

    # Test perplexity
    perplexity = compute_perplexity(y_true, proba)
    assert perplexity > 1  # Perplexity is always > 1
    assert perplexity == np.exp(logloss)  # Definition of perplexity


def test_small_training():
    """Test training with minimal data."""
    from src.gblm_model.config import (
        GBLMTrainConfig,
        PathsConfig,
        TrainSplitConfig,
        LightGBMConfig
    )
    from src.gblm_data.tokenizer import Tokenizer

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create minimal tokenizer
        vocab = ['word1', 'word2', 'word3', 'word4', 'word5']
        tokenizer = Tokenizer(vocab)

        # Save tokenizer
        tokenizer_path = tmpdir / "tokenizer.json"
        with open(tokenizer_path, 'w') as f:
            json.dump(tokenizer.to_dict(), f)

        # Create minimal dataset
        n_samples = 100
        context_length = 4
        X = np.random.randint(0, len(tokenizer.itos), (n_samples, context_length), dtype=np.int32)
        y = np.random.randint(0, len(tokenizer.itos), n_samples, dtype=np.int32)

        # Save dataset
        data_path = tmpdir / "gblm_data.npz"
        np.savez(data_path, X=X, y=y)

        # Create config
        cfg = GBLMTrainConfig(
            paths=PathsConfig(
                artifacts_dir=tmpdir,
                tokenizer_json="tokenizer.json",
                data_npz="gblm_data.npz",
                model_file="test_model.txt"
            ),
            split=TrainSplitConfig(
                valid_size=0.2,
                shuffle=True,
                random_seed=42
            ),
            lgbm=LightGBMConfig(
                learning_rate=0.1,
                num_leaves=4,  # Very small
                num_boost_round=2,  # Very few rounds
                early_stopping_rounds=0,  # No early stopping
                verbose=-1  # Silent
            )
        )

        # Import and run training
        from src.gblm_model.train import train_gblm

        booster, metrics = train_gblm(cfg, verbose=False)

        # Check outputs
        assert isinstance(booster, lgb.Booster)
        assert 'train_accuracy' in metrics
        assert 'valid_accuracy' in metrics
        assert 'best_iteration' in metrics

        # Check model file was saved
        model_path = tmpdir / "test_model.txt"
        assert model_path.exists()


def test_inference():
    """Test inference functionality."""
    from src.gblm_model.inference import (
        prepare_context_ids,
        sample_from_proba_greedy,
        sample_from_proba_top_k
    )

    # Test context preparation
    token_ids = [1, 2, 3, 4, 5]
    context = prepare_context_ids(token_ids, context_length=4, pad_id=0)

    assert context.shape == (1, 4)
    assert context[0, -1] == 5  # Last token
    assert context[0, -2] == 4  # Second to last

    # Test with padding
    short_ids = [1, 2]
    context_padded = prepare_context_ids(short_ids, context_length=4, pad_id=0)
    assert context_padded[0, 0] == 0  # Padded
    assert context_padded[0, 1] == 0  # Padded
    assert context_padded[0, 2] == 1
    assert context_padded[0, 3] == 2

    # Test greedy sampling
    proba = np.array([0.1, 0.3, 0.6])  # Highest prob at index 2
    sampled = sample_from_proba_greedy(proba)
    assert sampled == 2

    # Test top-k sampling
    proba = np.array([0.4, 0.3, 0.2, 0.1])
    np.random.seed(42)
    sampled = sample_from_proba_top_k(proba, k=2)
    assert sampled in [0, 1]  # Should be one of top 2


def test_full_pipeline_minimal():
    """Test complete pipeline with minimal data."""
    from src.gblm_data.tokenizer import Tokenizer
    from src.gblm_model.config import (
        GBLMTrainConfig,
        PathsConfig,
        TrainSplitConfig,
        LightGBMConfig
    )
    from src.gblm_model.train import train_gblm
    from src.gblm_model.inference import load_booster, generate_text

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create tokenizer
        vocab = ['the', 'cat', 'dog', 'sat', 'on', 'mat', 'floor']
        tokenizer = Tokenizer(vocab)

        # Save tokenizer
        tokenizer_path = tmpdir / "tokenizer.json"
        with open(tokenizer_path, 'w') as f:
            json.dump(tokenizer.to_dict(), f)

        # Create simple patterns in data
        n_samples = 50
        context_length = 3
        X = []
        y = []

        # Create pattern with token indices
        # Get indices safely
        the_idx = tokenizer.stoi.get('the', 0)
        cat_idx = tokenizer.stoi.get('cat', 1)
        sat_idx = tokenizer.stoi.get('sat', 2)

        # Create pattern: 'the' -> 'cat'
        for _ in range(25):
            context = [the_idx, tokenizer.pad_id, tokenizer.pad_id]
            X.append(context)
            y.append(cat_idx)

        # Create pattern: 'cat' -> 'sat'
        for _ in range(25):
            context = [cat_idx, tokenizer.pad_id, tokenizer.pad_id]
            X.append(context)
            y.append(sat_idx)

        X = np.array(X, dtype=np.int32)
        y = np.array(y, dtype=np.int32)

        # Save dataset
        data_path = tmpdir / "gblm_data.npz"
        np.savez(data_path, X=X, y=y)

        # Train model
        cfg = GBLMTrainConfig(
            paths=PathsConfig(
                artifacts_dir=tmpdir,
                tokenizer_json="tokenizer.json",
                data_npz="gblm_data.npz",
                model_file="model.txt"
            ),
            split=TrainSplitConfig(valid_size=0.2, shuffle=False),
            lgbm=LightGBMConfig(
                num_leaves=4,
                num_boost_round=5,
                early_stopping_rounds=0,
                verbose=-1
            )
        )

        booster, metrics = train_gblm(cfg, verbose=False)

        # Test generation
        generated = generate_text(
            booster=booster,
            tokenizer=tokenizer,
            prompt="the",
            context_length=3,
            max_new_tokens=2,
            sampling="greedy",
            stop_at_eos=False,
            verbose=False
        )

        # Check that generation works (doesn't crash)
        assert isinstance(generated, str)
        assert len(generated) > 0