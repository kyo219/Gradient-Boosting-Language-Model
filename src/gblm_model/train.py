"""
Training pipeline for GBLM using LightGBM.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from src.gblm_model.config import GBLMTrainConfig
from src.gblm_data.tokenizer import Tokenizer
from src.gblm_model.metrics import (
    compute_accuracy,
    compute_multi_logloss,
    compute_perplexity,
)
from src.gblm_model.features import (
    FeatureConfig,
    build_features_for_lightgbm,
    load_embedding_matrix,
)


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """
    Load Tokenizer from artifacts/tokenizer.json.

    Args:
        tokenizer_path: Path to tokenizer JSON file.

    Returns:
        Tokenizer instance.
    """
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Tokenizer.from_dict(data)


def load_gblm_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load X, y from artifacts/gblm_data.npz.

    Args:
        data_path: Path to NPZ data file.

    Returns:
        Tuple of (X, y) arrays.
    """
    npz = np.load(data_path)
    X = npz["X"]
    y = npz["y"]
    return X, y


def train_gblm(
    cfg: GBLMTrainConfig,
    feature_cfg: Optional[FeatureConfig] = None,
    verbose: bool = True,
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """
    Train GBLM (LightGBM multiclass) and return trained booster and metrics.

    Args:
        cfg: Training configuration.
        feature_cfg: Optional feature engineering configuration.
        verbose: Whether to print training progress.

    Returns:
        booster: Trained LightGBM Booster.
        metrics: Dictionary containing train/valid accuracy, logloss, perplexity, etc.
    """
    # 1. Resolve paths
    artifacts_dir = cfg.paths.artifacts_dir
    tokenizer_path = artifacts_dir / cfg.paths.tokenizer_json
    data_path = artifacts_dir / cfg.paths.data_npz
    model_path = artifacts_dir / cfg.paths.model_file

    # 2. Load tokenizer and data
    if verbose:
        print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = load_tokenizer(tokenizer_path)

    if verbose:
        print(f"Loading data from {data_path}...")
    X, y = load_gblm_data(data_path)

    num_classes = len(tokenizer.itos)
    if verbose:
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"Number of classes (vocabulary size): {num_classes}")

    # 3. Apply feature engineering if configured
    embedding_matrix = None
    categorical_features = None

    if feature_cfg is not None:
        if verbose:
            print(f"\nApplying feature engineering...")

        # Load embeddings if needed
        if feature_cfg.use_embeddings and feature_cfg.embedding_path:
            if verbose:
                print(f"Loading embedding matrix from {feature_cfg.embedding_path}...")
            embedding_matrix = load_embedding_matrix(
                feature_cfg.embedding_path,
                tokenizer
            )
            if verbose:
                print(f"Loaded embeddings: shape={embedding_matrix.shape}")

        # Transform features
        X, categorical_features = build_features_for_lightgbm(
            X,
            tokenizer,
            feature_cfg,
            embedding_matrix
        )
        if verbose:
            print(f"Features after engineering: shape={X.shape}")
            print(f"Categorical feature indices: {categorical_features}")
    else:
        # Default: all features are categorical (token IDs)
        n_features = X.shape[1]
        categorical_features = list(range(n_features))

    # 4. Train/valid split
    if verbose:
        print(f"\nSplitting data with validation size={cfg.split.valid_size}")

    # Use stratify cautiously - if some classes are rare, it might fail
    try:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=cfg.split.valid_size,
            shuffle=cfg.split.shuffle,
            random_state=cfg.split.random_seed,
            stratify=y,  # Maintain class distribution
        )
    except ValueError:
        # If stratify fails (rare classes), fall back to non-stratified split
        if verbose:
            print("Warning: Stratified split failed, falling back to non-stratified split")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=cfg.split.valid_size,
            shuffle=cfg.split.shuffle,
            random_state=cfg.split.random_seed,
        )

    if verbose:
        print(f"Train size: {X_train.shape[0]}, Valid size: {X_valid.shape[0]}")

    # 5. Create LightGBM Datasets
    if verbose:
        print(f"\nCreating LightGBM datasets with {len(categorical_features)} categorical features...")

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical_features,
        reference=dtrain,
        free_raw_data=False,
    )

    # 6. Build parameters
    params = cfg.lgbm.to_lgbm_params(num_class=num_classes)

    if verbose:
        print(f"\nStarting training with num_boost_round={cfg.lgbm.num_boost_round}")
        print(f"Early stopping rounds: {cfg.lgbm.early_stopping_rounds}")

    # 7. Train with early stopping
    callbacks = []

    if verbose:
        callbacks.append(lgb.log_evaluation(period=50))
    else:
        callbacks.append(lgb.log_evaluation(period=0))

    if cfg.lgbm.early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=cfg.lgbm.early_stopping_rounds))

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=cfg.lgbm.num_boost_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # 8. Compute predictions and metrics
    if verbose:
        print("\nComputing final metrics...")

    # Use best iteration for predictions
    best_iteration = booster.best_iteration if booster.best_iteration else cfg.lgbm.num_boost_round
    train_proba = booster.predict(X_train, num_iteration=best_iteration)
    valid_proba = booster.predict(X_valid, num_iteration=best_iteration)

    train_pred = train_proba.argmax(axis=1)
    valid_pred = valid_proba.argmax(axis=1)

    metrics = {
        "train_accuracy": float(compute_accuracy(y_train, train_pred)),
        "valid_accuracy": float(compute_accuracy(y_valid, valid_pred)),
        "train_logloss": float(compute_multi_logloss(y_train, train_proba)),
        "valid_logloss": float(compute_multi_logloss(y_valid, valid_proba)),
        "train_perplexity": float(compute_perplexity(y_train, train_proba)),
        "valid_perplexity": float(compute_perplexity(y_valid, valid_proba)),
        "best_iteration": int(best_iteration),
        "num_classes": num_classes,
        "train_size": int(X_train.shape[0]),
        "valid_size": int(X_valid.shape[0]),
        "num_features": int(X_train.shape[1]),
        "num_categorical_features": len(categorical_features),
        "config": {
            "train_split": asdict(cfg.split),
            "lgbm": asdict(cfg.lgbm),
            "features": asdict(feature_cfg) if feature_cfg else None,
        },
    }

    # 9. Save model
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    if verbose:
        print(f"\nModel saved to {model_path}")

    # Also save metrics
    metrics_path = artifacts_dir / "gblm_train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"Metrics saved to {metrics_path}")

    # Save feature config if present
    if feature_cfg is not None:
        feature_cfg_path = artifacts_dir / "feature_config.json"
        with open(feature_cfg_path, "w", encoding="utf-8") as f:
            json.dump(asdict(feature_cfg), f, indent=2, ensure_ascii=False, default=str)
        if verbose:
            print(f"Feature config saved to {feature_cfg_path}")

    return booster, metrics