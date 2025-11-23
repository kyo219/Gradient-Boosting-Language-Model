"""
Configuration classes for GBLM model training and inference.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class TrainSplitConfig:
    """Configuration for train/validation split."""
    valid_size: float = 0.1           # Validation set ratio (0.0-1.0)
    shuffle: bool = True               # Whether to shuffle before splitting
    random_seed: int = 42              # Random seed for shuffling


@dataclass
class LightGBMConfig:
    """Configuration for LightGBM model parameters."""
    # LightGBM parameters (vanilla)
    objective: str = "multiclass"
    num_class: Optional[int] = None   # Vocabulary size V (will be overridden)
    learning_rate: float = 0.1
    num_leaves: int = 64
    max_depth: int = -1                # -1 = no limit
    min_data_in_leaf: int = 20
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    bagging_freq: int = 0
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    num_boost_round: int = 500
    early_stopping_rounds: int = 20
    n_jobs: int = -1
    verbose: int = -1                  # -1 = silent

    def to_lgbm_params(self, num_class: int) -> Dict[str, Any]:
        """
        Convert to LightGBM Booster parameters dict.

        Args:
            num_class: Number of classes (vocabulary size).

        Returns:
            Dictionary of LightGBM parameters.
        """
        return {
            "objective": self.objective,
            "num_class": num_class,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            # Use multi_logloss and multiclasserror as default metrics
            "metric": ["multi_logloss", "multi_error"],
        }


@dataclass
class PathsConfig:
    """Configuration for file paths."""
    artifacts_dir: Path = Path("artifacts")
    tokenizer_json: str = "tokenizer.json"
    data_npz: str = "gblm_data.npz"
    model_file: str = "gblm_model.txt"       # LightGBM model save path

    def resolve_paths(self) -> None:
        """Ensure artifacts directory exists."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class GBLMTrainConfig:
    """Main configuration class for GBLM training."""
    paths: PathsConfig
    split: TrainSplitConfig
    lgbm: LightGBMConfig

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        paths_dict = asdict(self.paths)
        # Convert Path to string for JSON serialization
        paths_dict["artifacts_dir"] = str(paths_dict["artifacts_dir"])
        return {
            "paths": paths_dict,
            "split": asdict(self.split),
            "lgbm": asdict(self.lgbm),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GBLMTrainConfig":
        """Create from dictionary."""
        paths_data = data.get("paths", {})
        # Convert string to Path if needed
        if "artifacts_dir" in paths_data:
            paths_data["artifacts_dir"] = Path(paths_data["artifacts_dir"])
        paths = PathsConfig(**paths_data)
        split = TrainSplitConfig(**data.get("split", {}))
        lgbm = LightGBMConfig(**data.get("lgbm", {}))
        return cls(paths=paths, split=split, lgbm=lgbm)