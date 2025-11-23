"""Configuration classes for GBLM data processing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import json


@dataclass
class VocabConfig:
    """Configuration for vocabulary building."""

    min_freq: int = 5  # Minimum frequency for a word to be included in vocab
    top_k: Optional[int] = 5000  # Top k most frequent words (None = no limit)
    lowercase: bool = True  # Convert text to lowercase before tokenization
    max_docs: Optional[int] = None  # Maximum number of documents to process (None = all)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    context_length: int = 16  # Context window size (L)
    max_samples: Optional[int] = None  # Maximum number of samples (None = all)
    shuffle: bool = True  # Whether to shuffle samples
    random_seed: int = 42  # Random seed for reproducibility


@dataclass
class PathsConfig:
    """Configuration for file paths."""

    corpus_file: Path  # Path to corpus file (text or CSV)
    text_column: Optional[str] = None  # Column name if CSV (None for text files)
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    is_csv: bool = False  # Whether the corpus is a CSV file


@dataclass
class GBLMConfig:
    """Main configuration class combining all sub-configs."""

    vocab: VocabConfig
    dataset: DatasetConfig
    paths: PathsConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GBLMConfig":
        """Create config from dictionary."""
        vocab_dict = config_dict.get("vocab", {})
        dataset_dict = config_dict.get("dataset", {})
        paths_dict = config_dict.get("paths", {})

        # Convert paths to Path objects
        if "corpus_file" in paths_dict:
            paths_dict["corpus_file"] = Path(paths_dict["corpus_file"])
        if "artifacts_dir" in paths_dict:
            paths_dict["artifacts_dir"] = Path(paths_dict["artifacts_dir"])

        return cls(
            vocab=VocabConfig(**vocab_dict),
            dataset=DatasetConfig(**dataset_dict),
            paths=PathsConfig(**paths_dict)
        )

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "GBLMConfig":
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "vocab": {
                "min_freq": self.vocab.min_freq,
                "top_k": self.vocab.top_k,
                "lowercase": self.vocab.lowercase,
                "max_docs": self.vocab.max_docs
            },
            "dataset": {
                "context_length": self.dataset.context_length,
                "max_samples": self.dataset.max_samples,
                "shuffle": self.dataset.shuffle,
                "random_seed": self.dataset.random_seed
            },
            "paths": {
                "corpus_file": str(self.paths.corpus_file),
                "text_column": self.paths.text_column,
                "artifacts_dir": str(self.paths.artifacts_dir),
                "is_csv": self.paths.is_csv
            }
        }

    def save_json(self, json_path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def create_default_config(corpus_file: Union[str, Path]) -> GBLMConfig:
    """Create a default configuration for the given corpus file."""
    corpus_path = Path(corpus_file)
    is_csv = corpus_path.suffix.lower() == '.csv'

    return GBLMConfig(
        vocab=VocabConfig(
            min_freq=5,
            top_k=5000,
            lowercase=True,
            max_docs=None
        ),
        dataset=DatasetConfig(
            context_length=16,
            max_samples=None,
            shuffle=True,
            random_seed=42
        ),
        paths=PathsConfig(
            corpus_file=corpus_path,
            text_column="text" if is_csv else None,
            artifacts_dir=Path("artifacts"),
            is_csv=is_csv
        )
    )