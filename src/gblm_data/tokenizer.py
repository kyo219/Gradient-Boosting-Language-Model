"""Tokenizer class for converting text to token IDs and vice versa."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import json
from pathlib import Path

from .vocab import simple_word_tokenize, SPECIAL_TOKENS


@dataclass
class Tokenizer:
    """
    Word-level tokenizer with special tokens support.

    Attributes:
        itos: List mapping index to string token.
        stoi: Dictionary mapping string token to index.
        pad_id: ID for padding token.
        unk_id: ID for unknown token.
        bos_id: ID for beginning of sequence token.
        eos_id: ID for end of sequence token.
        lowercase: Whether to lowercase text during tokenization.
    """

    itos: List[str] = field(default_factory=list)
    stoi: Dict[str, int] = field(default_factory=dict)
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    lowercase: bool = True

    @classmethod
    def from_vocab(
        cls,
        vocab_tokens: List[str],
        lowercase: bool = True
    ) -> "Tokenizer":
        """
        Create a Tokenizer from vocabulary tokens.

        The final vocabulary will be:
        [<PAD>, <UNK>, <BOS>, <EOS>, ...vocab_tokens]

        Args:
            vocab_tokens: List of regular vocabulary tokens.
            lowercase: Whether to lowercase text during tokenization.

        Returns:
            tokenizer: Initialized Tokenizer instance.
        """
        # Combine special tokens with vocabulary
        itos = SPECIAL_TOKENS + vocab_tokens

        # Create string-to-index mapping
        stoi = {token: idx for idx, token in enumerate(itos)}

        return cls(
            itos=itos,
            stoi=stoi,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            lowercase=lowercase
        )

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.itos)

    @property
    def pad_token(self) -> str:
        """Return the padding token."""
        return self.itos[self.pad_id]

    @property
    def unk_token(self) -> str:
        """Return the unknown token."""
        return self.itos[self.unk_id]

    @property
    def bos_token(self) -> str:
        """Return the beginning of sequence token."""
        return self.itos[self.bos_id]

    @property
    def eos_token(self) -> str:
        """Return the end of sequence token."""
        return self.itos[self.eos_id]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of token strings.

        Args:
            text: Input text to tokenize.

        Returns:
            tokens: List of token strings.
        """
        return simple_word_tokenize(text, lowercase=self.lowercase)

    def encode(
        self,
        text: str,
        add_bos_eos: bool = False,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Encode text into a list of token IDs.

        Args:
            text: Input text to encode.
            add_bos_eos: If True, add BOS at start and EOS at end.
            max_length: Maximum sequence length (truncates if exceeded).

        Returns:
            token_ids: List of token IDs.
        """
        # Tokenize text
        tokens = self.tokenize(text)

        # Convert tokens to IDs (use UNK for out-of-vocabulary words)
        token_ids = [
            self.stoi.get(token, self.unk_id)
            for token in tokens
        ]

        # Add special tokens if requested
        if add_bos_eos:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]

        # Truncate if max_length is specified
        if max_length is not None and len(token_ids) > max_length:
            if add_bos_eos:
                # Keep BOS and EOS, truncate middle
                token_ids = token_ids[:max_length-1] + [self.eos_id]
            else:
                token_ids = token_ids[:max_length]

        return token_ids

    def decode(
        self,
        ids: List[int],
        skip_special: bool = True
    ) -> str:
        """
        Decode token IDs back into text.

        Args:
            ids: List of token IDs to decode.
            skip_special: If True, skip special tokens in output.

        Returns:
            text: Decoded text string.
        """
        tokens = []

        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                # Invalid ID, use UNK token
                token = self.unk_token
            else:
                token = self.itos[idx]

            # Skip special tokens if requested
            if skip_special and token in SPECIAL_TOKENS:
                continue

            tokens.append(token)

        # Join tokens with spaces
        text = " ".join(tokens)

        return text

    def batch_encode(
        self,
        texts: List[str],
        add_bos_eos: bool = False,
        max_length: Optional[int] = None,
        pad_to_length: Optional[int] = None
    ) -> List[List[int]]:
        """
        Encode multiple texts.

        Args:
            texts: List of texts to encode.
            add_bos_eos: If True, add BOS and EOS tokens.
            max_length: Maximum sequence length.
            pad_to_length: Pad sequences to this length (None = no padding).

        Returns:
            batch_ids: List of token ID lists.
        """
        batch_ids = []

        for text in texts:
            ids = self.encode(text, add_bos_eos, max_length)

            # Apply padding if requested
            if pad_to_length is not None:
                if len(ids) < pad_to_length:
                    # Pad on the right
                    ids = ids + [self.pad_id] * (pad_to_length - len(ids))
                elif len(ids) > pad_to_length:
                    # Truncate
                    ids = ids[:pad_to_length]

            batch_ids.append(ids)

        return batch_ids

    def batch_decode(
        self,
        batch_ids: List[List[int]],
        skip_special: bool = True
    ) -> List[str]:
        """
        Decode multiple token ID sequences.

        Args:
            batch_ids: List of token ID lists.
            skip_special: If True, skip special tokens.

        Returns:
            texts: List of decoded texts.
        """
        return [
            self.decode(ids, skip_special)
            for ids in batch_ids
        ]

    def to_dict(self) -> dict:
        """
        Convert tokenizer to a JSON-serializable dictionary.

        Returns:
            data: Dictionary containing tokenizer state.
        """
        return {
            "itos": self.itos,
            "stoi": self.stoi,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "lowercase": self.lowercase,
            "vocab_size": self.vocab_size
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Tokenizer":
        """
        Create a Tokenizer from a dictionary.

        Args:
            data: Dictionary containing tokenizer state.

        Returns:
            tokenizer: Reconstructed Tokenizer instance.
        """
        return cls(
            itos=data["itos"],
            stoi=data["stoi"],
            pad_id=data.get("pad_id", 0),
            unk_id=data.get("unk_id", 1),
            bos_id=data.get("bos_id", 2),
            eos_id=data.get("eos_id", 3),
            lowercase=data.get("lowercase", True)
        )

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save tokenizer to a JSON file.

        Args:
            file_path: Path to save the tokenizer.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"Tokenizer saved to {file_path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "Tokenizer":
        """
        Load tokenizer from a JSON file.

        Args:
            file_path: Path to the tokenizer file.

        Returns:
            tokenizer: Loaded Tokenizer instance.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __repr__(self) -> str:
        """Return string representation of the tokenizer."""
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, "
            f"lowercase={self.lowercase}, "
            f"special_tokens={SPECIAL_TOKENS})"
        )