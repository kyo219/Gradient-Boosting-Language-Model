"""Module for loading corpus data from various file formats."""

from pathlib import Path
from typing import List, Optional, Union
import pandas as pd


def load_corpus_texts(
    file_path: Union[str, Path],
    text_column: Optional[str] = None,
    max_docs: Optional[int] = None,
    is_csv: bool = False,
    min_doc_length: int = 10,
) -> List[str]:
    """
    Load text corpus from a file (CSV or plain text).

    Args:
        file_path: Path to the corpus file.
        text_column: Column name containing text (for CSV files).
        max_docs: Maximum number of documents to load (None = all).
        is_csv: Whether the file is a CSV.
        min_doc_length: Minimum document length in characters (filters out short texts).

    Returns:
        texts: List of document texts.

    Raises:
        FileNotFoundError: If the corpus file doesn't exist.
        ValueError: If CSV column is not found.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")

    texts = []

    if is_csv:
        # Load from CSV file
        if text_column is None:
            text_column = "text"  # Default column name

        try:
            df = pd.read_csv(file_path)

            if text_column not in df.columns:
                raise ValueError(
                    f"Column '{text_column}' not found in CSV. "
                    f"Available columns: {list(df.columns)}"
                )

            # Extract text column
            texts_series = df[text_column].dropna()
            texts = texts_series.astype(str).tolist()

        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file: {e}")

    else:
        # Load from plain text file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Try to split by common story delimiters
        # Looking for patterns like "Story Title." or multiple newlines
        import re

        # Split by double newlines or title patterns
        # This handles cases where stories are separated by blank lines
        parts = re.split(r'\n\s*\n+', content)

        # Filter out very short parts (likely empty or titles only)
        texts = [
            part.strip()
            for part in parts
            if len(part.strip()) > min_doc_length
        ]

        # If we got very few documents, try treating each paragraph as a document
        if len(texts) < 10:
            # Split by single newlines and filter
            paragraphs = content.split('\n')
            texts = [
                para.strip()
                for para in paragraphs
                if len(para.strip()) > min_doc_length
            ]

    # Apply max_docs limit if specified
    if max_docs is not None and len(texts) > max_docs:
        texts = texts[:max_docs]

    # Final validation
    if not texts:
        raise ValueError(f"No valid text documents found in {file_path}")

    print(f"Loaded {len(texts)} documents from {file_path.name}")

    # Print sample statistics
    lengths = [len(text) for text in texts]
    avg_length = sum(lengths) / len(lengths)
    print(f"Average document length: {avg_length:.0f} characters")
    print(f"Min/Max length: {min(lengths)}/{max(lengths)} characters")

    return texts


def split_into_chunks(
    texts: List[str],
    chunk_size: int = 1000,
    overlap: int = 100
) -> List[str]:
    """
    Split long texts into overlapping chunks.

    This is useful when dealing with very long documents that should be
    treated as multiple training examples.

    Args:
        texts: List of document texts.
        chunk_size: Target size of each chunk in characters.
        overlap: Number of overlapping characters between chunks.

    Returns:
        chunks: List of text chunks.
    """
    chunks = []

    for text in texts:
        if len(text) <= chunk_size:
            chunks.append(text)
        else:
            # Split into overlapping chunks
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end]
                chunks.append(chunk)

                # Move start position with overlap
                start += chunk_size - overlap

                # Avoid tiny final chunks
                if len(text) - start < overlap:
                    break

    return chunks


def load_corpus_from_lines(
    file_path: Union[str, Path],
    max_lines: Optional[int] = None,
    min_line_length: int = 10,
) -> List[str]:
    """
    Load corpus where each line is treated as a separate document.

    Args:
        file_path: Path to the text file.
        max_lines: Maximum number of lines to load.
        min_line_length: Minimum line length in characters.

    Returns:
        texts: List of text lines as documents.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    texts = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break

            line = line.strip()
            if len(line) >= min_line_length:
                texts.append(line)

    return texts