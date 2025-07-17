"""
Base classes for sequence foundation model wrappers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class PreprocessingConfig:
    """Configuration for sequence preprocessing."""

    # Model configuration
    model_name: str
    device: str = "cuda"
    batch_size: int = 4
    max_length: Optional[int] = None

    # Output configuration
    return_embeddings: bool = True
    return_attention: bool = False
    normalize_embeddings: bool = False

    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None


class SequenceModelWrapper(ABC):
    """
    Base class for sequence foundation model wrappers.

    This class provides a standardized interface for different sequence
    foundation models with preprocessing capabilities.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.model = None
        self._is_initialized = False

    @abstractmethod
    def _load_model(self) -> Any:
        """Load the underlying model."""
        pass

    @abstractmethod
    def _preprocess_sequences(self, sequences: list[str]) -> Any:
        """Preprocess raw sequences for the model."""
        pass

    @abstractmethod
    def _get_embeddings(self, processed_data: Any) -> torch.Tensor:
        """Extract embeddings from processed data."""
        pass

    def initialize(self):
        """Initialize the model if not already done."""
        if not self._is_initialized:
            self.model = self._load_model()
            self._is_initialized = True

    def preprocess_and_embed(self, sequences: list[str]) -> torch.Tensor:
        """
        Preprocess sequences and extract embeddings.

        Args:
            sequences: List of DNA/RNA sequences

        Returns:
            Tensor of embeddings with shape (num_sequences, embedding_dim)
        """
        self.initialize()

        # Preprocess sequences
        processed_data = self._preprocess_sequences(sequences)

        # Get embeddings
        embeddings = self._get_embeddings(processed_data)

        # Post-process embeddings
        if self.config.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def __call__(self, sequences: list[str]) -> torch.Tensor:
        """Convenience method to call preprocess_and_embed."""
        return self.preprocess_and_embed(sequences)
