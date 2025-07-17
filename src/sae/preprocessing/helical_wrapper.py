"""
Helical model wrapper for embedding generation.

This wrapper focuses solely on interfacing with the Helical model for embedding generation.
All preprocessing should be handled by RefSeqPreprocessor before passing sequences to this wrapper.
"""

import logging
from typing import Any, Optional

import torch
from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig

from .base import PreprocessingConfig, SequenceModelWrapper

logger = logging.getLogger(__name__)


class HelicalWrapper(SequenceModelWrapper):
    """
    Wrapper for Helical mRNA model focused on embedding generation.

    This wrapper handles the Helical model interface and embedding extraction.
    All preprocessing (including codon processing and RefSeq handling) should be
    done by RefSeqPreprocessor before passing sequences to this wrapper.
    """

    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)
        self._helical_config = None

    def _load_model(self) -> HelixmRNA:
        """Load the Helical mRNA model."""
        try:
            # Create Helical configuration
            self._helical_config = HelixmRNAConfig(
                batch_size=self.config.batch_size,
                device=self.config.device,
                max_length=self.config.max_length
            )

            # Initialize the model
            model = HelixmRNA(configurer=self._helical_config)
            logger.info(f"Helical model loaded successfully on {self.config.device}")
            return model

        except Exception as e:
            logger.error(f"Failed to load Helical model: {e}")
            raise

    def _preprocess_sequences(self, sequences: list[str]) -> Any:
        """
        Use Helical's built-in preprocessing for already-processed sequences.

        Args:
            sequences: List of preprocessed DNA/RNA sequences (should already have codon tokens)

        Returns:
            Preprocessed data ready for Helical model
        """
        # Validate that sequences are not empty
        if not sequences:
            raise ValueError("No sequences provided for processing")

        # Use Helical's built-in preprocessing
        try:
            processed_data = self.model.process_data(sequences)
            logger.info(f"Processed {len(sequences)} sequences with Helical model")
            return processed_data
        except Exception as e:
            logger.error(f"Helical preprocessing failed: {e}")
            raise

    def _get_embeddings(self, processed_data: Any) -> torch.Tensor:
        """
        Extract embeddings from processed data using Helical model.

        Args:
            processed_data: Data processed by Helical's process_data method

        Returns:
            Tensor of embeddings
        """
        try:
            embeddings = self.model.get_embeddings(processed_data)

            # Ensure embeddings are on the correct device
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.to(self.config.device)

            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def get_model_info(self) -> dict:
        """
        Get information about the loaded Helical model.

        Returns:
            Dictionary with model information
        """
        if not self._is_initialized:
            return {"status": "Model not initialized"}

        return {
            "model_type": "HelixmRNA",
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "status": "Model initialized and ready"
        }


def create_helical_wrapper(
    device: str = "cuda",
    batch_size: int = 4,
    max_length: Optional[int] = None,
    normalize_embeddings: bool = False
) -> HelicalWrapper:
    """
    Convenience function to create a Helical wrapper with default settings.

    Args:
        device: Device to run the model on
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        normalize_embeddings: Whether to normalize embeddings

    Returns:
        Configured HelicalWrapper instance
    """
    config = PreprocessingConfig(
        model_name="helical",
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        normalize_embeddings=normalize_embeddings
    )

    return HelicalWrapper(config)
