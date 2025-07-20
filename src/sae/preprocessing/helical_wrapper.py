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

    # Layer mapping for Helical model (0-indexed)
    # Model structure: M+M*+M+M+ where M=Mamba, *=Attention, +=MLP
    # Layers: 0=Mamba, 1=MLP, 2=Mamba, 3=Attention+MLP, 4=Mamba, 5=MLP, 6=Mamba, 7=MLP, 8=final_norm
    LAYER_MAPPING = {
        "initial": -1,  # After initial embedding, before first layer
        "after_mlp_1": 1,  # After first MLP layer
        "after_mlp_2": 3,  # After second MLP layer (attention + MLP)
        "after_mlp_3": 5,  # After third MLP layer
        "final": 8,  # After final normalization
    }

    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)
        self._helical_config = None
        self._hooks = []
        self._layer_outputs = {}

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

    def _setup_layer_hooks(self):
        """Setup hooks to capture layer outputs."""
        self._clear_hooks()
        self._layer_outputs = {}

        def create_hook(layer_name):
            def hook(module, input, output):
                self._layer_outputs[layer_name] = output
            return hook

        # Check if model has a 'model' attribute (the actual underlying model)
        if hasattr(self.model, "model"):
            actual_model = self.model.model

            # Hook into the actual model's layers
            if hasattr(actual_model, "embeddings"):
                self._hooks.append(actual_model.embeddings.register_forward_hook(
                    create_hook("initial")
                ))

            if hasattr(actual_model, "layers"):
                mlp_layers = [1, 3, 5]  # MLP layer indices
                layer_name_mapping = {1: "after_mlp_1", 3: "after_mlp_2", 5: "after_mlp_3"}

                for i in mlp_layers:
                    if i < len(actual_model.layers):
                        layer = actual_model.layers[i]
                        if hasattr(layer, "feed_forward"):
                            layer_name = layer_name_mapping.get(i, f"mlp_{i}")
                            self._hooks.append(layer.feed_forward.register_forward_hook(
                                create_hook(layer_name)
                            ))

            if hasattr(actual_model, "norm_f"):
                self._hooks.append(actual_model.norm_f.register_forward_hook(
                    create_hook("final")
                ))
        else:
            logger.warning("Model does not have 'model' attribute - cannot set up hooks")

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_layer_embeddings(self, sequences: list[str], layer_name: str) -> torch.Tensor:
        """
        Get embeddings from a specific layer.

        Args:
            sequences: List of preprocessed sequences
            layer_name: Name of the layer to extract from (see LAYER_MAPPING)

        Returns:
            Tensor of embeddings from the specified layer
        """
        if layer_name not in self.LAYER_MAPPING:
            raise ValueError(f"Unknown layer: {layer_name}. Available: {list(self.LAYER_MAPPING.keys())}")

        self.initialize()
        self._setup_layer_hooks()

        try:
            # Process sequences to trigger hooks
            processed_data = self._preprocess_sequences(sequences)

            # Use get_embeddings which we know works
            with torch.no_grad():
                _ = self.model.get_embeddings(processed_data)

            # Get the requested layer output
            if layer_name in self._layer_outputs:
                embeddings = self._layer_outputs[layer_name]

                # Handle tuple outputs (some layers return tuples)
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]  # Use first element

                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.to(self.config.device)
                    return embeddings
                else:
                    raise ValueError(f"Unexpected output type for layer {layer_name}: {type(embeddings)}")
            else:
                raise ValueError(f"Layer {layer_name} output not captured. Available: {list(self._layer_outputs.keys())}")

        finally:
            self._clear_hooks()

    def get_all_layer_embeddings(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """
        Get embeddings from all available layers in one forward pass.

        Args:
            sequences: List of preprocessed sequences

        Returns:
            Dictionary mapping layer names to their embeddings
        """
        self.initialize()
        self._setup_layer_hooks()

        try:
            # Process sequences to trigger hooks
            processed_data = self._preprocess_sequences(sequences)

            # Forward pass to capture all layer outputs
            with torch.no_grad():
                _ = self.model.get_embeddings(processed_data)

            # Collect all layer outputs
            layer_embeddings = {}
            for layer_name, embeddings in self._layer_outputs.items():
                # Skip mlp_7 since we don't care about it
                if layer_name == "mlp_7":
                    continue

                # Handle tuple outputs
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]  # Use first element

                if isinstance(embeddings, torch.Tensor):
                    layer_embeddings[layer_name] = embeddings.to(self.config.device)

            return layer_embeddings

        finally:
            self._clear_hooks()

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
            "available_layers": list(self.LAYER_MAPPING.keys()),
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
