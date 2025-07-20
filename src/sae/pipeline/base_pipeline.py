"""
Base SAE Training Pipeline

This module provides a base pipeline class that contains shared functionality
for training different types of Sparse Autoencoders on embeddings generated
from HelicalmRNA model.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .embedding_generator import EmbeddingGenerator
from sae.training import SAETrainer, TrainingConfig


class BaseSAETrainingPipeline:
    """
    Base pipeline class containing shared functionality for SAE training
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        sparsity_weight: float = 0.1,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        cache_dir: str = "./outputs/embeddings_cache",
        model_save_dir: str = "./outputs/sae_models",
        layer_idx: Optional[int] = None,
        layer_name: Optional[str] = None
    ):
        """
        Initialize the base training pipeline

        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Dimension of SAE hidden layer (number of features)
            sparsity_weight: Weight for sparsity loss
            learning_rate: Learning rate for training
            device: Device to run training on
            cache_dir: Directory to cache embeddings
            model_save_dir: Directory to save trained models
            layer_idx: Layer index to extract embeddings from (None for final layer)
            layer_name: Layer name for identification (e.g., "final", "layer_0", etc.)
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.sparsity_weight = sparsity_weight
        self.learning_rate = learning_rate
        self.layer_idx = layer_idx
        self.layer_name = layer_name or f"layer_{layer_idx}" if layer_idx is not None else "final"

        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup directories
        self.cache_dir = Path(cache_dir)
        self.model_save_dir = Path(model_save_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.embedding_generator = None
        self.sae_model = None
        self.trainer = None
        self.training_config = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_embedding_generator(self, model_name: str = "helicalmRNA", **kwargs):
        """Setup the embedding generator"""
        self.embedding_generator = EmbeddingGenerator(
            model_name=model_name,
            device=str(self.device),
            cache_dir=str(self.cache_dir),
            **kwargs
        )
        self.logger.info("✅ Embedding generator setup complete")

    def setup_sae_model(self):
        """
        Setup the SAE model - to be overridden by subclasses
        """
        raise NotImplementedError("Subclasses must implement setup_sae_model()")

    def prepare_data(
        self,
        refseq_file: str,
        max_samples: int,
        train_split: float = 0.8,
        batch_size: int = 4,
        filter_by_type: str = "mRNA",
        use_cds: bool = True,
        dataset_name: Optional[str] = None,
        apply_sequence_pooling: bool = False
    ) -> tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data from embeddings (RefSeq only)

        Args:
            refseq_file: Path to the RefSeq GenBank file
            max_samples: Number of samples to process
            train_split: Fraction of data for training
            batch_size: Batch size for dataloaders
            filter_by_type: Filter by molecule type (e.g., 'mRNA')
            use_cds: Whether to use CDS features for sequence extraction
            dataset_name: Optional name for logging purposes
            apply_sequence_pooling: Whether to apply sequence-level pooling (default: False)
                If False, 3D embeddings (batch, seq_len, embed_dim) are passed directly to SAE
                If True, 3D embeddings are pooled to 2D (batch, embed_dim)

        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self.embedding_generator is None:
            raise RuntimeError("Embedding generator not setup. Call setup_embedding_generator() first.")

        log_name = dataset_name or f"RefSeq_{Path(refseq_file).stem}"
        layer_info = f" from {self.layer_name}" if self.layer_name else ""
        self.logger.info(f"Generating embeddings from {log_name}{layer_info}: {refseq_file}")
        result = self.embedding_generator.generate_embeddings_from_refseq(
            refseq_file=refseq_file,
            max_samples=max_samples,
            filter_by_type=filter_by_type,
            use_cds=use_cds,
            layer_idx=self.layer_idx,
            dataset_name=dataset_name
        )

        embeddings = result["embeddings"]
        self.logger.info(f"Generated embeddings shape: {embeddings.shape}")

        # Auto-detect embedding dimension if not set
        if self.embedding_dim is None:
            if apply_sequence_pooling and len(embeddings.shape) == 3:
                # Apply sequence-level pooling (mean pooling)
                embeddings = embeddings.mean(dim=1)  # Shape: (num_sequences, embedding_dim)
                self.logger.info(f"Applied sequence-level pooling. New shape: {embeddings.shape}")

            # Now detect the embedding dimension
            self.embedding_dim = embeddings.shape[-1]
            self.logger.info(f"Auto-detected embedding dimension: {self.embedding_dim}")

        # Handle 3D embeddings (token-level) vs 2D embeddings (sequence-level)
        if apply_sequence_pooling and len(embeddings.shape) == 3:
            # Apply sequence-level pooling (mean pooling)
            embeddings = embeddings.mean(dim=1)  # Shape: (num_sequences, embedding_dim)
            self.logger.info(f"Applied sequence-level pooling. Final shape: {embeddings.shape}")

        # Split into train/val
        num_train = int(len(embeddings) * train_split)
        train_embeddings = embeddings[:num_train]
        val_embeddings = embeddings[num_train:]

        # Create datasets - for autoencoder, input and target are the same
        # Handle both 2D and 3D embeddings
        train_tensor = torch.FloatTensor(train_embeddings)
        val_tensor = torch.FloatTensor(val_embeddings)

        train_dataset = TensorDataset(train_tensor, train_tensor)
        val_dataset = TensorDataset(val_tensor, val_tensor)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True
        )

        self.logger.info(f"✅ Data prepared: {len(train_loader)} train batches, {len(val_loader)} val batches")
        return train_loader, val_loader

    def setup_trainer(self, train_loader: DataLoader, val_loader: DataLoader, **kwargs):
        """Setup the trainer with the prepared data"""
        if self.sae_model is None:
            raise RuntimeError("SAE model not setup. Call setup_sae_model() first.")

        # Create training config
        self.training_config = TrainingConfig(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            sparsity_weight=self.sparsity_weight,
            learning_rate=self.learning_rate,
            batch_size=train_loader.batch_size,
            device=str(self.device),
            log_dir=str(self.model_save_dir / "logs"),
            save_dir=str(self.model_save_dir),
            **kwargs
        )

        # Create trainer
        self.trainer = SAETrainer(
            model=self.sae_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.training_config
        )

        self.logger.info("✅ Trainer setup complete")

    def train(self, epochs: int = 100) -> dict[str, list[float]]:
        """
        Train the SAE model - to be overridden by subclasses if needed
        Default implementation uses the existing SAETrainer

        Args:
            epochs: Number of training epochs

        Returns:
            Training history
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not setup. Call setup_trainer() first.")

        self.logger.info(f"Starting training for {epochs} epochs...")

        # Train using the existing SAETrainer
        training_history = self.trainer.train(epochs)

        # Save the best model
        self.save_model("best_model.pth")

        self.logger.info("✅ Training completed!")
        return training_history

    def save_model(self, filename: str):
        """Save the trained model with metadata"""
        if self.sae_model is None:
            raise RuntimeError("SAE model not setup")

        save_path = self.model_save_dir / filename

        # Get model-specific save data
        save_data = self._get_model_save_data()

        # Save model state and metadata
        torch.save(save_data, str(save_path))

        # Also save layer information in a separate metadata file
        metadata = self._get_metadata()
        metadata_path = save_path.with_suffix(".metadata.json")
        import json
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"✅ Model saved to {save_path}")
        self.logger.info(f"✅ Layer metadata saved to {metadata_path}")

    def _get_model_save_data(self) -> dict[str, Any]:
        """
        Get model-specific save data - to be overridden by subclasses
        """
        return {
            "model_state_dict": self.sae_model.state_dict(),
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "sparsity_weight": self.sparsity_weight,
            "layer_idx": self.layer_idx,
            "layer_name": self.layer_name
        }

    def _get_metadata(self) -> dict[str, Any]:
        """
        Get metadata for saving - to be overridden by subclasses
        """
        return {
            "layer_idx": self.layer_idx,
            "layer_name": self.layer_name,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "sparsity_weight": self.sparsity_weight,
            "model_type": "base"
        }

    def load_model(self, filename: str):
        """Load a trained model"""
        load_path = self.model_save_dir / filename

        # Setup model if not already done
        if self.sae_model is None:
            self.setup_sae_model()

        # Load model state
        checkpoint = torch.load(str(load_path), map_location=self.device)
        self.sae_model.load_state_dict(checkpoint["model_state_dict"])

        # Update parameters from checkpoint
        self._update_from_checkpoint(checkpoint)

        self.logger.info(f"✅ Model loaded from {load_path}")

    def _update_from_checkpoint(self, checkpoint: dict[str, Any]):
        """
        Update pipeline parameters from checkpoint - to be overridden by subclasses
        """
        self.embedding_dim = checkpoint.get("embedding_dim", self.embedding_dim)
        self.hidden_dim = checkpoint.get("hidden_dim", self.hidden_dim)
        self.sparsity_weight = checkpoint.get("sparsity_weight", self.sparsity_weight)
        self.layer_idx = checkpoint.get("layer_idx", self.layer_idx)
        self.layer_name = checkpoint.get("layer_name", self.layer_name)

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history using the trainer's logger"""
        if self.trainer is None or not hasattr(self.trainer, "logger"):
            self.logger.warning("No training history available to plot")
            return

        # Use the trainer's logger to plot
        self.trainer.logger.plot_training_curves(save_path)

    def extract_features(self, embeddings: np.ndarray) -> np.ndarray:
        """Extract features from embeddings using the trained SAE"""
        if self.sae_model is None:
            raise RuntimeError("SAE model not loaded")

        self.sae_model.eval()
        embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)

        with torch.no_grad():
            _, activations = self.sae_model(embeddings_tensor)

        return activations.cpu().numpy()

    def run_complete_pipeline(
        self,
        refseq_file: str,
        max_samples: int = 1000,
        embedding_dim: Optional[int] = None,
        hidden_dim: int = 1000,
        epochs: int = 50,
        batch_size: int = 4,
        filter_by_type: str = "mRNA",
        use_cds: bool = True,
        dataset_name: Optional[str] = None,
        apply_sequence_pooling: bool = False,
        **kwargs
    ):
        """
        Run the complete pipeline: generate embeddings -> train SAE -> extract features
        To be overridden by subclasses with specific pipeline logic

        Args:
            refseq_file: Path to the RefSeq GenBank file
            max_samples: Number of samples to process
            embedding_dim: Expected embedding dimension (auto-detected if None)
            hidden_dim: Number of SAE features to learn
            epochs: Number of training epochs
            batch_size: Batch size for training
            filter_by_type: Filter by molecule type (e.g., 'mRNA')
            use_cds: Whether to use CDS features for sequence extraction
            dataset_name: Optional name for logging purposes
            apply_sequence_pooling: Whether to apply sequence-level pooling
            **kwargs: Additional arguments for the pipeline

        Returns:
            Trained pipeline
        """
        raise NotImplementedError("Subclasses must implement run_complete_pipeline()")
