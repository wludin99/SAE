"""
SAE Training Pipeline

This module provides a complete pipeline for training Sparse Autoencoders on
embeddings generated from HelicalmRNA model.
"""

from pathlib import Path
from typing import Any, Optional

import torch

from sae.models.sae import SAE
from sae.pipeline.base_pipeline import BaseSAETrainingPipeline


class SAETrainingPipeline(BaseSAETrainingPipeline):
    """
    Complete pipeline for training regular SAE on HelicalmRNA embeddings
    """

    def setup_sae_model(self):
        """Setup the regular SAE model"""
        self.sae_model = SAE(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim
        ).to(self.device)

        self.logger.info(f"✅ SAE model setup complete: {self.embedding_dim} -> {self.hidden_dim}")

    def _get_model_save_data(self) -> dict[str, Any]:
        """Get model-specific save data for regular SAE"""
        return {
            "model_state_dict": self.sae_model.state_dict(),
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "sparsity_weight": self.sparsity_weight,
            "layer_idx": self.layer_idx,
            "layer_name": self.layer_name
        }

    def _get_metadata(self) -> dict[str, Any]:
        """Get metadata for regular SAE"""
        return {
            "layer_idx": self.layer_idx,
            "layer_name": self.layer_name,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "sparsity_weight": self.sparsity_weight,
            "model_type": "regular"
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> "SAETrainingPipeline":
        """
        Load a trained SAE pipeline from a checkpoint directory.

        Args:
            checkpoint_path: Path to the checkpoint directory containing model files

        Returns:
            Loaded SAETrainingPipeline instance
        """
        checkpoint_path = Path(checkpoint_path)

        # Load metadata
        metadata_path = checkpoint_path / "best_model.metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Load model weights
        model_path = checkpoint_path / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create pipeline instance with metadata
        # Handle device properly - if device is 'auto' or None, let the base pipeline handle it
        device = metadata.get("device")
        if device == "auto":
            device = None  # Let base pipeline auto-detect

        pipeline = cls(
            embedding_dim=metadata.get("embedding_dim", 1024),
            hidden_dim=metadata.get("hidden_dim", 1000),
            layer_idx=metadata.get("layer_idx"),
            layer_name=metadata.get("layer_name"),
            sparsity_weight=metadata.get("sparsity_weight", 0.01),
            device=device
        )

        # Setup SAE model before loading weights
        pipeline.setup_sae_model()

        # Load model weights
        checkpoint = torch.load(model_path, map_location=pipeline.device)
        if "model_state_dict" in checkpoint:
            pipeline.sae_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            pipeline.sae_model.load_state_dict(checkpoint)

        # Set model to evaluation mode
        pipeline.sae_model.eval()

        # Setup embedding generator
        pipeline.setup_embedding_generator()

        return pipeline

    def run_complete_pipeline(
        self,
        refseq_file: str,
        max_samples: int = 1000,
        embedding_dim: Optional[int] = None,  # Will be auto-detected from embeddings
        hidden_dim: int = 1000,  # Large for monosemantic features
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

        Args:
            refseq_file: Path to the RefSeq GenBank file
            max_samples: Number of samples to process
            embedding_dim: Expected embedding dimension (auto-detected if None)
            hidden_dim: Number of SAE features to learn (default: 1000 for monosemantic features)
            epochs: Number of training epochs
            batch_size: Batch size for training
            filter_by_type: Filter by molecule type (e.g., 'mRNA')
            use_cds: Whether to use CDS features for sequence extraction
            dataset_name: Optional name for logging purposes
            apply_sequence_pooling: Whether to apply sequence-level pooling (default: False)
            **kwargs: Additional arguments for the pipeline

        Returns:
            Trained pipeline
        """
        layer_info = f" from {self.layer_name or f'layer_{self.layer_idx}' if self.layer_idx is not None else 'final'}"
        print(f"🚀 Starting Complete SAE Training Pipeline{layer_info}")
        print("=" * 60)

        # Extract pipeline-specific kwargs
        pipeline_kwargs = {}
        for key in ["sparsity_weight", "learning_rate", "device", "cache_dir", "model_save_dir"]:
            if key in kwargs:
                pipeline_kwargs[key] = kwargs.pop(key)

        # Setup components
        print("1. Setting up embedding generator...")
        self.setup_embedding_generator(**kwargs)

        # Prepare data first to get embedding dimension
        print("2. Preparing training data...")
        train_loader, val_loader = self.prepare_data(
            refseq_file=refseq_file,
            max_samples=max_samples,
            batch_size=batch_size,
            filter_by_type=filter_by_type,
            use_cds=use_cds,
            dataset_name=dataset_name,
            apply_sequence_pooling=apply_sequence_pooling
        )

        print("3. Setting up SAE model...")
        self.setup_sae_model()

        # Setup trainer
        print("4. Setting up trainer...")
        self.setup_trainer(train_loader, val_loader)

        # Train
        print("5. Training SAE model...")
        history = self.train(epochs=epochs)

        # Convert training history to the format expected by ablation study
        # history is a list of dictionaries, convert to dict with lists
        if history and isinstance(history, list):
            converted_history = {
                "train_loss": [epoch["total_loss"] for epoch in history],
                "val_loss": [epoch.get("val_total_loss", epoch.get("total_loss", 0)) for epoch in history],
                "val_reconstruction_loss": [epoch.get("val_reconstruction_loss", 0) for epoch in history],
                "val_l0_sparsity": [epoch.get("val_l0_sparsity", 0) for epoch in history],
                "val_sparsity_percentage": [epoch.get("val_sparsity_percentage", 0) for epoch in history]
            }
            self._last_training_history = converted_history
        else:
            self._last_training_history = history

        # Plot results
        print("6. Plotting training history...")
        self.plot_training_history("outputs/training_history.png")

        print("✅ Pipeline completed successfully!")
        return self


def run_complete_pipeline(
    refseq_file: str,
    max_samples: int = 1000,
    embedding_dim: Optional[int] = None,  # Will be auto-detected from embeddings
    hidden_dim: int = 1000,  # Large for monosemantic features
    epochs: int = 50,
    batch_size: int = 4,
    filter_by_type: str = "mRNA",
    use_cds: bool = True,
    dataset_name: Optional[str] = None,
    apply_sequence_pooling: bool = False,
    layer_idx: Optional[int] = None,
    layer_name: Optional[str] = None,
    **kwargs
) -> SAETrainingPipeline:
    """
    Run the complete pipeline: generate embeddings -> train SAE -> extract features

    Args:
        refseq_file: Path to the RefSeq GenBank file
        max_samples: Number of samples to process
        embedding_dim: Expected embedding dimension (auto-detected if None)
        hidden_dim: Number of SAE features to learn (default: 1000 for monosemantic features)
        epochs: Number of training epochs
        batch_size: Batch size for training
        filter_by_type: Filter by molecule type (e.g., 'mRNA')
        use_cds: Whether to use CDS features for sequence extraction
        dataset_name: Optional name for logging purposes
        apply_sequence_pooling: Whether to apply sequence-level pooling (default: False)
        layer_idx: Layer index to extract embeddings from (None for final layer)
        layer_name: Layer name for identification (e.g., "final", "layer_0", etc.)
        **kwargs: Additional arguments for the pipeline

    Returns:
        Trained pipeline
    """
    # Extract pipeline-specific kwargs
    pipeline_kwargs = {}
    for key in ["sparsity_weight", "learning_rate", "device", "cache_dir", "model_save_dir"]:
        if key in kwargs:
            pipeline_kwargs[key] = kwargs.pop(key)

    # Initialize pipeline
    pipeline = SAETrainingPipeline(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        layer_idx=layer_idx,
        layer_name=layer_name,
        **pipeline_kwargs
    )

    return pipeline.run_complete_pipeline(
        refseq_file=refseq_file,
        max_samples=max_samples,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        filter_by_type=filter_by_type,
        use_cds=use_cds,
        dataset_name=dataset_name,
        apply_sequence_pooling=apply_sequence_pooling,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    print("SAE Training Pipeline Example")
    print("=" * 50)

    try:
        # Run complete pipeline with small dataset
        pipeline = run_complete_pipeline(
            refseq_file="../data/vertebrate_mammalian.1.rna.gbff",
            max_samples=100,
            hidden_dim=1000,  # Large for monosemantic features
            epochs=10
        )

        print(f"✅ Pipeline completed! Model saved to {pipeline.model_save_dir}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all dependencies are installed:")
        print("  poetry add helical torch numpy matplotlib seaborn tqdm")
