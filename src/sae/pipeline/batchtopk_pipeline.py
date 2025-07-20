"""
BatchTopK SAE Training Pipeline

This module provides a complete pipeline for training BatchTopK Sparse Autoencoders on
embeddings generated from HelicalmRNA model. It uses the existing SAE pipeline structure
with minimal modifications.
"""

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch

from sae.models.batchtopk_sae import BatchTopKSAE

# Import the base pipeline and BatchTopK model
from sae.pipeline.base_pipeline import BaseSAETrainingPipeline


class BatchTopKSAETrainingPipeline(BaseSAETrainingPipeline):
    """
    BatchTopK SAE training pipeline that extends the base pipeline
    with BatchTopK-specific functionality
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        topk: int = 10,
        sparsity_weight: float = 0.01,
        sparsity_target: float = 0.05,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        cache_dir: str = "./outputs/batchtopk_embeddings_cache",
        model_save_dir: str = "./outputs/batchtopk_sae_models",
        layer_idx: Optional[int] = None,
        layer_name: Optional[str] = None
    ):
        """
        Initialize the BatchTopK training pipeline

        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Dimension of SAE hidden layer (number of features)
            topk: Number of top values to keep in BatchTopK
            sparsity_weight: Weight for sparsity penalty
            sparsity_target: Target sparsity level
            learning_rate: Learning rate for training
            device: Device to run training on
            cache_dir: Directory to cache embeddings
            model_save_dir: Directory to save trained models
            layer_idx: Layer index to extract embeddings from (None for final layer)
            layer_name: Layer name for identification (e.g., "final", "layer_0", etc.)
        """
        # Store BatchTopK-specific parameters
        self.topk = topk
        self.sparsity_target = sparsity_target

        # Call parent constructor
        super().__init__(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            sparsity_weight=sparsity_weight,
            learning_rate=learning_rate,
            device=device,
            cache_dir=cache_dir,
            model_save_dir=model_save_dir,
            layer_idx=layer_idx,
            layer_name=layer_name
        )

    def setup_sae_model(self):
        """Setup the BatchTopK SAE model"""
        self.sae_model = BatchTopKSAE(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            sparsity_weight=self.sparsity_weight,
            sparsity_target=self.sparsity_target,
            topk=self.topk
        ).to(self.device)

        self.logger.info(f"✅ BatchTopK SAE model setup complete: {self.embedding_dim} -> {self.hidden_dim} (topk={self.topk})")

    def train(self, epochs: int = 100) -> dict[str, list[float]]:
        """
        Train the BatchTopK SAE model using custom training loop
        Override parent method to handle BatchTopK-specific training requirements

        Args:
            epochs: Number of training epochs

        Returns:
            Training history
        """
        if self.sae_model is None:
            raise RuntimeError("SAE model not setup. Call setup_sae_model() first.")

        self.logger.info(f"Starting BatchTopK SAE training for {epochs} epochs...")

        # Custom training loop for BatchTopK SAE
        history = {"train_loss": [], "val_loss": []}

        # Create optimizer (excluding decoder weights which are updated orthogonally)
        optimizer = torch.optim.Adam([
            {"params": self.sae_model.encoder.parameters()},
            {"params": self.sae_model.encoder_bias},
            {"params": self.sae_model.pre_bias}
        ], lr=self.learning_rate)

        # Get training data
        train_data = []
        for batch in self.trainer.train_loader:
            train_data.append(batch[0])  # Input data
        train_data = torch.cat(train_data, dim=0).to(self.device)

        # Get validation data
        val_data = []
        for batch in self.trainer.val_loader:
            val_data.append(batch[0])  # Input data
        val_data = torch.cat(val_data, dim=0).to(self.device)

        batch_size = self.trainer.train_loader.batch_size
        num_samples = train_data.shape[0]

        # Initialize progress bar
        from tqdm import tqdm
        pbar = tqdm(range(epochs), desc="Training BatchTopK SAE", unit="epoch")

        for epoch in pbar:
            self.sae_model.train()

            # Create batches
            indices = torch.randperm(num_samples)
            total_loss = 0
            num_batches = 0

            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_data = train_data[batch_indices]

                # Training step
                loss, loss_dict = self.sae_model.train_step(
                    batch_data,
                    optimizer=optimizer,
                    lr=self.learning_rate
                )

                total_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_loss / num_batches

            # Validation
            self.sae_model.eval()
            with torch.no_grad():
                val_loss, _ = self.sae_model.compute_loss(val_data)
                avg_val_loss = val_loss.item()

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            # Update progress bar with current losses
            pbar.set_postfix({
                "Train Loss": f"{avg_train_loss:.6f}",
                "Val Loss": f"{avg_val_loss:.6f}"
            })

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        pbar.close()

        # Save the best model
        self.save_model("best_batchtopk_model.pth")

        # Store training history for plotting
        self._last_training_history = history

        # Update trainer logger with our custom training history for compatibility with plotting
        if self.trainer and hasattr(self.trainer, "logger"):
            self.trainer.logger.train_losses = history["train_loss"]
            self.trainer.logger.val_losses = history["val_loss"]

        self.logger.info("✅ BatchTopK SAE training completed!")
        return history

    def _get_model_save_data(self) -> dict[str, Any]:
        """Get model-specific save data for BatchTopK SAE"""
        return {
            "model_state_dict": self.sae_model.state_dict(),
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "topk": self.topk,
            "sparsity_weight": self.sparsity_weight,
            "sparsity_target": self.sparsity_target,
            "layer_idx": self.layer_idx,
            "layer_name": self.layer_name
        }

    def _get_metadata(self) -> dict[str, Any]:
        """Get metadata for BatchTopK SAE"""
        return {
            "layer_idx": self.layer_idx,
            "layer_name": self.layer_name,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "topk": self.topk,
            "sparsity_weight": self.sparsity_weight,
            "sparsity_target": self.sparsity_target,
            "model_type": "batchtopk"
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> "BatchTopKSAETrainingPipeline":
        """
        Load a trained BatchTopK SAE pipeline from a checkpoint directory.

        Args:
            checkpoint_path: Path to the checkpoint directory containing model files

        Returns:
            Loaded BatchTopKSAETrainingPipeline instance
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
            topk=metadata.get("topk", 10),
            sparsity_weight=metadata.get("sparsity_weight", 0.01),
            sparsity_target=metadata.get("sparsity_target", 0.05),
            layer_idx=metadata.get("layer_idx"),
            layer_name=metadata.get("layer_name"),
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

    def _update_from_checkpoint(self, checkpoint: dict[str, Any]):
        """Update pipeline parameters from checkpoint for BatchTopK"""
        super()._update_from_checkpoint(checkpoint)
        self.topk = checkpoint.get("topk", self.topk)
        self.sparsity_target = checkpoint.get("sparsity_target", self.sparsity_target)

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history for BatchTopK SAE with custom training loop"""
        if hasattr(self, "_last_training_history") and self._last_training_history:
            # Use our custom training history
            history = self._last_training_history

            plt.figure(figsize=(10, 6))
            plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
            plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"BatchTopK SAE Training History (TopK={self.topk})")
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info(f"✅ Training history plot saved to {save_path}")

            plt.show()
        else:
            # Fall back to parent method
            super().plot_training_history(save_path)

    def run_complete_pipeline(
        self,
        refseq_file: str,
        max_samples: int = 1000,
        embedding_dim: Optional[int] = None,  # Will be auto-detected from embeddings
        hidden_dim: int = 1000,  # Large for monosemantic features
        topk: int = 10,
        epochs: int = 50,
        batch_size: int = 4,
        filter_by_type: str = "mRNA",
        use_cds: bool = True,
        dataset_name: Optional[str] = None,
        apply_sequence_pooling: bool = False,
        **kwargs
    ):
        """
        Run the complete BatchTopK SAE pipeline using the base pipeline structure

        Args:
            refseq_file: Path to the RefSeq GenBank file
            max_samples: Number of samples to process
            embedding_dim: Expected embedding dimension (auto-detected if None)
            hidden_dim: Number of SAE features to learn (default: 1000 for monosemantic features)
            topk: Number of top values to keep in BatchTopK
            epochs: Number of training epochs
            batch_size: Batch size for training
            filter_by_type: Filter by molecule type (e.g., 'mRNA')
            use_cds: Whether to use CDS features for sequence extraction
            dataset_name: Optional name for logging purposes
            apply_sequence_pooling: Whether to apply sequence-level pooling (default: False)
            **kwargs: Additional arguments for the pipeline

        Returns:
            Trained BatchTopK pipeline
        """
        layer_info = f" from {self.layer_name or f'layer_{self.layer_idx}' if self.layer_idx is not None else 'final'}"
        print(f"🚀 Starting Complete BatchTopK SAE Training Pipeline{layer_info}")
        print("=" * 60)

        # Setup components using parent methods
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

        print("3. Setting up BatchTopK SAE model...")
        self.setup_sae_model()

        # Setup trainer
        print("4. Setting up trainer...")
        self.setup_trainer(train_loader, val_loader)

        # Train using overridden method
        print("5. Training BatchTopK SAE model...")
        history = self.train(epochs=epochs)

        # Plot results using parent method
        print("6. Plotting training history...")
        self.plot_training_history("outputs/batchtopk_training_history.png")

        print("✅ BatchTopK SAE Pipeline completed successfully!")
        return self


def run_complete_batchtopk_pipeline(
    refseq_file: str,
    max_samples: int = 1000,
    embedding_dim: Optional[int] = None,  # Will be auto-detected from embeddings
    hidden_dim: int = 1000,  # Large for monosemantic features
    topk: int = 10,
    epochs: int = 50,
    batch_size: int = 4,
    filter_by_type: str = "mRNA",
    use_cds: bool = True,
    dataset_name: Optional[str] = None,
    apply_sequence_pooling: bool = False,
    layer_idx: Optional[int] = None,
    layer_name: Optional[str] = None,
    **kwargs
) -> BatchTopKSAETrainingPipeline:
    """
    Run the complete BatchTopK SAE pipeline using the base pipeline structure

    Args:
        refseq_file: Path to the RefSeq GenBank file
        max_samples: Number of samples to process
        embedding_dim: Expected embedding dimension (auto-detected if None)
        hidden_dim: Number of SAE features to learn (default: 1000 for monosemantic features)
        topk: Number of top values to keep in BatchTopK
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
        Trained BatchTopK pipeline
    """
    # Extract pipeline-specific kwargs
    pipeline_kwargs = {}
    for key in ["sparsity_weight", "sparsity_target", "learning_rate", "device", "cache_dir", "model_save_dir"]:
        if key in kwargs:
            pipeline_kwargs[key] = kwargs.pop(key)

    # Initialize BatchTopK pipeline
    pipeline = BatchTopKSAETrainingPipeline(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        topk=topk,
        layer_idx=layer_idx,
        layer_name=layer_name,
        **pipeline_kwargs
    )

    return pipeline.run_complete_pipeline(
        refseq_file=refseq_file,
        max_samples=max_samples,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        topk=topk,
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
    print("BatchTopK SAE Training Pipeline Example")
    print("=" * 50)

    try:
        # Example 1: Train on final layer embeddings
        print("🔧 Example 1: Training on final layer embeddings")
        pipeline_final = run_complete_batchtopk_pipeline(
            refseq_file="../data/vertebrate_mammalian.1.rna.gbff",
            max_samples=100,
            hidden_dim=1000,  # Large for monosemantic features
            topk=10,
            epochs=10,
            layer_idx=None,  # Final layer
            layer_name="final"
        )

        # Example 2: Train on specific layer embeddings
        print("\n🔧 Example 2: Training on layer 0 embeddings")
        pipeline_layer0 = run_complete_batchtopk_pipeline(
            refseq_file="../data/vertebrate_mammalian.1.rna.gbff",
            max_samples=100,
            hidden_dim=1000,
            topk=10,
            epochs=10,
            layer_idx=0,  # First layer
            layer_name="layer_0"
        )

        print("✅ Both pipelines completed! Models saved to respective directories")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all dependencies are installed:")
        print("  poetry add helical torch numpy matplotlib seaborn tqdm")
