"""
SAE Pipeline Components Example

This script breaks down the SAE training pipeline into individual components,
making it easier to use in Jupyter notebooks and understand each step.

The pipeline consists of:
1. Embedding Generation: Extract embeddings from HelicalmRNA model
2. Data Preparation: Create training and validation dataloaders
3. Model Setup: Initialize the Sparse Autoencoder
4. Training: Train the model with sparsity constraints
5. Visualization: Plot training progress and results
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

# Add the src directory to the path
script_dir = Path(__file__).parent
src_path = script_dir.parent / "src"
sys.path.append(str(src_path))

# Import SAE components
from sae.losses.losses import SAELoss
from sae.models.sae import SAE
from sae.pipeline.embedding_generator import EmbeddingGenerator
from sae.training.trainer import SAETrainer, TrainingConfig

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def setup_embedding_generator(
    model_name: str = "helicalmRNA",
    device: str = "cuda",
    max_length: int = 1024,
    batch_size: int = 4,
    cache_dir: str = "./outputs/embeddings_cache",
    normalize_embeddings: bool = False
):
    """
    Setup the embedding generator to extract embeddings from HelicalmRNA model.

    Args:
        model_name: Name of the HelicalmRNA model to use
        device: Device to run the model on ('cuda', 'cpu', or None for auto)
        max_length: Maximum sequence length for processing
        batch_size: Batch size for embedding generation
        cache_dir: Directory to cache embeddings
        normalize_embeddings: Whether to normalize embeddings

    Returns:
        EmbeddingGenerator instance
    """
    print("🔧 Setting up embedding generator...")

    embedding_generator = EmbeddingGenerator(
        model_name=model_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        cache_dir=cache_dir,
        normalize_embeddings=normalize_embeddings
    )

    print("✅ Embedding generator setup complete!")
    print(f"   Model: {embedding_generator.model_name}")
    print(f"   Device: {embedding_generator.device}")
    print(f"   Max length: {embedding_generator.max_length}")
    print(f"   Batch size: {embedding_generator.batch_size}")

    return embedding_generator


def prepare_data(
    embedding_generator: EmbeddingGenerator,
    refseq_file: str,
    max_samples: int = 1000,
    batch_size: int = 4,
    filter_by_type: str = "mRNA",
    use_cds: bool = True,
    dataset_name: str = "Example_Dataset",
    layer_name: str = "final",
    apply_sequence_pooling: bool = False
):
    """
    Generate embeddings and create training/validation dataloaders.

    Args:
        embedding_generator: Initialized EmbeddingGenerator
        refseq_file: Path to the RefSeq GenBank file
        max_samples: Number of samples to process
        batch_size: Batch size for training
        filter_by_type: Filter by molecule type (e.g., 'mRNA')
        use_cds: Whether to use CDS features for sequence extraction
        dataset_name: Name for logging purposes
        layer_name: Layer name for identification
        apply_sequence_pooling: Whether to apply sequence-level pooling

    Returns:
        tuple: (train_loader, val_loader, embedding_dim)
    """
    print("🔧 Preparing training data...")

    # Generate embeddings from RefSeq file
    print(f"   Generating embeddings from {dataset_name} from {layer_name}: {refseq_file}")

    if layer_name == "final":
        # Use final layer embeddings
        result = embedding_generator.generate_embeddings_from_refseq(
            refseq_file=refseq_file,
            max_samples=max_samples,
            filter_by_type=filter_by_type,
            use_cds=use_cds,
            dataset_name=dataset_name
        )
        embeddings = result["embeddings"]
    else:
        # Use specific layer embeddings
        result = embedding_generator.generate_layer_embeddings_from_refseq(
            refseq_file=refseq_file,
            layer_name=layer_name,
            max_samples=max_samples,
            filter_by_type=filter_by_type,
            use_cds=use_cds,
            dataset_name=dataset_name
        )
        embeddings = result["embeddings"]

    print(f"   Generated embeddings shape: {embeddings.shape}")

    # Auto-detect embedding dimension
    if len(embeddings.shape) == 3:
        # (num_samples, seq_len, embedding_dim)
        embedding_dim = embeddings.shape[2]
        if apply_sequence_pooling:
            # Apply sequence-level pooling (mean across sequence dimension)
            embeddings = embeddings.mean(dim=1)  # (num_samples, embedding_dim)
    else:
        # (num_samples, embedding_dim)
        embedding_dim = embeddings.shape[1]

    print(f"   Auto-detected embedding dimension: {embedding_dim}")

    # Convert to PyTorch tensors
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Split into train/validation sets (80/20 split)
    num_samples = embeddings_tensor.shape[0]
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    train_embeddings = embeddings_tensor[:train_size]
    val_embeddings = embeddings_tensor[train_size:]

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_embeddings, train_embeddings)
    val_dataset = torch.utils.data.TensorDataset(val_embeddings, val_embeddings)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("✅ Data preparation complete!")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Training samples: {train_size}")
    print(f"   Validation samples: {val_size}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Batch size: {batch_size}")

    return train_loader, val_loader, embedding_dim


def setup_sae_model(
    embedding_dim: int,
    hidden_dim: int = 1000,
    sparsity_weight: float = 0.01,
    sparsity_target: float = 0.05
):
    """
    Initialize the Sparse Autoencoder model.

    Args:
        embedding_dim: Input embedding dimension
        hidden_dim: Number of SAE features to learn
        sparsity_weight: Weight for sparsity penalty
        sparsity_target: Target sparsity level

    Returns:
        SAE model instance
    """
    print("🔧 Setting up SAE model...")

    sae_model = SAE(
        input_size=embedding_dim,
        hidden_size=hidden_dim,
        sparsity_weight=sparsity_weight,
        sparsity_target=sparsity_target
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_model = sae_model.to(device)

    print("✅ SAE model setup complete!")
    print(f"   Input size: {embedding_dim}")
    print(f"   Hidden size: {hidden_dim}")
    print(f"   Model parameters: {sum(p.numel() for p in sae_model.parameters()):,}")
    print(f"   Device: {device}")

    return sae_model


def setup_trainer(
    sae_model: SAE,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    learning_rate: float = 0.001,
    sparsity_weight: float = 0.01,
    sparsity_target: float = 0.05
):
    """
    Setup the trainer with optimizer, loss function, and training configuration.

    Args:
        sae_model: Initialized SAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        learning_rate: Learning rate for optimizer
        sparsity_weight: Weight for sparsity penalty
        sparsity_target: Target sparsity level

    Returns:
        SAETrainer instance
    """
    print("🔧 Setting up trainer...")

    # Setup training configuration
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=train_loader.batch_size,
        device=str(next(sae_model.parameters()).device)
    )

    # Setup loss function
    loss_fn = SAELoss(
        sparsity_weight=sparsity_weight,
        sparsity_target=sparsity_target
    )

    # Setup trainer
    trainer = SAETrainer(
        model=sae_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        loss_fn=loss_fn
    )

    print("✅ Trainer setup complete!")
    print(f"   Optimizer: {type(trainer.optimizer).__name__}")
    print(f"   Learning rate: {trainer.optimizer.param_groups[0]['lr']}")
    print(f"   Loss function: {type(trainer.loss_fn).__name__}")
    print(f"   Sparsity weight: {trainer.loss_fn.sparsity_weight}")
    print(f"   Sparsity target: {trainer.loss_fn.sparsity_target}")

    return trainer


def train_model(
    trainer: SAETrainer,
    epochs: int = 50,
    save_history: bool = True
):
    """
    Train the SAE model and monitor progress.

    Args:
        trainer: Initialized SAETrainer
        epochs: Number of training epochs
        save_history: Whether to save training history

    Returns:
        Training history (list of dictionaries)
    """
    print(f"🚀 Starting training for {epochs} epochs...")
    print("=" * 60)

    # Train the model
    history = trainer.train(epochs=epochs)

    print("\n✅ Training completed!")
    print(f"   Total epochs: {len(history)}")
    print(f"   Final training loss: {history[-1]['total_loss']:.6f}")
    print(f"   Final validation loss: {history[-1].get('val_total_loss', 'N/A')}")
    if "val_l0_sparsity" in history[-1]:
        print(f"   Final L0 sparsity: {history[-1]['val_l0_sparsity']:.1f}")

    # Save training history if requested
    if save_history:
        save_dir = Path("outputs")
        save_dir.mkdir(exist_ok=True)

        history_path = save_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"   Training history saved to: {history_path}")

    return history


def plot_training_history(history: list, save_path: str = "outputs/training_history.png"):
    """
    Create comprehensive plots to visualize the training progress.

    Args:
        history: Training history (list of dictionaries)
        save_path: Path to save the plot
    """
    print("📊 Creating training visualization...")

    # Convert history to DataFrame for easier plotting
    history_df = pd.DataFrame(history)
    history_df["epoch"] = range(1, len(history_df) + 1)

    print("📊 Training History Summary:")
    print(history_df[["epoch", "total_loss", "reconstruction_loss", "sparsity_loss"]].tail())

    # Create comprehensive training plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"SAE Training Progress - {len(history_df)} Epochs", fontsize=16)

    # 1. Total Loss
    ax1 = axes[0, 0]
    ax1.plot(history_df["epoch"], history_df["total_loss"], "b-", linewidth=2, label="Training")
    if "val_total_loss" in history_df.columns:
        ax1.plot(history_df["epoch"], history_df["val_total_loss"], "r--", linewidth=2, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Total Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Reconstruction Loss
    ax2 = axes[0, 1]
    ax2.plot(history_df["epoch"], history_df["reconstruction_loss"], "g-", linewidth=2, label="Training")
    if "val_reconstruction_loss" in history_df.columns:
        ax2.plot(history_df["epoch"], history_df["val_reconstruction_loss"], "m--", linewidth=2, label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Reconstruction Loss")
    ax2.set_title("Reconstruction Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Sparsity Loss
    ax3 = axes[0, 2]
    ax3.plot(history_df["epoch"], history_df["sparsity_loss"], "orange", linewidth=2, label="Training")
    if "val_sparsity_loss" in history_df.columns:
        ax3.plot(history_df["epoch"], history_df["val_sparsity_loss"], "brown", linewidth=2, label="Validation")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Sparsity Loss")
    ax3.set_title("Sparsity Loss")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. L0 Sparsity (if available)
    ax4 = axes[1, 0]
    if "val_l0_sparsity" in history_df.columns:
        ax4.plot(history_df["epoch"], history_df["val_l0_sparsity"], "purple", linewidth=2, marker="o")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("L0 Sparsity")
        ax4.set_title("L0 Sparsity (Non-zero features per token)")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "L0 Sparsity\nNot Available", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("L0 Sparsity")

    # 5. Loss Components Comparison
    ax5 = axes[1, 1]
    ax5.plot(history_df["epoch"], history_df["reconstruction_loss"], "g-", linewidth=2, label="Reconstruction")
    ax5.plot(history_df["epoch"], history_df["sparsity_loss"], "orange", linewidth=2, label="Sparsity")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Loss")
    ax5.set_title("Loss Components")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Final Loss Comparison
    ax6 = axes[1, 2]
    final_epoch = history_df.iloc[-1]
    loss_types = ["reconstruction_loss", "sparsity_loss"]
    loss_values = [final_epoch["reconstruction_loss"], final_epoch["sparsity_loss"]]
    colors = ["green", "orange"]

    bars = ax6.bar(loss_types, loss_values, color=colors, alpha=0.7)
    ax6.set_ylabel("Loss Value")
    ax6.set_title("Final Loss Breakdown")
    ax6.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, loss_values, strict=False):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{value:.4f}", ha="center", va="bottom")

    plt.tight_layout()

    # Save plot
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Training visualization saved to: {save_path}")


def save_model(
    sae_model: SAE,
    embedding_dim: int,
    hidden_dim: int,
    config: dict,
    save_dir: str = "outputs"
):
    """
    Save the trained model and configuration.

    Args:
        sae_model: Trained SAE model
        embedding_dim: Input embedding dimension
        hidden_dim: Hidden dimension
        config: Configuration dictionary
        save_dir: Directory to save the model
    """
    print("💾 Saving model and configuration...")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = save_path / "sae_model.pth"
    torch.save({
        "model_state_dict": sae_model.state_dict(),
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "config": config
    }, model_path)

    # Save configuration
    config_path = save_path / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Configuration saved to: {config_path}")


def run_complete_pipeline_example():
    """
    Example of running the complete SAE pipeline using individual components.
    This function demonstrates how to use each component separately.
    """
    print("🚀 SAE Pipeline Components Example")
    print("=" * 60)

    # Configuration
    config = {
        "refseq_file": "../data/vertebrate_mammalian.1.rna.gbff",
        "max_samples": 500,
        "hidden_dim": 1000,
        "epochs": 30,
        "batch_size": 8,
        "sparsity_weight": 0.01,
        "learning_rate": 0.001,
        "layer_name": "final",
        "model_name": "helicalmRNA",
        "device": "cuda",
        "max_length": 1024,
        "normalize_embeddings": False
    }

    print("🔧 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    # Step 1: Setup embedding generator
    embedding_generator = setup_embedding_generator(
        model_name=config["model_name"],
        device=config["device"],
        max_length=config["max_length"],
        batch_size=config["batch_size"],
        normalize_embeddings=config["normalize_embeddings"]
    )

    # Step 2: Prepare data
    train_loader, val_loader, embedding_dim = prepare_data(
        embedding_generator=embedding_generator,
        refseq_file=config["refseq_file"],
        max_samples=config["max_samples"],
        batch_size=config["batch_size"],
        layer_name=config["layer_name"]
    )

    # Step 3: Setup SAE model
    sae_model = setup_sae_model(
        embedding_dim=embedding_dim,
        hidden_dim=config["hidden_dim"],
        sparsity_weight=config["sparsity_weight"]
    )

    # Step 4: Setup trainer
    trainer = setup_trainer(
        sae_model=sae_model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["learning_rate"],
        sparsity_weight=config["sparsity_weight"]
    )

    # Step 5: Train model
    history = train_model(
        trainer=trainer,
        epochs=config["epochs"]
    )

    # Step 6: Plot results
    plot_training_history(history)

    # Step 7: Save model
    save_model(
        sae_model=sae_model,
        embedding_dim=embedding_dim,
        hidden_dim=config["hidden_dim"],
        config=config
    )

    print("\n🎉 Pipeline completed successfully!")
    print("=" * 60)
    print("📊 Final Results:")
    print(f"   Model: {embedding_dim} → {config['hidden_dim']} features")
    print(f"   Training epochs: {len(history)}")
    print(f"   Final reconstruction loss: {history[-1]['reconstruction_loss']:.6f}")
    print(f"   Final sparsity loss: {history[-1]['sparsity_loss']:.6f}")
    if "val_l0_sparsity" in history[-1]:
        print(f"   Final L0 sparsity: {history[-1]['val_l0_sparsity']:.1f} features per token")
    print(f"   Model parameters: {sum(p.numel() for p in sae_model.parameters()):,}")

    return {
        "embedding_generator": embedding_generator,
        "sae_model": sae_model,
        "trainer": trainer,
        "history": history,
        "config": config,
        "embedding_dim": embedding_dim
    }


if __name__ == "__main__":
    # Run the complete pipeline example
    results = run_complete_pipeline_example()
