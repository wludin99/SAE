from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from sae.losses import SAELoss
from sae.models import SAE
from sae.training.callbacks import Callback
from sae.training.utils import TrainingConfig, TrainingLogger


class SAETrainer:
    """
    Trainer class for Sparse Autoencoder training
    """

    def __init__(self,
                 model: SAE,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 config: Optional[TrainingConfig] = None,
                 callbacks: Optional[list[Callback]] = None):
        """
        Initialize the trainer

        Args:
            model: SAE model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            callbacks: List of callbacks for monitoring and control
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.callbacks = callbacks or []

        # Setup optimizer and loss function
        self.optimizer = self._setup_optimizer()
        self.loss_fn = SAELoss(
            sparsity_weight=self.config.sparsity_weight,
            sparsity_target=self.config.sparsity_target
        )

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float("inf")
        self.training_history = []

        # Setup logger
        self.logger = TrainingLogger(self.config.log_dir)

        # Register callbacks
        for callback in self.callbacks:
            callback.set_trainer(self)

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on config"""
        if self.config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            "reconstruction_loss": [],
            "sparsity_loss": [],
            "total_loss": []
        }

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.config.device:
                data = data.to(self.config.device)
                target = target.to(self.config.device)

            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, encoded = self.model(data)

            # Compute loss
            loss, loss_dict = self.loss_fn(reconstructed, target, encoded)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()

            # Record metrics
            epoch_losses.append(loss.item())
            for key, value in loss_dict.items():
                epoch_metrics[key].append(value)

            self.current_step += 1

            # Call callbacks
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, loss_dict)

        # Compute epoch averages
        epoch_avg = {
            "total_loss": float(np.mean(epoch_metrics["total_loss"])),
            "reconstruction_loss": float(np.mean(epoch_metrics["reconstruction_loss"])),
            "sparsity_loss": float(np.mean(epoch_metrics["sparsity_loss"]))
        }

        return epoch_avg

    def validate(self) -> dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = []
        val_metrics = {
            "reconstruction_loss": [],
            "sparsity_loss": [],
            "total_loss": []
        }
        
        # Track L0 sparsity (number of non-zero activations)
        l0_counts = []
        sparsity_percentages = []

        with torch.no_grad():
            for data, target in self.val_loader:
                if self.config.device:
                    data = data.to(self.config.device)
                    target = target.to(self.config.device)

                reconstructed, encoded = self.model(data)
                loss, loss_dict = self.loss_fn(reconstructed, target, encoded)

                val_losses.append(loss.item())
                for key, value in loss_dict.items():
                    val_metrics[key].append(value)
                
                # Calculate L0 sparsity (number of non-zero features per hidden dimension)
                # encoded shape: (batch_size, hidden_dim) or (batch_size, seq_len, hidden_dim)
                if len(encoded.shape) == 3:
                    # Handle 3D case: (batch_size, seq_len, hidden_dim)
                    batch_size, seq_len, hidden_dim = encoded.shape
                    # Count non-zero features per token position
                    non_zero_per_token = (encoded != 0).sum(dim=2).float()  # (batch_size, seq_len)
                    # Average across all tokens and samples
                    avg_non_zero_per_token = non_zero_per_token.mean() 
                    sparsity_percentage = avg_non_zero_per_token.cpu().numpy()
                else:
                    # Handle 2D case: (batch_size, hidden_dim)
                    batch_size, hidden_dim = encoded.shape
                    # Count non-zero activations per sample
                    avg_non_zero_per_token = (encoded != 0).sum(dim=1).float().mean()  # Average across samples
                    # Calculate sparsity percentage
                    sparsity_percentage = avg_non_zero_per_token.cpu().numpy()
                
                l0_counts.extend([avg_non_zero_per_token.cpu().numpy()])
                sparsity_percentages.extend([sparsity_percentage])

        val_avg = {
            "total_loss": float(np.mean(val_metrics["total_loss"])),
            "reconstruction_loss": float(np.mean(val_metrics["reconstruction_loss"])),
            "sparsity_loss": float(np.mean(val_metrics["sparsity_loss"])),
            "l0_sparsity": float(np.mean(l0_counts)),  # Average number of non-zero activations
            "sparsity_percentage": float(np.mean(sparsity_percentages))  # Average sparsity percentage
        }

        return val_avg

    def train(self, epochs: int) -> dict[str, list[float]]:
        """Train the model for specified number of epochs"""
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Call callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate()

            # Log metrics
            metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self.training_history.append(metrics)
            self.logger.log_metrics(epoch, metrics)

            # Print progress
            self._print_epoch_summary(epoch, metrics)

            # Call callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, metrics)

            # Early stopping check
            if val_metrics and val_metrics["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total_loss"]

        print("Training completed!")
        return self.training_history

    def _print_epoch_summary(self, epoch: int, metrics: dict[str, float]):
        """Print epoch summary"""
        train_loss = metrics.get("total_loss", 0)
        val_loss = metrics.get("val_total_loss", 0)
        val_l0 = metrics.get("val_l0_sparsity", 0)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val L0: {val_l0:.1f}")

    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "training_history": self.training_history,
            "best_val_loss": self.best_val_loss,
            "current_epoch": self.current_epoch
        }, path)

    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history = checkpoint.get("training_history", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.current_epoch = checkpoint.get("current_epoch", 0)
