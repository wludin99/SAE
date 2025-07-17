import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""

    # Model parameters
    input_size: int = 1000
    hidden_size: int = 100
    sparsity_weight: float = 0.01
    sparsity_target: float = 0.05

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 4
    epochs: int = 100
    optimizer: str = "adam"  # 'adam' or 'sgd'
    weight_decay: float = 0.0
    momentum: float = 0.9  # for SGD
    gradient_clip: Optional[float] = None

    # Device
    device: Optional[str] = None  # 'cuda', 'cpu', or None for auto-detect

    # Logging
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"

    def __post_init__(self):
        """Auto-detect device if not specified"""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def save(self, path: str):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from file"""
        with open(path) as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class TrainingLogger:
    """Logger for training metrics and visualization"""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Initialize log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.json")
        self.metrics_history = []

    def log_metrics(self, epoch: int, metrics: dict[str, float]):
        """Log metrics for an epoch"""
        log_entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }

        self.metrics_history.append(log_entry)

        # Save to file
        with open(self.log_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves"""
        if not self.metrics_history:
            print("No metrics to plot")
            return

        epochs = [entry["epoch"] for entry in self.metrics_history]

        # Get all metric names (excluding epoch and timestamp)
        metric_names = set()
        for entry in self.metrics_history:
            metric_names.update([k for k in entry.keys() if k not in ["epoch", "timestamp"]])

        # Create subplots
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(sorted(metric_names)):
            values = [entry.get(metric, 0) for entry in self.metrics_history]
            axes[i].plot(epochs, values, label=metric)
            axes[i].set_title(metric)
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel("Value")
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def get_best_epoch(self, metric: str = "val_total_loss") -> Optional[int]:
        """Get the epoch with the best value for a given metric"""
        if not self.metrics_history:
            return None

        best_epoch = None
        best_value = float("inf")

        for entry in self.metrics_history:
            if metric in entry:
                value = entry[metric]
                if value < best_value:
                    best_value = value
                    best_epoch = entry["epoch"]

        return best_epoch

    def print_summary(self):
        """Print training summary"""
        if not self.metrics_history:
            print("No training history available")
            return

        print("Training Summary:")
        print(f"Total epochs: {len(self.metrics_history)}")

        # Get final metrics
        final_metrics = self.metrics_history[-1]
        print("\nFinal metrics:")
        for key, value in final_metrics.items():
            if key not in ["epoch", "timestamp"]:
                print(f"  {key}: {value:.6f}")

        # Get best epoch
        best_epoch = self.get_best_epoch()
        if best_epoch is not None:
            print(f"\nBest validation loss at epoch: {best_epoch}")


def create_training_config(**kwargs) -> TrainingConfig:
    """Helper function to create training configuration with custom parameters"""
    return TrainingConfig(**kwargs)


def setup_device(device: Optional[str] = None) -> torch.device:
    """Setup and return the appropriate device"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    return torch.device(device)
