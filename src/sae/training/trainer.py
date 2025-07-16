import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Callable
import os
import json
from datetime import datetime

from sae.models import SAE
from sae.losses import SAELoss
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
                 callbacks: Optional[List[Callback]] = None):
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
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Setup logger
        self.logger = TrainingLogger(self.config.log_dir)
        
        # Register callbacks
        for callback in self.callbacks:
            callback.set_trainer(self)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on config"""
        if self.config.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'reconstruction_loss': [],
            'sparsity_loss': [],
            'total_loss': []
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
            'total_loss': np.mean(epoch_metrics['total_loss']),
            'reconstruction_loss': np.mean(epoch_metrics['reconstruction_loss']),
            'sparsity_loss': np.mean(epoch_metrics['sparsity_loss'])
        }
        
        return epoch_avg
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = []
        val_metrics = {
            'reconstruction_loss': [],
            'sparsity_loss': [],
            'total_loss': []
        }
        
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
        
        val_avg = {
            'total_loss': np.mean(val_metrics['total_loss']),
            'reconstruction_loss': np.mean(val_metrics['reconstruction_loss']),
            'sparsity_loss': np.mean(val_metrics['sparsity_loss'])
        }
        
        return val_avg
    
    def train(self, epochs: int) -> Dict[str, List[float]]:
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
            metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
            self.training_history.append(metrics)
            self.logger.log_metrics(epoch, metrics)
            
            # Print progress
            self._print_epoch_summary(epoch, metrics)
            
            # Call callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, metrics)
            
            # Early stopping check
            if val_metrics and val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
        
        print("Training completed!")
        return self.training_history
    
    def _print_epoch_summary(self, epoch: int, metrics: Dict[str, float]):
        """Print epoch summary"""
        train_loss = metrics.get('total_loss', 0)
        val_loss = metrics.get('val_total_loss', 0)
        
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}")
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'current_epoch': self.current_epoch
        }, path)
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.current_epoch = checkpoint.get('current_epoch', 0) 