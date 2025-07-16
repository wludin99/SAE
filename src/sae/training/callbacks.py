import torch
import os
from typing import Dict, Optional
from abc import ABC, abstractmethod


class Callback(ABC):
    """Base callback class"""
    
    def __init__(self):
        self.trainer = None
    
    def set_trainer(self, trainer):
        """Set the trainer reference"""
        self.trainer = trainer
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int):
        """Called at the beginning of each epoch"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch"""
        pass
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, float]):
        """Called at the end of each batch"""
        pass


class EarlyStopping(Callback):
    """Early stopping callback"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, monitor: str = 'val_total_loss'):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_begin(self, epoch: int):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        if self.monitor not in metrics:
            return
        
        current_score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            self.should_stop = True


class ModelCheckpoint(Callback):
    """Model checkpointing callback"""
    
    def __init__(self, 
                 filepath: str,
                 monitor: str = 'val_total_loss',
                 save_best_only: bool = True,
                 save_weights_only: bool = False):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.best_score = None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def on_epoch_begin(self, epoch: int):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        if self.monitor not in metrics:
            return
        
        current_score = metrics[self.monitor]
        
        if self.best_score is None or current_score < self.best_score:
            self.best_score = current_score
            
            if self.save_best_only:
                # Save only the best model
                if self.save_weights_only:
                    torch.save(self.trainer.model.state_dict(), self.filepath)
                else:
                    self.trainer.save_model(self.filepath)
                print(f"Saved best model with {self.monitor}: {current_score:.6f}")
        elif not self.save_best_only:
            # Save every epoch
            epoch_filepath = self.filepath.replace('.pth', f'_epoch_{epoch}.pth')
            if self.save_weights_only:
                torch.save(self.trainer.model.state_dict(), epoch_filepath)
            else:
                self.trainer.save_model(epoch_filepath)


class LearningRateScheduler(Callback):
    """Learning rate scheduling callback"""
    
    def __init__(self, scheduler_type: str = 'step', **kwargs):
        super().__init__()
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = kwargs
        self.scheduler = None
    
    def on_epoch_begin(self, epoch: int):
        if self.scheduler is None:
            self._setup_scheduler()
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        if self.scheduler is not None:
            if self.scheduler_type == 'reduce_on_plateau':
                # For ReduceLROnPlateau, we need a metric to monitor
                monitor = self.scheduler_kwargs.get('monitor', 'val_total_loss')
                if monitor in metrics:
                    self.scheduler.step(metrics[monitor])
            else:
                self.scheduler.step()
    
    def _setup_scheduler(self):
        """Setup the learning rate scheduler"""
        if self.scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.trainer.optimizer,
                step_size=self.scheduler_kwargs.get('step_size', 30),
                gamma=self.scheduler_kwargs.get('gamma', 0.1)
            )
        elif self.scheduler_type == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.trainer.optimizer,
                gamma=self.scheduler_kwargs.get('gamma', 0.95)
            )
        elif self.scheduler_type == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.trainer.optimizer,
                mode='min',
                factor=self.scheduler_kwargs.get('factor', 0.1),
                patience=self.scheduler_kwargs.get('patience', 10),
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")


class ProgressLogger(Callback):
    """Progress logging callback"""
    
    def __init__(self, log_interval: int = 10):
        super().__init__()
        self.log_interval = log_interval
    
    def on_epoch_begin(self, epoch: int):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        if epoch % self.log_interval == 0:
            print(f"Epoch {epoch}: {metrics}")
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, float]):
        if batch_idx % self.log_interval == 0:
            print(f"Batch {batch_idx}: {metrics}") 