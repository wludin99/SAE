"""
Sparse Autoencoders for Interpretability of mRNA Models

A package for training sparse autoencoders on various layers of Helical's Helix-mRNA model
to uncover biologically relevant factors.
"""

__version__ = "0.1.0"
__author__ = "William Ludington"
__email__ = "whfludington@gmail.com"

# Import main components for easy access
from .models import SAE
from .losses import SAELoss
from .training import SAETrainer, TrainingConfig
from .training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

__all__ = [
    'SAE',
    'SAELoss', 
    'SAETrainer',
    'TrainingConfig',
    'EarlyStopping',
    'ModelCheckpoint', 
    'LearningRateScheduler'
] 