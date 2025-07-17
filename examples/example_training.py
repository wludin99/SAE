"""
Example training script for SAE
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from ..models import SAE
from ..training import SAETrainer, TrainingConfig
from ..training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


def create_dummy_data(n_samples=1000, input_size=1000, train_split=0.8):
    """Create dummy data for demonstration"""
    # Generate random data
    X = np.random.randn(n_samples, input_size)
    
    # Split into train/val
    split_idx = int(n_samples * train_split)
    X_train, X_val = X[:split_idx], X[split_idx:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    
    return X_train, X_val


def main():
    """Main training function"""
    
    # Create dummy data
    print("Creating dummy data...")
    X_train, X_val = create_dummy_data(n_samples=1000, input_size=1000)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, X_train)  # Autoencoder: input = target
    val_dataset = TensorDataset(X_val, X_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("Creating SAE model...")
    model = SAE(
        input_size=1000,
        hidden_size=100,
        sparsity_weight=0.01,
        sparsity_target=0.05
    )
    
    # Create training configuration
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        optimizer='adam',
        weight_decay=1e-5,
        sparsity_weight=0.01,
        sparsity_target=0.05,
        log_dir='./logs',
        save_dir='./checkpoints'
    )
    
    # Create callbacks
    callbacks = [
        EarlyStopping(patience=10, monitor='val_total_loss'),
        ModelCheckpoint(
            filepath='./checkpoints/best_model.pth',
            monitor='val_total_loss',
            save_best_only=True
        ),
        LearningRateScheduler(
            scheduler_type='reduce_on_plateau',
            monitor='val_total_loss',
            patience=5,
            factor=0.5
        )
    ]
    
    # Create trainer
    trainer = SAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        callbacks=callbacks
    )
    
    # Train the model
    print("Starting training...")
    history = trainer.train(epochs=config.epochs)
    
    # Print summary
    trainer.logger.print_summary()
    
    # Plot training curves
    trainer.logger.plot_training_curves(save_path='./logs/training_curves.png')
    
    # Save final model
    trainer.save_model('./checkpoints/final_model.pth')
    
    print("Training completed!")


if __name__ == "__main__":
    main() 