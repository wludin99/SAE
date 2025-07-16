# SAE
Sparse Autoencoders for Interpretability of mRNA Models

Trains a sparse autoencoder on various layers of Helical's Helix-mRNA model to uncover biologically relevant factors

## Installation

### Quick Setup (Recommended)
```bash
# Clone repo
git clone git@github.com:wludin99/SAE.git
cd SAE

# Complete environment setup (includes CUDA toolkit, dependencies, and vortex)
make setup
```

### Manual Setup
```bash
# Clone repo
git clone git@github.com:wludin99/SAE.git
cd SAE

# Install CUDA toolkit (if using conda)
conda install cuda-toolkit=12.4 -c nvidia

# Set environment variables
export CUDNN_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvtx/include

# Install Python dependencies
poetry install

# Setup vortex (cloned in parent directory)
make setup-vortex
```

### Alternative Setup Scripts
```bash
# Using bash script
chmod +x setup_env.sh
./setup_env.sh

# Using Python script
python setup_env.py
```

### Project Structure
After setup, your directory structure will look like:
```
sae-project/
├── SAE/                    # Your main SAE project
│   ├── src/
│   ├── pyproject.toml
│   └── ...
├── vortex/                 # Cloned in parent directory
│   ├── ...
│   └── ...
└── ...
```

## Development Setup

### Install Development Dependencies
```bash
# Install all dependencies including dev tools
poetry install

# Setup pre-commit hooks
make setup-dev
```

### Code Quality Tools

This project uses **Ruff** for linting and formatting, along with pre-commit hooks for automatic code quality checks.

#### Available Commands
```bash
# Lint code
make lint

# Format code
make format

# Run all checks (lint + format)
make check

# Clean cache files
make clean

# Verify installation
make verify

# Show all available commands
make help
```

#### Manual Ruff Commands
```bash
# Check for issues
poetry run ruff check .

# Fix auto-fixable issues
poetry run ruff check --fix .

# Format code
poetry run ruff format .

# Show what rules are being violated
poetry run ruff check --statistics .
```

### Pre-commit Hooks

The project includes pre-commit hooks that automatically:
- Run Ruff linter and formatter
- Check for trailing whitespace
- Ensure files end with newline
- Validate YAML files
- Check for large files
- Detect merge conflicts

These run automatically on every commit, but you can also run them manually:
```bash
poetry run pre-commit run --all-files
```

## Usage

### Activate Environment
```bash
# Activate Poetry shell
poetry shell

# Or run commands directly
poetry run python your_script.py
```

### Training Example
```python
from sae import SAE, SAETrainer, TrainingConfig
from sae.training.callbacks import EarlyStopping, ModelCheckpoint

# Create model and train
model = SAE(input_size=1000, hidden_size=100)
config = TrainingConfig(epochs=100, learning_rate=0.001)

# Setup callbacks
callbacks = [
    EarlyStopping(patience=10),
    ModelCheckpoint(filepath='./checkpoints/best_model.pth')
]

# Create trainer and train
trainer = SAETrainer(model, train_loader, val_loader, config, callbacks)
history = trainer.train(epochs=100)
```

## Dependencies

- **Python**: 3.11+
- **PyTorch**: 2.6.0
- **Helical**: Latest from GitHub
- **CUDA Toolkit**: 12.4 (optional, for GPU support)
- **Vortex**: Cloned from GitHub (located in parent directory)

## Environment Variables

If using conda for CUDA toolkit, add these to your shell profile:
```bash
export CUDNN_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvtx/include
```