# SAE
Sparse Autoencoders for Interpretability of mRNA Models

Trains a sparse autoencoder on various layers of Helical's Helix-mRNA model to uncover biologically relevant factors

## Installation

### Quick Setup (Recommended)
```bash
# Clone repo
git clone git@github.com:wludin99/SAE.git
cd SAE


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

### Loading Genomic Datasets

The project includes utilities for loading various genomic datasets from Hugging Face. We provide several smaller, manageable datasets perfect for development and testing.

#### Available Datasets

```python
from sae.data.genomic_datasets import (
    list_available_datasets,
    load_genomic_dataset,
    load_human_dna,
    load_drosophila_dna,
    load_small_proteins,
    load_dna_promoters
)

# List all available datasets
datasets = list_available_datasets()
print(datasets)

# Load human DNA dataset (50K samples, ~50MB)
human_dataloader = load_human_dna(max_samples=1000, batch_size=32)

# Load drosophila DNA dataset (50K samples, ~50MB)  
drosophila_dataloader = load_drosophila_dna(max_samples=1000, batch_size=32)

# Load small proteins dataset (10K samples, ~5MB)
proteins_dataloader = load_small_proteins(max_samples=1000, batch_size=32)

# Load DNA promoters dataset (10K samples, ~10MB)
promoters_dataloader = load_dna_promoters(max_samples=1000, batch_size=32)
```

**Available Datasets:**
- **human_dna**: Human DNA sequences (~50K samples, ~50MB)
- **drosophila_dna**: Drosophila melanogaster DNA (~50K samples, ~50MB)  
- **yeast_dna**: Yeast DNA sequences (~50K samples, ~50MB)
- **small_proteins**: Protein sequences subset (~10K samples, ~5MB)
- **dna_promoters**: DNA promoter sequences (~10K samples, ~10MB)
- **genomic_sequences**: Various genomic sequences (~100K samples, ~100MB)

#### Example Script
Run the example script to see all loading options:
```bash
poetry run python examples/load_genomic_datasets.py
```

### Sequence Preprocessing Module

The project includes a comprehensive preprocessing module for sequence foundation models, with special support for Helical and codon-based preprocessing.

#### Helical Model Wrapper

```python
from sae.preprocessing import HelicalWrapper, create_helical_wrapper, CodonPreprocessor

# Create Helical wrapper with codon preprocessing
wrapper = create_helical_wrapper(
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=32,
    codon_start_token="E",  # Each codon starts with 'E'
    add_codon_start=True,
    normalize_embeddings=False
)

# Generate embeddings from sequences
sequences = ["ATGCGTACGTACGT", "GCTAGCTAGCTAGC"]
embeddings = wrapper(sequences)
print(f"Generated embeddings: {embeddings.shape}")

# Get codon statistics
stats = wrapper.get_codon_statistics(sequences)
print(f"Total codons: {stats['total_codons']}")
```

#### Codon Preprocessing

```python
from sae.preprocessing import CodonPreprocessor

# Create codon preprocessor
preprocessor = CodonPreprocessor(start_token="E")

# Process sequences
sequences = ["ATGCGTACGTACGT", "GCTAGCTAGCTAGC"]
processed = preprocessor.process_sequences(sequences)

# Original: "ATGCGTACGTACGT"
# Processed: "EATGEACGEACGEACG"
print(f"Original: {sequences[0]}")
print(f"Processed: {processed[0]}")

# Get codon statistics
stats = preprocessor.get_codon_statistics(sequences)
print(f"Unique codons: {stats['unique_codons']}")
```

#### Example Script
```bash
poetry run python examples/preprocessing_example.py
```

### Complete SAE Pipeline

The project provides a complete pipeline for training SAE models on HelicalmRNA embeddings:

#### Quick Start
```python
from sae import run_complete_pipeline

# Run complete pipeline (embeddings -> SAE training -> feature extraction)
pipeline = run_complete_pipeline(
    dataset_name="human_dna",
    max_samples=1000,
    embedding_dim=768,
    hidden_dim=50,
    epochs=50,
    batch_size=32
)

# Extract features from new embeddings
features = pipeline.extract_features(new_embeddings)
```

#### Step-by-Step Pipeline
```python
from sae import SAETrainingPipeline, EmbeddingGenerator

# 1. Setup pipeline
pipeline = SAETrainingPipeline(
    embedding_dim=768,
    hidden_dim=50,
    sparsity_weight=0.1
)

# 2. Setup components
pipeline.setup_embedding_generator()
pipeline.setup_sae_model()

# 3. Prepare data
train_loader, val_loader = pipeline.prepare_data(
    dataset_name="human_dna",
    max_samples=1000
)

# 4. Train SAE
history = pipeline.train(train_loader, val_loader, epochs=50)

# 5. Analyze results
pipeline.plot_training_history("training_history.png")
```

#### Manual Embedding Generation
```python
from sae import EmbeddingGenerator

# Generate embeddings from genomic sequences
generator = EmbeddingGenerator()
result = generator.generate_embeddings_from_dataset(
    dataset_name="human_dna",
    max_samples=1000,
    layer_idx=None  # Use last layer, or specify layer number
)

embeddings = result['embeddings']
print(f"Generated embeddings: {embeddings.shape}")
```

#### Example Scripts
```bash
# Run complete pipeline with analysis
poetry run python examples/complete_sae_pipeline.py

# Quick test (small dataset)
poetry run python examples/complete_sae_pipeline.py --quick

# Test genomic datasets
poetry run python examples/load_genomic_datasets.py
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