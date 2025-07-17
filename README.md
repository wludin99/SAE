# SAE: Sparse Autoencoders for mRNA Model Interpretability

Trains sparse autoencoders on HelicalmRNA model embeddings to uncover biologically relevant factors in mRNA sequences.

## Installation

This project uses Poetry for dependency management within a Conda environment.

### Prerequisites

1. **Conda Environment**: Create and activate a conda environment
```bash
conda create -n helical python=3.11
conda activate helical
```

2. **CUDA Toolkit** (optional, for GPU support):
```bash
conda install cuda-toolkit=12.4 -c nvidia
```

### Project Setup

```bash
# Clone the repository
git clone <repository-url>
cd SAE

# Install dependencies with Poetry
poetry install

# Activate the Poetry shell
poetry shell
```

### Development Setup

```bash
# Install development dependencies
poetry install

# Setup pre-commit hooks
make setup-dev
```

## Data

This project is designed to work with **RefSeq GenBank files** (.gbff format). The pipeline extracts mRNA sequences and generates embeddings using the HelicalmRNA model.

### Data Requirements

- **Format**: RefSeq GenBank files (.gbff)
- **Location**: Place your RefSeq files in a `data/` directory (relative to the project root)
- **Example**: `../data/vertebrate_mammalian.1.rna.gbff`

### Downloading RefSeq Data

RefSeq files can be downloaded from NCBI:
```bash
# Example: Download vertebrate mammalian RNA
wget https://ftp.ncbi.nlm.nih.gov/refseq/release/vertebrate_mammalian/vertebrate_mammalian.1.rna.gbff.gz
gunzip vertebrate_mammalian.1.rna.gbff.gz
```

## Usage

### Quick Start

```python
from sae.pipeline import run_complete_pipeline

# Run complete pipeline with RefSeq data
pipeline = run_complete_pipeline(
    refseq_file="../data/vertebrate_mammalian.1.rna.gbff",
    max_samples=1000,
    hidden_dim=1000,
    epochs=50,
    apply_sequence_pooling=False  # Use token-level embeddings
)
```

### Step-by-Step Pipeline

```python
from sae.pipeline import SAETrainingPipeline

# 1. Initialize pipeline
pipeline = SAETrainingPipeline(
    embedding_dim=None,  # Will be auto-detected
    hidden_dim=1000,
    sparsity_weight=0.1
)

# 2. Setup components
pipeline.setup_embedding_generator()
pipeline.setup_sae_model()

# 3. Prepare data
train_loader, val_loader = pipeline.prepare_data(
    refseq_file="../data/vertebrate_mammalian.1.rna.gbff",
    max_samples=1000,
    apply_sequence_pooling=False  # Token-level embeddings
)

# 4. Train SAE
history = pipeline.train(epochs=50)

# 5. Extract features
features = pipeline.extract_features(embeddings)
```

### Token-Level vs Sequence-Level Embeddings

The pipeline supports two modes:

**Token-Level Embeddings** (default):
```python
# Preserves individual token information
pipeline.prepare_data(
    refseq_file="your_file.gbff",
    apply_sequence_pooling=False  # 3D: (batch, seq_len, embed_dim)
)
```

**Sequence-Level Embeddings**:
```python
# Pools tokens to sequence-level representations
pipeline.prepare_data(
    refseq_file="your_file.gbff",
    apply_sequence_pooling=True  # 2D: (batch, embed_dim)
)
```

### Example Scripts

```bash
# RefSeq pipeline example
poetry run python examples/refseq_pipeline_example.py

# Sequence length filtering
poetry run python examples/sequence_length_filter_example.py

# Clean architecture example
poetry run python examples/clean_architecture_example.py
```

## Development

### Available Commands

```bash
# Code quality
make lint          # Run Ruff linter
make format        # Format code with Ruff
make check         # Run linting and formatting
make clean         # Clean cache files

# Testing
make test          # Run tests
make test-preprocessing  # Test preprocessing module

# Setup
make install       # Install dependencies
make setup-dev     # Setup development environment
make verify        # Verify installation
make help          # Show all commands
```

### Code Quality

This project uses **Ruff** for linting and formatting. Pre-commit hooks automatically run quality checks on commit.

```bash
# Manual quality checks
poetry run ruff check .
poetry run ruff format .
```

## Project Structure

```
SAE/
├── src/sae/                    # Main package
│   ├── data/                   # Data loading and parsing
│   ├── models/                 # SAE model implementation
│   ├── pipeline/               # Training pipeline
│   ├── preprocessing/          # Sequence preprocessing
│   └── training/               # Training utilities
├── examples/                   # Usage examples
├── tests/                      # Test files
├── outputs/                    # Generated outputs
├── pyproject.toml             # Poetry configuration
└── Makefile                   # Development commands
```

## Dependencies

- **Python**: 3.11+
- **PyTorch**: 2.6.0
- **Helical**: Latest from GitHub
- **Biopython**: For RefSeq parsing
- **CUDA Toolkit**: 12.4 (optional, for GPU support)

## Environment Variables

If using conda for CUDA toolkit, add these to your shell profile:
```bash
export CUDNN_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvtx/include
```
