"""
SAE - Sparse Autoencoders for Interpretability of mRNA Models

A package for training sparse autoencoders on genomic embeddings to uncover
biologically relevant features from pre-trained language models.
"""

from .models.sae import SAE
from .losses import SAELoss
from .data import (
    RefSeqDataset,
    load_refseq_dataset,
    create_refseq_dataloader,
    analyze_refseq_file,
    print_refseq_analysis
)
from .pipeline import (
    EmbeddingGenerator,
    SAETrainingPipeline,
    run_complete_pipeline
)
from .preprocessing import (
    RefSeqWrapper,
    create_refseq_wrapper,
    RefSeqPreprocessor,
    CodonPreprocessor,
    SequenceModelWrapper,
    PreprocessingConfig
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "SAE",
    "SAELoss",
    
    # Data loading (RefSeq only)
    "RefSeqDataset",
    "load_refseq_dataset",
    "create_refseq_dataloader",
    "analyze_refseq_file",
    "print_refseq_analysis",
    
    # Pipeline
    "EmbeddingGenerator",
    "SAETrainingPipeline", 
    "run_complete_pipeline",
    
    # Preprocessing
    "RefSeqWrapper",
    "create_refseq_wrapper",
    "RefSeqPreprocessor",
    "CodonPreprocessor",
    "SequenceModelWrapper",
    "PreprocessingConfig"
] 