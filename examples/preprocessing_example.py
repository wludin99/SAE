"""
Example script demonstrating the preprocessing module with Helical model.

This script shows how to use the HelicalWrapper with codon preprocessing
to generate embeddings from DNA/RNA sequences.
"""

import torch
import logging
from typing import List

from sae.preprocessing import HelicalWrapper, create_helical_wrapper, CodonPreprocessor
from sae.data import load_small_genomic_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_codon_preprocessing():
    """Demonstrate codon preprocessing functionality."""
    print("=== Codon Preprocessing Demo ===")
    
    # Sample DNA sequences
    sequences = [
        "ATGCGTACGTACGT",
        "GCTAGCTAGCTAGC",
        "TACGTACGTACGTA"
    ]
    
    # Create codon preprocessor
    preprocessor = CodonPreprocessor(start_token="E")
    
    print("Original sequences:")
    for i, seq in enumerate(sequences):
        print(f"  {i}: {seq}")
    
    print("\nCodon statistics:")
    stats = preprocessor.get_codon_statistics(sequences)
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Total codons: {stats['total_codons']}")
    print(f"  Unique codons: {stats['unique_codons']}")
    print(f"  Average sequence length: {stats['average_sequence_length']:.1f}")
    
    print("\nProcessed sequences with codon start tokens:")
    processed = preprocessor.process_sequences(sequences)
    for i, seq in enumerate(processed):
        print(f"  {i}: {seq}")
    
    print("\nCodon breakdown for first sequence:")
    codons = preprocessor.split_into_codons(sequences[0])
    print(f"  Original: {sequences[0]}")
    print(f"  Codons: {codons}")
    print(f"  With start tokens: {processed[0]}")


def demonstrate_helical_wrapper():
    """Demonstrate Helical wrapper functionality."""
    print("\n=== Helical Wrapper Demo ===")
    
    # Load a small dataset
    try:
        dataset = load_small_genomic_dataset("human_dna", max_samples=10)
        sequences = dataset["sequences"]
        print(f"Loaded {len(sequences)} sequences from human_dna dataset")
    except Exception as e:
        logger.warning(f"Could not load dataset: {e}")
        # Use sample sequences instead
        sequences = [
            "ATGCGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC",
            "TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA"
        ]
        print(f"Using {len(sequences)} sample sequences")
    
    # Create Helical wrapper
    try:
        wrapper = create_helical_wrapper(
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=2,
            codon_start_token="E",
            add_codon_start=True,
            normalize_embeddings=False
        )
        
        print(f"Created Helical wrapper: {wrapper.get_model_info()}")
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        embeddings = wrapper(sequences)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        print(f"Embedding dtype: {embeddings.dtype}")
        print(f"Embedding device: {embeddings.device}")
        
        # Show some statistics
        print(f"\nEmbedding statistics:")
        print(f"  Mean: {embeddings.mean().item():.4f}")
        print(f"  Std: {embeddings.std().item():.4f}")
        print(f"  Min: {embeddings.min().item():.4f}")
        print(f"  Max: {embeddings.max().item():.4f}")
        
        # Get codon statistics
        print(f"\nCodon statistics:")
        codon_stats = wrapper.get_codon_statistics(sequences)
        print(f"  Total codons: {codon_stats['total_codons']}")
        print(f"  Unique codons: {codon_stats['unique_codons']}")
        
        # Show top codons
        top_codons = sorted(codon_stats['codon_frequencies'].items(), 
                          key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 codons: {top_codons}")
        
    except Exception as e:
        logger.error(f"Helical wrapper demo failed: {e}")
        print(f"Error: {e}")


def demonstrate_preprocessing_only():
    """Demonstrate preprocessing without embedding generation."""
    print("\n=== Preprocessing Only Demo ===")
    
    sequences = [
        "ATGCGTACGTACGT",
        "GCTAGCTAGCTAGC"
    ]
    
    try:
        wrapper = create_helical_wrapper(
            device="cpu",  # Use CPU for preprocessing only
            codon_start_token="E",
            add_codon_start=True
        )
        
        # Preprocess only
        processed = wrapper.preprocess_only(sequences)
        print("Original sequences:")
        for i, seq in enumerate(sequences):
            print(f"  {i}: {seq}")
        
        print("\nPreprocessed sequences:")
        for i, seq in enumerate(processed):
            print(f"  {i}: {seq}")
            
    except Exception as e:
        logger.error(f"Preprocessing only demo failed: {e}")
        print(f"Error: {e}")


def main():
    """Run all demonstrations."""
    print("SAE Preprocessing Module Demo")
    print("=" * 50)
    
    # Demonstrate codon preprocessing
    demonstrate_codon_preprocessing()
    
    # Demonstrate Helical wrapper
    demonstrate_helical_wrapper()
    
    # Demonstrate preprocessing only
    demonstrate_preprocessing_only()
    
    print("\n" + "=" * 50)
    print("Demo completed!")


if __name__ == "__main__":
    main() 