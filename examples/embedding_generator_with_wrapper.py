"""
Example script demonstrating the updated EmbeddingGenerator with HelicalWrapper.

This script shows how the EmbeddingGenerator now automatically uses the HelicalWrapper
with codon preprocessing for generating embeddings from genomic sequences.
"""

import torch
import logging
from pathlib import Path

from sae.pipeline import EmbeddingGenerator, generate_embeddings_for_training
from sae.data import load_genomic_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_embedding_generator():
    """Demonstrate the updated EmbeddingGenerator with wrapper."""
    print("=== EmbeddingGenerator with HelicalWrapper Demo ===")
    
    # Load a small dataset
    try:
        dataset = load_genomic_dataset("human_dna", max_samples=10)
        sequences = [item['sequence'] for item in dataset]
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
    
    # Create EmbeddingGenerator (automatically uses HelicalWrapper)
    try:
        generator = EmbeddingGenerator(
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=2,
            codon_start_token="E",
            add_codon_start=True,
            normalize_embeddings=False
        )
        
        print(f"Created EmbeddingGenerator: {generator.get_model_info()}")
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        embeddings = generator.generate_embeddings(sequences)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        print(f"Embedding dtype: {embeddings.dtype}")
        
        # Show some statistics
        print(f"\nEmbedding statistics:")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        print(f"  Min: {embeddings.min():.4f}")
        print(f"  Max: {embeddings.max():.4f}")
        
        # Test preprocessing
        print(f"\nTesting preprocessing...")
        processed = generator.preprocess_sequences(sequences[:2])
        print(f"  Processed tensor shape: {processed.shape}")
        
    except Exception as e:
        logger.error(f"EmbeddingGenerator demo failed: {e}")
        print(f"Error: {e}")


def demonstrate_convenience_function():
    """Demonstrate the convenience function."""
    print("\n=== Convenience Function Demo ===")
    
    try:
        # Use the convenience function
        result = generate_embeddings_for_training(
            dataset_name="human_dna",
            max_samples=5,
            codon_start_token="E",
            add_codon_start=True,
            normalize_embeddings=False
        )
        
        print(f"✅ Generated embeddings: {result['embeddings'].shape}")
        print(f"Dataset: {result['dataset_name']}")
        print(f"Samples: {result['num_samples']}")
        print(f"Embedding dim: {result['embedding_dim']}")
        
    except Exception as e:
        logger.error(f"Convenience function demo failed: {e}")
        print(f"Error: {e}")


def demonstrate_codon_preprocessing():
    """Demonstrate codon preprocessing in the wrapper."""
    print("\n=== Codon Preprocessing Demo ===")
    
    # Sample sequences
    sequences = [
        "ATGCGTACGTACGT",  # 15 bases = 5 codons
        "GCTAGCTAGCTAGC"   # 15 bases = 5 codons
    ]
    
    try:
        generator = EmbeddingGenerator(
            device="cpu",  # Use CPU for demo
            codon_start_token="E",
            add_codon_start=True
        )
        
        # Test preprocessing only
        processed = generator.preprocess_sequences(sequences)
        print(f"Original sequences: {sequences}")
        print(f"Processed tensor shape: {processed.shape}")
        
        # Get codon statistics from wrapper
        if hasattr(generator.wrapper, 'get_codon_statistics'):
            stats = generator.wrapper.get_codon_statistics(sequences)
            print(f"Codon statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Codon preprocessing demo failed: {e}")
        print(f"Error: {e}")


def main():
    """Run all demonstrations."""
    print("SAE EmbeddingGenerator with HelicalWrapper Demo")
    print("=" * 60)
    
    # Demonstrate main functionality
    demonstrate_embedding_generator()
    
    # Demonstrate convenience function
    demonstrate_convenience_function()
    
    # Demonstrate codon preprocessing
    demonstrate_codon_preprocessing()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nKey features:")
    print("✅ Always uses HelicalWrapper with codon preprocessing")
    print("✅ Each codon starts with 'E' token")
    print("✅ Automatic caching of embeddings")
    print("✅ Batch processing for efficiency")
    print("✅ Configurable codon preprocessing parameters")


if __name__ == "__main__":
    main() 