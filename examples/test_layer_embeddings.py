#!/usr/bin/env python3
"""
Test script for layer-specific embedding extraction from Helical model.
This script demonstrates how to extract embeddings from different layers
of the Helical model and verify their shapes and properties.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sae.preprocessing import create_helical_wrapper    
from sae.pipeline import EmbeddingGenerator
from sae.data import RefSeqDataset


def test_single_layer_extraction():
    """Test extracting embeddings from a single specific layer."""
    print("=== Testing Single Layer Extraction ===")
    
    # Initialize the wrapper
    wrapper = create_helical_wrapper()
    
    # Test sequence
    test_sequence = "AUGCAUGCAUGCAUGCAUGC"
    
    # Test different layers
    layers_to_test = ["initial", "after_mlp_1", "after_mlp_2", "after_mlp_3", "final"]
    
    for layer_name in layers_to_test:
        print(f"\nTesting layer: {layer_name}")
        
        # Extract embeddings from specific layer
        embeddings = wrapper.get_layer_embeddings(
            sequences=[test_sequence],
            layer_name=layer_name
        )
        
        print(f"  Shape: {embeddings.shape}")
        print(f"  Type: {type(embeddings)}")
        print(f"  Device: {embeddings.device}")
        
        # Verify it's a 3D tensor (batch_size, sequence_length, embedding_dim)
        assert embeddings.dim() == 3, f"Expected 3D tensor, got {embeddings.dim()}D"
        assert embeddings.shape[0] == 1, f"Expected batch size 1, got {embeddings.shape[0]}"
        assert embeddings.shape[2] == 256, f"Expected embedding dim 256, got {embeddings.shape[2]}"


def test_all_layers_extraction():
    """Test extracting embeddings from all layers at once."""
    print("\n=== Testing All Layers Extraction ===")
    
    # Initialize the wrapper
    wrapper = create_helical_wrapper()
    
    # Test sequence
    test_sequence = "AUGCAUGCAUGCAUGCAUGC"
    
    # Extract embeddings from all layers
    all_embeddings = wrapper.get_all_layer_embeddings(
        sequences=[test_sequence]
    )
    
    print(f"Number of layers extracted: {len(all_embeddings)}")
    
    # Check each layer
    for layer_name, embeddings in all_embeddings.items():
        print(f"\nLayer: {layer_name}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Type: {type(embeddings)}")
        
        # Verify it's a 3D tensor
        assert embeddings.dim() == 3, f"Expected 3D tensor for {layer_name}, got {embeddings.dim()}D"
        assert embeddings.shape[0] == 1, f"Expected batch size 1 for {layer_name}, got {embeddings.shape[0]}"
        assert embeddings.shape[2] == 256, f"Expected embedding dim 256 for {layer_name}, got {embeddings.shape[2]}"


def test_embedding_generator_integration():
    """Test the embedding generator with layer extraction."""
    print("\n=== Testing Embedding Generator Integration ===")
    
    # Initialize the embedding generator
    generator = EmbeddingGenerator()
    
    # Test sequence
    test_sequence = "AUGCAUGCAUGCAUGCAUGC"
    
    # Test single layer extraction using the wrapper directly
    print("Testing single layer extraction through generator wrapper:")
    embeddings = generator.wrapper.get_layer_embeddings(
        sequences=[test_sequence],
        layer_name="after_mlp_2"
    )
    
    print(f"  Shape: {embeddings.shape}")
    print(f"  Type: {type(embeddings)}")
    
    # Test all layers extraction using the wrapper directly
    print("\nTesting all layers extraction through generator wrapper:")
    all_embeddings = generator.wrapper.get_all_layer_embeddings(
        sequences=[test_sequence]
    )
    
    print(f"  Number of layers: {len(all_embeddings)}")
    for layer_name, emb in all_embeddings.items():
        print(f"    {layer_name}: {emb.shape}")


def test_with_refseq_data():
    """Test layer extraction with actual RefSeq data."""
    print("\n=== Testing with RefSeq Data ===")
    
    # Initialize components
    wrapper = create_helical_wrapper()
    
    # Create a small RefSeq dataset
    data_dir = Path(__file__).parent.parent / "data" / "refseq"
    if not data_dir.exists():
        print("  RefSeq data directory not found, skipping this test")
        return
    
    try:
        dataset = RefSeqDataset(
            data_dir=data_dir,
            max_length=100,  # Keep sequences short for testing
            max_sequences=5   # Only use a few sequences
        )
        
        # Get a few sequences
        sequences = []
        for i in range(min(3, len(dataset))):
            sequences.append(dataset[i])
        
        print(f"  Loaded {len(sequences)} sequences from RefSeq dataset")
        
        # Test single layer extraction
        embeddings = wrapper.get_layer_embeddings(
            sequences=sequences,
            layer_name="final"
        )
        
        print(f"  Final layer embeddings shape: {embeddings.shape}")
        
        # Test all layers extraction
        all_embeddings = wrapper.get_all_layer_embeddings(
            sequences=sequences
        )
        
        print(f"  All layers extracted: {len(all_embeddings)} layers")
        for layer_name, emb in all_embeddings.items():
            print(f"    {layer_name}: {emb.shape}")
            
    except Exception as e:
        print(f"  Error loading RefSeq data: {e}")


def test_embedding_properties():
    """Test properties of extracted embeddings."""
    print("\n=== Testing Embedding Properties ===")
    
    wrapper = create_helical_wrapper()
    test_sequence = "AUGCAUGCAUGCAUGCAUGC"
    
    # Extract embeddings from different layers
    initial_emb = wrapper.get_layer_embeddings([test_sequence], "initial")
    final_emb = wrapper.get_layer_embeddings([test_sequence], "final")
    
    print(f"Initial embeddings shape: {initial_emb.shape}")
    print(f"Final embeddings shape: {final_emb.shape}")
    
    # Check that embeddings are different between layers
    if not torch.allclose(initial_emb, final_emb, atol=1e-6):
        print("  ✓ Embeddings from different layers are different")
    else:
        print("  ⚠ Embeddings from different layers are identical (unexpected)")
    
    # Check for NaN values
    if not torch.isnan(initial_emb).any():
        print("  ✓ Initial embeddings contain no NaN values")
    else:
        print("  ⚠ Initial embeddings contain NaN values")
    
    if not torch.isnan(final_emb).any():
        print("  ✓ Final embeddings contain no NaN values")
    else:
        print("  ⚠ Final embeddings contain NaN values")
    
    # Check embedding statistics
    print(f"  Initial embeddings - Mean: {initial_emb.mean():.4f}, Std: {initial_emb.std():.4f}")
    print(f"  Final embeddings - Mean: {final_emb.mean():.4f}, Std: {final_emb.std():.4f}")


def main():
    """Run all tests."""
    print("Starting Layer Embedding Extraction Tests")
    print("=" * 50)
    
    try:
        test_single_layer_extraction()
        test_all_layers_extraction()
        test_embedding_generator_integration()
        test_with_refseq_data()
        test_embedding_properties()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully! ✓")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 