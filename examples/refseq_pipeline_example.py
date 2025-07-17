"""
Example: Using RefSeq Data with SAE Pipeline

This script demonstrates how to use the RefSeq data with the SAE training pipeline,
including gene feature extraction and codon preprocessing for Helical model input.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sae.pipeline import run_complete_pipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("RefSeq SAE Pipeline Example")
    print("="*60)
    
    # Path to your RefSeq file
    refseq_file = "../data/vertebrate_mammalian.1.rna.gbff"
    
    if not os.path.exists(refseq_file):
        print(f"❌ RefSeq file not found: {refseq_file}")
        print("Please make sure you have downloaded the RefSeq file to the data/ directory")
        return
    
    print(f"📁 Using RefSeq file: {refseq_file}")
    
    try:
        # Run the complete pipeline with RefSeq data
        print("\n🚀 Starting SAE training pipeline with RefSeq data...")
        
        pipeline = run_complete_pipeline(
            # Data configuration
            refseq_file=refseq_file,
            dataset_name="vertebrate_mammalian",  # Name for logging purposes
            max_samples=500,  # Start with a small number for testing
            filter_by_type="mRNA",  # Only use mRNA sequences
            use_cds=True,  # Use CDS features for better codon alignment
            
            # Model configuration
            embedding_dim=768,  # Helical model embedding dimension
            hidden_dim=50,  # Number of SAE features to learn
            epochs=20,  # Training epochs
            batch_size=16,  # Smaller batch size for memory efficiency
            
            # Training configuration
            sparsity_weight=0.1,
            learning_rate=1e-3,
            
            # Output configuration
            cache_dir="./outputs/refseq_embeddings_cache",
            model_save_dir="./outputs/refseq_sae_models"
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Model saved to: {pipeline.model_save_dir}")
        print(f"📁 Embeddings cached to: {pipeline.cache_dir}")
        
        # Show some information about the trained model
        print(f"\n📊 Model Information:")
        print(f"  Input dimension: {pipeline.embedding_dim}")
        print(f"  Hidden dimension: {pipeline.hidden_dim}")
        print(f"  Sparsity weight: {pipeline.sparsity_weight}")
        print(f"  Learning rate: {pipeline.learning_rate}")
        
        # Example of using the trained model
        print(f"\n🔍 Example: Extracting features from sample embeddings...")
        
        # Create some dummy embeddings for demonstration
        import numpy as np
        sample_embeddings = np.random.randn(10, pipeline.embedding_dim)
        
        # Extract features using the trained SAE
        features = pipeline.extract_features(sample_embeddings)
        print(f"  Input embeddings shape: {sample_embeddings.shape}")
        print(f"  Extracted features shape: {features.shape}")
        print(f"  Feature sparsity: {(features == 0).mean():.2%}")
        
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("🎉 RefSeq SAE Pipeline Example Completed!")
    print("\nNext steps:")
    print("1. Increase max_samples for full training")
    print("2. Experiment with different hidden_dim values")
    print("3. Try different filter_by_type values (rRNA, tRNA)")
    print("4. Analyze the learned features")
    print("5. Use the trained model for downstream tasks")


def run_small_test():
    """Run a very small test to verify everything works"""
    print("\n🧪 Running small test...")
    
    refseq_file = "../data/vertebrate_mammalian.1.rna.gbff"
    
    try:
        pipeline = run_complete_pipeline(
            refseq_file=refseq_file,
            dataset_name="refseq_test",
            max_samples=50,  # Very small for testing
            filter_by_type="mRNA",
            use_cds=True,
            embedding_dim=768,
            hidden_dim=10,  # Small for testing
            epochs=5,  # Few epochs for testing
            batch_size=8,
            sparsity_weight=0.1,
            learning_rate=1e-3,
            cache_dir="./outputs/test_cache",
            model_save_dir="./outputs/test_models"
        )
        
        print("✅ Small test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Small test failed: {e}")
        return False


if __name__ == "__main__":
    # First run a small test
    if run_small_test():
        print("\n" + "="*60)
        # If test passes, run the full example
        main()
    else:
        print("\n❌ Small test failed. Please check your setup.") 