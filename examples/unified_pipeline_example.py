"""
Unified Pipeline Example

This example demonstrates how to use both regular SAE and BatchTopK SAE pipelines
with the new unified base pipeline structure.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sae.pipeline import (
    SAETrainingPipeline, 
    run_complete_pipeline,
    BatchTopKSAETrainingPipeline, 
    run_complete_batchtopk_pipeline
)


def example_regular_sae():
    """Example using regular SAE pipeline"""
    print("🔧 Example: Regular SAE Pipeline")
    print("=" * 50)
    
    try:
        # Run regular SAE pipeline
        pipeline = run_complete_pipeline(
            refseq_file="../data/vertebrate_mammalian.1.rna.gbff",
            max_samples=100,
            hidden_dim=1000,
            epochs=10,
            layer_idx=None,  # Final layer
            layer_name="final"
        )
        
        print(f"✅ Regular SAE pipeline completed!")
        print(f"   Model saved to: {pipeline.model_save_dir}")
        
        # Test feature extraction
        import numpy as np
        test_embeddings = np.random.randn(10, pipeline.embedding_dim)
        features = pipeline.extract_features(test_embeddings)
        print(f"   Feature extraction test: {test_embeddings.shape} -> {features.shape}")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Error in regular SAE pipeline: {e}")
        return None


def example_batchtopk_sae():
    """Example using BatchTopK SAE pipeline"""
    print("\n🔧 Example: BatchTopK SAE Pipeline")
    print("=" * 50)
    
    try:
        # Run BatchTopK SAE pipeline
        pipeline = run_complete_batchtopk_pipeline(
            refseq_file="../data/vertebrate_mammalian.1.rna.gbff",
            max_samples=100,
            hidden_dim=1000,
            topk=10,
            epochs=10,
            layer_idx=0,  # First layer
            layer_name="layer_0"
        )
        
        print(f"✅ BatchTopK SAE pipeline completed!")
        print(f"   Model saved to: {pipeline.model_save_dir}")
        
        # Test feature extraction
        import numpy as np
        test_embeddings = np.random.randn(10, pipeline.embedding_dim)
        features = pipeline.extract_features(test_embeddings)
        print(f"   Feature extraction test: {test_embeddings.shape} -> {features.shape}")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Error in BatchTopK SAE pipeline: {e}")
        return None


def example_direct_pipeline_usage():
    """Example using pipeline classes directly"""
    print("\n🔧 Example: Direct Pipeline Usage")
    print("=" * 50)
    
    try:
        # Create regular SAE pipeline directly
        regular_pipeline = SAETrainingPipeline(
            embedding_dim=512,  # Will be auto-detected
            hidden_dim=1000,
            layer_idx=None,
            layer_name="final"
        )
        
        # Create BatchTopK SAE pipeline directly
        batchtopk_pipeline = BatchTopKSAETrainingPipeline(
            embedding_dim=512,  # Will be auto-detected
            hidden_dim=1000,
            topk=10,
            layer_idx=0,
            layer_name="layer_0"
        )
        
        print("✅ Both pipeline instances created successfully!")
        print(f"   Regular SAE: {type(regular_pipeline).__name__}")
        print(f"   BatchTopK SAE: {type(batchtopk_pipeline).__name__}")
        
        # Both inherit from the same base class
        from sae.pipeline.base_pipeline import BaseSAETrainingPipeline
        print(f"   Both inherit from: {BaseSAETrainingPipeline.__name__}")
        
        return regular_pipeline, batchtopk_pipeline
        
    except Exception as e:
        print(f"❌ Error in direct pipeline usage: {e}")
        return None, None


def main():
    """Run all examples"""
    print("🚀 Unified Pipeline Examples")
    print("=" * 60)
    print("This example demonstrates the unified pipeline structure where")
    print("both regular SAE and BatchTopK SAE pipelines inherit from a")
    print("common base pipeline class.\n")
    
    # Example 1: Regular SAE
    regular_pipeline = example_regular_sae()
    
    # Example 2: BatchTopK SAE
    batchtopk_pipeline = example_batchtopk_sae()
    
    # Example 3: Direct usage
    direct_regular, direct_batchtopk = example_direct_pipeline_usage()
    
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
    
    if regular_pipeline:
        print("✅ Regular SAE pipeline: SUCCESS")
    else:
        print("❌ Regular SAE pipeline: FAILED")
        
    if batchtopk_pipeline:
        print("✅ BatchTopK SAE pipeline: SUCCESS")
    else:
        print("❌ BatchTopK SAE pipeline: FAILED")
        
    if direct_regular and direct_batchtopk:
        print("✅ Direct pipeline usage: SUCCESS")
    else:
        print("❌ Direct pipeline usage: FAILED")
    
    print("\n🎯 Key Benefits of Unified Structure:")
    print("   • Shared functionality in BaseSAETrainingPipeline")
    print("   • Consistent interface across all pipeline types")
    print("   • Easy to extend with new SAE variants")
    print("   • Reduced code duplication")
    print("   • Unified data preparation and embedding generation")


if __name__ == "__main__":
    main() 