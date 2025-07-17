#!/usr/bin/env python3
"""
Test script to verify the pipeline works correctly
"""

def test_pipeline_initialization():
    """Test that the pipeline can be initialized correctly"""
    print("🧪 Testing pipeline initialization...")

    try:
        from sae.pipeline import SAETrainingPipeline

        # Test basic initialization
        pipeline = SAETrainingPipeline(
            embedding_dim=768,
            hidden_dim=50,
            sparsity_weight=0.1,
            reconstruction_weight=1.0,
            learning_rate=1e-3
        )
        print("   ✅ Pipeline initialized successfully")

        # Test with batch_size (should not be passed to __init__)
        try:
            pipeline = SAETrainingPipeline(
                embedding_dim=768,
                hidden_dim=50,
                batch_size=32  # This should cause an error
            )
            print("   ❌ Should have failed with batch_size parameter")
            return False
        except TypeError:
            print("   ✅ Correctly rejected batch_size in __init__")

        return True

    except Exception as e:
        print(f"   ❌ Pipeline initialization failed: {e}")
        return False


def test_run_complete_pipeline():
    """Test the run_complete_pipeline function"""
    print("\n🧪 Testing run_complete_pipeline function...")

    try:
        from sae.pipeline import run_complete_pipeline

        # Test that it accepts batch_size parameter
        print("   Testing with batch_size parameter...")

        # This should work without error
        pipeline = run_complete_pipeline(
            dataset_name="human_dna",
            max_samples=10,  # Very small for testing
            embedding_dim=768,
            hidden_dim=10,
            epochs=2,
            batch_size=4
        )

        print("   ✅ run_complete_pipeline worked with batch_size")
        return True

    except Exception as e:
        print(f"   ❌ run_complete_pipeline failed: {e}")
        return False


if __name__ == "__main__":
    print("🧬 SAE Pipeline Test")
    print("=" * 50)

    # Test pipeline initialization
    init_ok = test_pipeline_initialization()

    if init_ok:
        # Test complete pipeline
        pipeline_ok = test_run_complete_pipeline()

        if pipeline_ok:
            print("\n✅ All pipeline tests passed!")
            print("\n💡 The pipeline is working correctly.")
        else:
            print("\n❌ Pipeline function test failed.")
    else:
        print("\n❌ Pipeline initialization test failed.")
