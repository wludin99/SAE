#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

def test_imports():
    """Test all the main imports"""
    print("🧪 Testing imports...")

    try:
        # Test basic imports
        print("1. Testing basic imports...")
        from sae import SAE, CombinedLoss
        print("   ✅ SAE and CombinedLoss imported successfully")

        # Test data imports
        print("2. Testing data imports...")
        from sae import list_available_datasets, load_genomic_dataset
        print("   ✅ Data loading functions imported successfully")

        # Test pipeline imports
        print("3. Testing pipeline imports...")
        from sae import EmbeddingGenerator, SAETrainingPipeline
        print("   ✅ Pipeline components imported successfully")

        # Test losses
        print("4. Testing losses...")
        from sae.losses import CombinedLoss, L1SparsityLoss, L2SparsityLoss, SAELoss
        print("   ✅ All loss functions imported successfully")

        # Test models
        print("5. Testing models...")
        from sae.models.sae import SAE
        print("   ✅ SAE model imported successfully")

        print("\n🎉 All imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality"""
    print("\n🔧 Testing basic functionality...")

    try:
        # Test SAE model creation
        print("1. Testing SAE model creation...")
        from sae import SAE
        model = SAE(input_size=100, hidden_size=20)
        print("   ✅ SAE model created successfully")

        # Test loss function
        print("2. Testing loss function...")
        import torch

        from sae import CombinedLoss
        loss_fn = CombinedLoss(sparsity_weight=0.1, reconstruction_weight=1.0)

        # Create dummy data
        reconstructed = torch.randn(10, 100)
        original = torch.randn(10, 100)
        activations = torch.randn(10, 20)

        total_loss, recon_loss, sparsity_loss = loss_fn(reconstructed, original, activations)
        print(f"   ✅ Loss computed successfully: {total_loss.item():.4f}")

        # Test dataset listing
        print("3. Testing dataset listing...")
        from sae import list_available_datasets
        datasets = list_available_datasets()
        print(f"   ✅ Found {len(datasets)} available datasets")

        print("\n🎉 All functionality tests passed!")
        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧬 SAE Package Import Test")
    print("=" * 50)

    # Test imports
    imports_ok = test_imports()

    if imports_ok:
        # Test functionality
        functionality_ok = test_basic_functionality()

        if functionality_ok:
            print("\n✅ All tests passed! The package is ready to use.")
            print("\n💡 You can now run:")
            print("   poetry run python examples/complete_sae_pipeline.py --quick")
        else:
            print("\n❌ Functionality tests failed.")
    else:
        print("\n❌ Import tests failed.")
        print("\n💡 Try installing dependencies:")
        print("   poetry install")
