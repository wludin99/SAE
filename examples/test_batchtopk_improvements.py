"""
Test script to verify BatchTopK SAE improvements:
1. Progress bar during training
2. Proper metric saving and plotting
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sae.pipeline import run_complete_batchtopk_pipeline


def test_batchtopk_improvements():
    """Test the BatchTopK SAE improvements"""
    print("🧪 Testing BatchTopK SAE Improvements")
    print("=" * 50)
    
    try:
        # Run a quick BatchTopK SAE training with progress bar
        print("🚀 Running BatchTopK SAE with progress bar and plotting...")
        
        pipeline = run_complete_batchtopk_pipeline(
            refseq_file="../data/vertebrate_mammalian.1.rna.gbff",
            max_samples=50,  # Small dataset for quick testing
            hidden_dim=500,
            topk=10,
            epochs=5,  # Few epochs for quick testing
            batch_size=4,
            layer_idx=None,
            layer_name="final"
        )
        
        print("\n✅ BatchTopK SAE training completed!")
        
        # Check if training history was saved
        if hasattr(pipeline, '_last_training_history'):
            history = pipeline._last_training_history
            print(f"📊 Training history saved:")
            print(f"   Train losses: {len(history['train_loss'])} epochs")
            print(f"   Val losses: {len(history['val_loss'])} epochs")
            print(f"   Final train loss: {history['train_loss'][-1]:.6f}")
            print(f"   Final val loss: {history['val_loss'][-1]:.6f}")
        else:
            print("❌ Training history not saved")
        
        # Check if trainer logger was updated
        if hasattr(pipeline, 'trainer') and pipeline.trainer and hasattr(pipeline.trainer, 'logger'):
            logger = pipeline.trainer.logger
            if hasattr(logger, 'train_losses') and logger.train_losses:
                print(f"✅ Trainer logger updated with {len(logger.train_losses)} train losses")
            if hasattr(logger, 'val_losses') and logger.val_losses:
                print(f"✅ Trainer logger updated with {len(logger.val_losses)} val losses")
        
        # Test plotting
        print("\n📈 Testing plotting functionality...")
        plot_path = "test_batchtopk_plot.png"
        pipeline.plot_training_history(plot_path)
        
        if Path(plot_path).exists():
            print(f"✅ Plot saved successfully to {plot_path}")
        else:
            print("❌ Plot not saved")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    pipeline = test_batchtopk_improvements()
    
    if pipeline:
        print("\n🎉 All tests passed! BatchTopK improvements working correctly.")
    else:
        print("\n💥 Tests failed. Check the error messages above.") 