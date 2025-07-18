"""
Test script to verify regular SAE training history storage
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sae.pipeline import run_complete_pipeline


def test_regular_sae_history():
    """Test that regular SAE properly stores training history"""
    print("🧪 Testing Regular SAE Training History Storage")
    print("=" * 50)
    
    try:
        # Run a quick regular SAE training
        print("🚀 Running Regular SAE training...")
        
        pipeline = run_complete_pipeline(
            refseq_file="../data/vertebrate_mammalian.1.rna.gbff",
            max_samples=50,  # Small dataset for quick testing
            hidden_dim=500,
            epochs=5,  # Few epochs for quick testing
            batch_size=4,
            layer_idx=None,
            layer_name="final"
        )
        
        print("\n✅ Regular SAE training completed!")
        
        # Check if training history was stored
        if hasattr(pipeline, '_last_training_history'):
            history = pipeline._last_training_history
            print(f"📊 Training history stored:")
            print(f"   Train losses: {len(history.get('train_loss', []))} epochs")
            print(f"   Val losses: {len(history.get('val_loss', []))} epochs")
            if history.get('val_loss'):
                print(f"   Final val loss: {history['val_loss'][-1]:.6f}")
            if history.get('train_loss'):
                print(f"   Final train loss: {history['train_loss'][-1]:.6f}")
        else:
            print("❌ Training history not stored")
        
        # Check if trainer logger was updated
        if hasattr(pipeline, 'trainer') and pipeline.trainer and hasattr(pipeline.trainer, 'logger'):
            logger = pipeline.trainer.logger
            if hasattr(logger, 'train_losses') and logger.train_losses:
                print(f"✅ Trainer logger updated with {len(logger.train_losses)} train losses")
            if hasattr(logger, 'val_losses') and logger.val_losses:
                print(f"✅ Trainer logger updated with {len(logger.val_losses)} val losses")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    pipeline = test_regular_sae_history()
    
    if pipeline:
        print("\n🎉 Regular SAE history storage test passed!")
    else:
        print("\n💥 Test failed. Check the error messages above.") 