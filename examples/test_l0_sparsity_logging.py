#!/usr/bin/env python3
"""
Test script to verify L0 sparsity logging from regular SAE pipeline
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from sae.pipeline.sae_pipeline import run_complete_pipeline


def test_l0_sparsity_logging():
    """Test that regular SAE pipeline correctly logs L0 sparsity"""
    print("🧪 Testing Regular SAE L0 Sparsity Logging")
    print("=" * 60)

    try:
        # Run a small training session
        pipeline = run_complete_pipeline(
            refseq_file="../data/vertebrate_mammalian.1.rna.gbff",
            max_samples=50,  # Small dataset for quick test
            hidden_dim=100,
            epochs=5,  # Few epochs for quick test
            batch_size=4
        )

        print("\n📊 Checking L0 sparsity logging...")

        # Check if training history is stored
        if hasattr(pipeline, "_last_training_history") and pipeline._last_training_history:
            print("✅ Training history stored successfully")
            print(f"   Keys: {list(pipeline._last_training_history.keys())}")

            # Check if L0 sparsity is included
            if "val_l0_sparsity" in pipeline._last_training_history:
                l0_values = pipeline._last_training_history["val_l0_sparsity"]
                print(f"✅ L0 sparsity values found: {len(l0_values)} epochs")
                print(f"   Final L0 sparsity: {l0_values[-1]:.1f}")
                print(f"   All L0 values: {[f'{x:.1f}' for x in l0_values]}")

                # Check if values are reasonable (should be between 0 and hidden_dim)
                if 0 <= l0_values[-1] <= 100:
                    print("✅ L0 sparsity values look reasonable")
                    print(f"   Interpretation: On average, {l0_values[-1]:.1f} out of 100 features are active per token position")
                else:
                    print("⚠️  L0 sparsity values seem out of range")
            else:
                print("❌ L0 sparsity not found in training history")
        else:
            print("❌ No training history stored")

        # Also check trainer's training history
        if hasattr(pipeline, "trainer") and pipeline.trainer and hasattr(pipeline.trainer, "training_history"):
            print(f"\n📊 Trainer training history: {len(pipeline.trainer.training_history)} epochs")
            if pipeline.trainer.training_history:
                last_epoch = pipeline.trainer.training_history[-1]
                print(f"   Last epoch keys: {list(last_epoch.keys())}")
                if "val_l0_sparsity" in last_epoch:
                    print(f"   Final trainer L0 sparsity: {last_epoch['val_l0_sparsity']:.1f}")
                else:
                    print("   ❌ No val_l0_sparsity in trainer history")

        print("\n✅ Test completed!")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_l0_sparsity_logging()
