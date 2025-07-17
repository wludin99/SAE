#!/usr/bin/env python3
"""
Complete SAE Pipeline Example

This script demonstrates the complete pipeline:
1. Load genomic sequences
2. Generate embeddings using HelicalmRNA
3. Train SAE on embeddings
4. Extract and analyze features
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sae.pipeline import run_complete_pipeline, EmbeddingGenerator
from sae.data.genomic_datasets import list_available_datasets


def main():
    print("🧬 Complete SAE Pipeline for Genomic Interpretability")
    print("=" * 70)
    
    # Configuration
    config = {
        'dataset_name': 'human_dna',
        'max_samples': 500,  # Start small for testing
        'embedding_dim': 768,  # Typical for transformer models
        'hidden_dim': 50,  # Number of features to learn
        'epochs': 30,
        'sparsity_weight': 0.1,
        'reconstruction_weight': 1.0,
        'learning_rate': 1e-3,
        'batch_size': 16
    }
    
    print("📋 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    
    # Step 1: List available datasets
    print("1. Available Datasets:")
    datasets = list_available_datasets()
    for name, info in datasets.items():
        print(f"  📊 {name}: {info['description']} ({info['samples']} samples)")
    
    print("\n" + "=" * 70)
    
    # Step 2: Run complete pipeline
    print("2. Running Complete Pipeline...")
    try:
        pipeline = run_complete_pipeline(**config)
        print("✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        print("\nTroubleshooting tips:")
        print("  - Make sure Helical is installed: poetry add helical")
        print("  - Check if you have enough GPU memory")
        print("  - Try reducing max_samples or batch_size")
        return
    
    print("\n" + "=" * 70)
    
    # Step 3: Analyze results
    print("3. Analyzing Results...")
    
    # Load some test data for analysis
    try:
        print("   Loading test data for analysis...")
        embedding_generator = EmbeddingGenerator()
        test_result = embedding_generator.generate_embeddings_from_dataset(
            dataset_name=config['dataset_name'],
            max_samples=50,
            layer_idx=None
        )
        
        test_embeddings = test_result['embeddings']
        test_sequences = test_result['sequences']
        
        print(f"   Test embeddings shape: {test_embeddings.shape}")
        
        # Extract features using trained SAE
        print("   Extracting features with trained SAE...")
        features = pipeline.extract_features(test_embeddings)
        print(f"   Extracted features shape: {features.shape}")
        
        # Analyze feature sparsity
        print("   Analyzing feature sparsity...")
        sparsity = np.mean(features == 0, axis=0)
        active_features = np.sum(features > 0, axis=0)
        
        print(f"   Average sparsity per feature: {np.mean(sparsity):.3f}")
        print(f"   Number of active features per sample: {np.mean(active_features):.1f}")
        
        # Plot feature analysis
        print("   Creating analysis plots...")
        create_analysis_plots(features, sparsity, active_features, test_sequences)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 Pipeline Analysis Complete!")
    print("\n📁 Generated files:")
    print(f"  - Model: {pipeline.model_save_dir}/best_model.pth")
    print(f"  - Training plot: outputs/training_history.png")
    print(f"  - Analysis plots: outputs/feature_analysis_*.png")
    
    print("\n💡 Next steps:")
    print("  - Increase max_samples for better training")
    print("  - Experiment with different hidden_dim values")
    print("  - Try different datasets (drosophila_dna, yeast_dna)")
    print("  - Analyze specific features for biological meaning")


def create_analysis_plots(features, sparsity, active_features, sequences):
    """Create analysis plots for the extracted features"""
    
    # Plot 1: Feature activation heatmap
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.heatmap(features[:20, :], cmap='viridis', cbar_kws={'label': 'Activation'})
    plt.title('Feature Activations (First 20 Samples)')
    plt.xlabel('Feature Index')
    plt.ylabel('Sample Index')
    
    # Plot 2: Feature sparsity distribution
    plt.subplot(2, 2, 2)
    plt.hist(sparsity, bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('Sparsity (fraction of zeros)')
    plt.ylabel('Number of Features')
    plt.title('Feature Sparsity Distribution')
    plt.axvline(np.mean(sparsity), color='red', linestyle='--', label=f'Mean: {np.mean(sparsity):.3f}')
    plt.legend()
    
    # Plot 3: Number of active features per sample
    plt.subplot(2, 2, 3)
    plt.hist(active_features, bins=20, alpha=0.7, color='lightgreen')
    plt.xlabel('Number of Active Features')
    plt.ylabel('Number of Samples')
    plt.title('Active Features per Sample')
    plt.axvline(np.mean(active_features), color='red', linestyle='--', label=f'Mean: {np.mean(active_features):.1f}')
    plt.legend()
    
    # Plot 4: Feature importance (mean activation)
    plt.subplot(2, 2, 4)
    feature_importance = np.mean(features, axis=0)
    plt.bar(range(len(feature_importance)), feature_importance, alpha=0.7, color='orange')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Activation')
    plt.title('Feature Importance (Mean Activation)')
    
    plt.tight_layout()
    plt.savefig('outputs/feature_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Sequence length vs feature activation
    plt.figure(figsize=(10, 6))
    sequence_lengths = [len(seq) for seq in sequences]
    
    # Calculate correlation between sequence length and feature activations
    correlations = []
    for i in range(features.shape[1]):
        corr = np.corrcoef(sequence_lengths, features[:, i])[0, 1]
        correlations.append(corr)
    
    plt.bar(range(len(correlations)), correlations, alpha=0.7, color='purple')
    plt.xlabel('Feature Index')
    plt.ylabel('Correlation with Sequence Length')
    plt.title('Feature Correlation with Sequence Length')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Strong correlation threshold')
    plt.axhline(-0.5, color='red', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/feature_analysis_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Analysis plots saved: outputs/feature_analysis_overview.png, outputs/feature_analysis_correlations.png")


def run_quick_test():
    """Run a very quick test to verify everything works"""
    print("🧪 Quick Test Mode")
    print("=" * 40)
    
    config = {
        'dataset_name': 'human_dna',
        'max_samples': 20,  # Very small for quick test
        'embedding_dim': 768,
        'hidden_dim': 10,
        'epochs': 5,
        'batch_size': 4
    }
    
    try:
        pipeline = run_complete_pipeline(**config)
        print("✅ Quick test passed!")
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    # Check if we should run quick test
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test()
    else:
        main() 