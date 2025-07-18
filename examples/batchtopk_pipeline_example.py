"""
Example: Using BatchTopK SAE with RefSeq Data

This script demonstrates how to train a BatchTopK SAE model on RefSeq data,
using layer-specific embeddings from the Helical model.
"""

import logging
import os
import torch
import numpy as np
from pathlib import Path

from sae.preprocessing.helical_wrapper import create_helical_wrapper
from sae.models.batchtopk_sae import BatchTopKSAE
from sae.preprocessing.refseq_preprocessor import RefSeqPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_batchtopk_pipeline(
    refseq_file: str,
    dataset_name: str = "refseq_batchtopk",
    max_samples: int = 100,
    filter_by_type: str = "mRNA",
    use_cds: bool = True,
    hidden_dim: int = 512,
    topk: int = 10,
    epochs: int = 20,
    batch_size: int = 2,  # Reduced from 8
    learning_rate: float = 1e-3,
    layer_name: str = "final",
    cache_dir: str = "./outputs/batchtopk_cache",
    model_save_dir: str = "./outputs/batchtopk_models",
    device: str = "auto"  # "auto", "cuda", or "cpu"
):
    """
    Run BatchTopK SAE training pipeline on RefSeq data.
    
    Args:
        refseq_file: Path to RefSeq GenBank file
        dataset_name: Name for logging and caching
        max_samples: Maximum number of sequences to process
        filter_by_type: Filter by molecule type (mRNA, rRNA, tRNA)
        use_cds: Whether to use CDS features
        hidden_dim: Number of SAE features to learn
        topk: Number of top values to keep in BatchTopK
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        layer_name: Which layer to extract embeddings from
        cache_dir: Directory to cache embeddings
        model_save_dir: Directory to save trained model
    """
    
    # Create output directories
    cache_dir = Path(cache_dir)
    model_save_dir = Path(model_save_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting BatchTopK SAE Pipeline")
    print(f"📁 RefSeq file: {refseq_file}")
    print(f"🎯 Layer: {layer_name}")
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔢 Hidden dimension: {hidden_dim}")
    print(f"📊 TopK: {topk}")
    print(f"📈 Epochs: {epochs}")
    print(f"💻 Device: {device}")
    
    # Step 1: Process RefSeq data
    print("\n📋 Step 1: Processing RefSeq data...")
    preprocessor = RefSeqPreprocessor()
    processed_records = preprocessor.process_refseq_file(
        file_path=refseq_file,
        max_samples=max_samples,
        filter_by_type=filter_by_type,
        use_cds=use_cds
    )
    
    if not processed_records:
        raise ValueError("No valid sequences found in RefSeq file")
    
    sequences = [record["final_sequence"] for record in processed_records]
    print(f"✅ Processed {len(sequences)} sequences")
    # Debug: Show some sequence info
    if sequences:
        print(f"First sequence length: {len(sequences[0])}")
        print(f"First sequence preview: {sequences[0][:50]}...")
        print(f"Average sequence length: {sum(len(s) for s in sequences) / len(sequences):.1f}")
    
    # Step 2: Extract embeddings using Helical model
    print(f"\n🧬 Step 2: Extracting {layer_name} embeddings...")
    wrapper = create_helical_wrapper(batch_size=1, device=device)  # Use batch_size=1 for embedding extraction
    
    # Extract embeddings from specified layer
    embeddings = wrapper.get_layer_embeddings(sequences, layer_name)
    
    # Convert to numpy for easier handling
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    print(f"✅ Extracted embeddings shape: {embeddings.shape}")
    
    # Step 3: Prepare data for training
    print("\n📊 Step 3: Preparing training data...")
    
    # Debug: Print original embedding shape
    print(f"Original embeddings shape: {embeddings.shape}")
    
    # Handle different embedding shapes
    original_shape = embeddings.shape
    if len(original_shape) == 3:
        # 3D shape: (num_sequences, seq_len, embed_dim)
        num_sequences, seq_len, embed_dim = original_shape
        print(f"3D embeddings: num_sequences={num_sequences}, seq_len={seq_len}, embed_dim={embed_dim}")
        
        # Flatten to (num_sequences * seq_len, embed_dim) for training
        embeddings_flat = embeddings.reshape(-1, embed_dim)
        print(f"Flattened to: {embeddings_flat.shape}")
        
    elif len(original_shape) == 2:
        # 2D shape: (num_sequences, embed_dim) - already flattened
        print(f"2D embeddings: shape={embeddings.shape}")
        embeddings_flat = embeddings
        embed_dim = embeddings.shape[-1]
        
    else:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    
    print(f"✅ Final embeddings shape: {embeddings_flat.shape}")
    print(f"✅ Embedding dimension: {embed_dim}")
    
    # Convert to torch tensors and move to appropriate device
    embeddings_tensor = torch.tensor(embeddings_flat, dtype=torch.float32).to(device)
    
    # Clear GPU cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Step 4: Initialize and train BatchTopK SAE
    print(f"\n🎯 Step 4: Training BatchTopK SAE...")
    
    # Initialize model with sample embeddings for pre-bias
    model = BatchTopKSAE(
        input_size=embed_dim,
        hidden_size=hidden_dim,
        topk=topk
    ).to(device)
    
    # Initialize weights using sample embeddings
    model.reinit_with_embeddings(embeddings_tensor)
    
    print(f"✅ Model initialized with {hidden_dim} features")
    
    # Training loop
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.encoder_bias},
        {'params': model.pre_bias}
    ], lr=learning_rate)
    
    print(f"🔄 Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        
        # Create batches
        num_samples = embeddings_tensor.shape[0]
        indices = torch.randperm(num_samples)
        
        total_loss = 0
        num_batches = 0
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = embeddings_tensor[batch_indices]
            
            # Training step
            loss, loss_dict = model.train_step(batch_data, optimizer=optimizer, lr=learning_rate)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.6f}")
    
    print(f"✅ Training completed!")
    
    # Step 5: Save model and results
    print(f"\n💾 Step 5: Saving model and results...")
    
    # Save model
    model_path = model_save_dir / f"batchtopk_sae_{dataset_name}_{layer_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_size': embed_dim,
            'hidden_size': hidden_dim,
            'topk': topk,
            'layer_name': layer_name,
            'dataset_name': dataset_name,
            'num_sequences': len(sequences)
        }
    }, model_path)
    
    # Save embeddings for later use
    embeddings_path = cache_dir / f"embeddings_{dataset_name}_{layer_name}.npy"
    np.save(embeddings_path, embeddings)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Embeddings saved to: {embeddings_path}")
    
    # Step 6: Evaluate model
    print(f"\n📊 Step 6: Evaluating model...")
    
    model.eval()
    with torch.no_grad():
        # Test reconstruction
        test_batch = embeddings_tensor[:batch_size]
        reconstructed = model(test_batch)
        
        # Compute reconstruction error
        mse = torch.nn.functional.mse_loss(reconstructed, test_batch)
        
        # Extract features
        features = model.encoder(test_batch - model.pre_bias) + model.encoder_bias
        features = model.topk_layer(features)
        
        # Compute sparsity
        sparsity = (features == 0).float().mean()
        
        print(f"✅ Reconstruction MSE: {mse:.6f}")
        print(f"✅ Feature sparsity: {sparsity:.2%}")
        print(f"✅ Features shape: {features.shape}")
    
    return {
        'model': model,
        'embeddings': embeddings,
        'model_path': model_path,
        'embeddings_path': embeddings_path,
        'config': {
            'input_size': embed_dim,
            'hidden_size': hidden_dim,
            'topk': topk,
            'layer_name': layer_name,
            'dataset_name': dataset_name,
            'num_sequences': len(sequences)
        }
    }


def main():
    """Main function to run the BatchTopK pipeline example."""
    print("BatchTopK SAE Pipeline Example")
    print("="*60)
    
    # Path to your RefSeq file
    refseq_file = "../data/vertebrate_mammalian.1.rna.gbff"
    
    if not os.path.exists(refseq_file):
        print(f"❌ RefSeq file not found: {refseq_file}")
        print("Please make sure you have downloaded the RefSeq file to the data/ directory")
        return
    
    print(f"📁 Using RefSeq file: {refseq_file}")
    
    try:
        # Run the BatchTopK pipeline
        results = run_batchtopk_pipeline(
            refseq_file=refseq_file,
            dataset_name="vertebrate_mammalian",
            max_samples=100,  # Reduced from 200
            filter_by_type="mRNA",
            use_cds=True,
            hidden_dim=128,  # Reduced from 256
            topk=5,  # Keep top 5 values
            epochs=15,
            batch_size=2,  # Reduced from 8
            learning_rate=1e-3,
            layer_name="final",  # Use final layer embeddings
            cache_dir="./outputs/batchtopk_cache",
            model_save_dir="./outputs/batchtopk_models"
        )
        
        print("\n✅ BatchTopK Pipeline completed successfully!")
        
        # Show results
        print("\n📊 Results Summary:")
        print(f"  Model saved to: {results['model_path']}")
        print(f"  Embeddings saved to: {results['embeddings_path']}")
        print(f"  Input dimension: {results['config']['input_size']}")
        print(f"  Hidden dimension: {results['config']['hidden_size']}")
        print(f"  TopK: {results['config']['topk']}")
        print(f"  Layer: {results['config']['layer_name']}")
        print(f"  Sequences processed: {results['config']['num_sequences']}")
        
        # Example of using the trained model
        print("\n🔍 Example: Using trained model for feature extraction...")
        model = results['model']
        
        # Create some test embeddings
        test_embeddings = torch.randn(5, results['config']['input_size'])
        
        # Extract features
        model.eval()
        with torch.no_grad():
            features = model.encoder(test_embeddings - model.pre_bias) + model.encoder_bias
            features = model.topk_layer(features)
        
        print(f"  Test embeddings shape: {test_embeddings.shape}")
        print(f"  Extracted features shape: {features.shape}")
        print(f"  Feature sparsity: {(features == 0).float().mean():.2%}")
        
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("🎉 BatchTopK SAE Pipeline Example Completed!")
    print("\nNext steps:")
    print("1. Try different layer names (initial, after_mlp_1, after_mlp_2, after_mlp_3)")
    print("2. Experiment with different topk values")
    print("3. Increase hidden_dim for more features")
    print("4. Analyze the learned dictionary elements")
    print("5. Use the model for downstream tasks")


def run_small_test():
    """Run a very small test to verify everything works"""
    print("\n🧪 Running small BatchTopK test...")
    
    refseq_file = "../data/vertebrate_mammalian.1.rna.gbff"
    
    try:
        results = run_batchtopk_pipeline(
            refseq_file=refseq_file,
            dataset_name="test",
            max_samples=10,  
            filter_by_type="mRNA",
            use_cds=True,
            hidden_dim=1000,  
            topk=20, 
            epochs=5,  
            batch_size=4,  
            learning_rate=1e-3,
            layer_name="final",
            cache_dir="./outputs/test_batchtopk_cache",
            model_save_dir="./outputs/test_batchtopk_models"
        )
        
        print("✅ Small BatchTopK test completed successfully!")
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