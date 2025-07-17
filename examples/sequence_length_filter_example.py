#!/usr/bin/env python3
"""
Example showing how to use the sequence length filter in the RefSeqDataset.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from sae.data.refseq_parser import create_refseq_dataloader, load_refseq_dataset
from sae.pipeline.pipeline import run_complete_pipeline


def demonstrate_refseq_dataset_filtering():
    """Demonstrate length filtering with RefSeqDataset."""

    print("🔬 RefSeqDataset Length Filtering Example")
    print("=" * 50)

    try:
        # Load dataset with different length filters
        print("Loading dataset with max_length=512...")
        dataset_512 = load_refseq_dataset(
            file_path="../data/vertebrate_mammalian.1.rna.gbff",
            max_samples=100,
            max_length=512,  # Only sequences <= 512 bases
            filter_by_type="mRNA",
            min_length=100
        )

        print(f"✅ Dataset loaded: {len(dataset_512)} sequences")

        # Show some statistics
        lengths = [seq["length"] for seq in dataset_512.sequences]
        print("📊 Length statistics:")
        print(f"  - Min length: {min(lengths)}")
        print(f"  - Max length: {max(lengths)}")
        print(f"  - Mean length: {sum(lengths) / len(lengths):.1f}")

        # Create dataloader
        dataloader = create_refseq_dataloader(dataset_512, batch_size=4)
        print(f"📦 DataLoader created: {len(dataloader)} batches")

        # Show a sample batch
        sample_batch = next(iter(dataloader))
        print(f"📋 Sample batch shape: {sample_batch['input_ids'].shape}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tips:")
        print("  - Make sure your RefSeq file exists")
        print("  - Try reducing max_length if you get memory errors")

def compare_dataset_length_filters():
    """Compare different length filters with RefSeqDataset."""

    print("🔬 Comparing Different Length Filters with RefSeqDataset")
    print("=" * 50)

    length_filters = [256, 512, 1024, None]  # None = no filter

    for max_length in length_filters:
        print(f"\n📏 Testing max_length = {max_length}")

        try:
            dataset = load_refseq_dataset(
                file_path="../data/vertebrate_mammalian.1.rna.gbff",
                max_samples=50,
                max_length=max_length,
                filter_by_type="mRNA",
                min_length=100
            )

            lengths = [seq["length"] for seq in dataset.sequences]
            print(f"  ✅ Success: {len(dataset)} sequences")
            print(f"  📊 Length range: {min(lengths)} - {max(lengths)}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")

def run_pipeline_with_dataset_filtering():
    """Run the SAE pipeline using RefSeqDataset filtering."""

    print("🚀 SAE Pipeline with RefSeqDataset Length Filtering")
    print("=" * 50)

    try:
        # Note: The pipeline now uses RefSeqPreprocessor which doesn't have length filtering
        # The length filtering should be done at the dataset level before preprocessing
        print("⚠️  Note: Length filtering is now handled in RefSeqDataset")
        print("   The pipeline uses RefSeqPreprocessor which processes all sequences")
        print("   To filter by length, use RefSeqDataset first, then preprocess")

        # Example of how to do this properly:
        print("\n📋 Proper workflow:")
        print("1. Load RefSeqDataset with max_length filter")
        print("2. Extract sequences from dataset")
        print("3. Process with RefSeqPreprocessor")
        print("4. Generate embeddings")
        print("5. Train SAE")

        # For now, run the pipeline without length filtering
        pipeline = run_complete_pipeline(
            refseq_file="../data/vertebrate_mammalian.1.rna.gbff",
            max_samples=50,
            hidden_dim=1000,
            epochs=5,
            batch_size=4,
            filter_by_type="mRNA",
            use_cds=True,
            dataset_name="dataset_filtered_test"
        )

        print("✅ Pipeline completed successfully!")
        print(f"📁 Model saved to: {pipeline.model_save_dir}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Choose an example:")
    print("1. Demonstrate RefSeqDataset length filtering")
    print("2. Compare different length filters")
    print("3. Run pipeline (note: length filtering in dataset)")

    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        demonstrate_refseq_dataset_filtering()
    elif choice == "2":
        compare_dataset_length_filters()
    elif choice == "3":
        run_pipeline_with_dataset_filtering()
    else:
        print("Invalid choice. Running basic example...")
        demonstrate_refseq_dataset_filtering()
