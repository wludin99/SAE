"""
Example: Loading and Analyzing RefSeq Data

This script demonstrates how to use the RefSeq parser to load and analyze
the vertebrate mammalian RNA data you downloaded.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


from sae.data import (
    analyze_refseq_file,
    create_refseq_dataloader,
    load_refseq_dataset,
    print_refseq_analysis,
)


def main():
    print("RefSeq Data Loading Example")
    print("="*60)

    # Path to your downloaded RefSeq file
    refseq_file = "../data/vertebrate_mammalian.1.rna.gbff"

    if not os.path.exists(refseq_file):
        print(f"❌ File not found: {refseq_file}")
        print("Please make sure you have downloaded the RefSeq file to the data/ directory")
        return

    print(f"📁 Found RefSeq file: {refseq_file}")

    # Step 1: Analyze the file to understand its contents
    print("\n🔍 Analyzing RefSeq file...")
    try:
        analysis = analyze_refseq_file(refseq_file)
        print_refseq_analysis(analysis)
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return

    # Step 2: Load a small subset for testing
    print("\n📥 Loading small dataset for testing...")
    try:
        dataset = load_refseq_dataset(
            file_path=refseq_file,
            max_samples=100,  # Only load 100 samples for testing
            max_length=1000,  # Limit sequence length to 1000 bp
            filter_by_type="mRNA",  # Only load mRNA sequences
            min_length=200,  # Minimum length of 200 bp
            encode_dna=True
        )

        print(f"✅ Loaded {len(dataset)} sequences")

        # Show some sample data
        if len(dataset) > 0:
            print("\n📋 Sample sequence information:")
            sample = dataset[0]
            print(f"  ID: {sample['metadata']['id']}")
            print(f"  Description: {sample['metadata']['description']}")
            print(f"  Organism: {sample['metadata']['organism']}")
            print(f"  Length: {sample['metadata']['length']} bp")
            print(f"  Tensor shape: {sample['input_ids'].shape}")
            print(f"  Tensor dtype: {sample['input_ids'].dtype}")

            # Show first few encoded bases
            first_bases = sample["input_ids"][:20].tolist()
            base_names = ["A", "C", "G", "T", "N"]
            decoded = [base_names[b] if b < 5 else "N" for b in first_bases]
            print(f"  First 20 bases: {''.join(decoded)}")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Step 3: Create a dataloader
    print("\n🔄 Creating dataloader...")
    try:
        dataloader = create_refseq_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # Use 0 for debugging
        )

        print(f"✅ Created dataloader with {len(dataloader)} batches")

        # Test the dataloader
        print("\n🧪 Testing dataloader...")
        for batch_idx, batch in enumerate(dataloader):
            print(f"  Batch {batch_idx + 1}:")
            print(f"    Input shape: {batch['input_ids'].shape}")
            print(f"    Metadata keys: {list(batch['metadata'].keys())}")
            print(f"    Sample IDs: {batch['metadata']['id'][:2]}...")  # Show first 2 IDs

            if batch_idx >= 2:  # Only show first 3 batches
                break

    except Exception as e:
        print(f"❌ Error with dataloader: {e}")
        return

    # Step 4: Show how to use with different filters
    print("\n🔧 Loading with different filters...")

    # Load only tRNA sequences
    try:
        trna_dataset = load_refseq_dataset(
            file_path=refseq_file,
            max_samples=50,
            filter_by_type="tRNA",
            min_length=50,
            max_length=200
        )
        print(f"  tRNA sequences: {len(trna_dataset)}")
    except Exception as e:
        print(f"  Error loading tRNA: {e}")

    # Load only rRNA sequences
    try:
        rrna_dataset = load_refseq_dataset(
            file_path=refseq_file,
            max_samples=50,
            filter_by_type="rRNA",
            min_length=100,
            max_length=5000
        )
        print(f"  rRNA sequences: {len(rrna_dataset)}")
    except Exception as e:
        print(f"  Error loading rRNA: {e}")

    print("\n" + "="*60)
    print("✅ Example completed successfully!")
    print("\nNext steps:")
    print("1. Adjust max_samples to load more data")
    print("2. Modify max_length based on your model requirements")
    print("3. Use different filter_by_type values to focus on specific RNA types")
    print("4. Integrate with your SAE training pipeline")


if __name__ == "__main__":
    main()
