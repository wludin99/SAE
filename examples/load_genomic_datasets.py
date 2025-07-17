#!/usr/bin/env python3
"""
Example script demonstrating how to load smaller genomic datasets
from Hugging Face for development and testing.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sae.data.genomic_datasets import (
    create_genomic_dataloader,
    get_dataset_info,
    list_available_datasets,
    load_dna_promoters,
    load_drosophila_dna,
    load_genomic_dataset,
)


def main():
    print("Smaller Genomic Datasets Examples")
    print("=" * 50)

    # List all available datasets
    print("1. Available smaller genomic datasets:")
    datasets = list_available_datasets()
    for name, info in datasets.items():
        print(f"  📊 {name}")
        print(f"     Description: {info['description']}")
        print(f"     Size: {info['size']}, Samples: {info['samples']}")
        print()

    print("=" * 50)

    # Example 1: Human DNA dataset
    print("2. Loading Human DNA dataset (first 100 samples)...")
    try:
        human_dataset = load_genomic_dataset("human_dna", max_samples=100)
        print(f"✅ Loaded {len(human_dataset)} human DNA samples")
        print(f"Sample data: {human_dataset[0]}")

        # Create dataloader
        human_dataloader = create_genomic_dataloader(human_dataset, batch_size=8)
        print(f"Created dataloader with {len(human_dataloader)} batches")

        # Show first batch
        for batch in human_dataloader:
            print(f"Batch shape: {batch['input_ids'].shape}")
            print(f"Data type: {batch['input_ids'].dtype}")
            print(f"Value range: {batch['input_ids'].min().item()} to {batch['input_ids'].max().item()}")
            break

    except Exception as e:
        print(f"❌ Error loading human DNA: {e}")

    print("\n" + "=" * 50)

    # Example 2: Drosophila DNA dataset
    print("3. Loading Drosophila DNA dataset (first 50 samples)...")
    try:
        drosophila_dataset = load_genomic_dataset("drosophila_dna", max_samples=50)
        print(f"✅ Loaded {len(drosophila_dataset)} drosophila DNA samples")

        # Use convenience function
        drosophila_dataloader = load_drosophila_dna(max_samples=50, batch_size=4)
        print(f"Created dataloader with {len(drosophila_dataloader)} batches")

    except Exception as e:
        print(f"❌ Error loading drosophila DNA: {e}")

    print("\n" + "=" * 50)

    # Example 3: Small proteins dataset
    print("4. Loading Small Proteins dataset...")
    try:
        proteins_dataset = load_genomic_dataset("small_proteins", max_samples=100)
        print(f"✅ Loaded {len(proteins_dataset)} protein samples")
        print(f"Sample protein: {proteins_dataset[0]}")

        # Create dataloader (note: encode_dna=False for proteins)
        proteins_dataloader = create_genomic_dataloader(
            proteins_dataset,
            batch_size=8,
            encode_dna=False
        )
        print(f"Created protein dataloader with {len(proteins_dataloader)} batches")

    except Exception as e:
        print(f"❌ Error loading proteins: {e}")

    print("\n" + "=" * 50)

    # Example 4: DNA Promoters dataset
    print("5. Loading DNA Promoters dataset (first 50 samples)...")
    try:
        promoters_dataset = load_genomic_dataset("dna_promoters", max_samples=50)
        print(f"✅ Loaded {len(promoters_dataset)} promoter samples")

        # Use convenience function
        promoters_dataloader = load_dna_promoters(max_samples=50, batch_size=4)
        print(f"Created promoters dataloader with {len(promoters_dataloader)} batches")

    except Exception as e:
        print(f"❌ Error loading promoters: {e}")

    print("\n" + "=" * 50)

    # Example 5: Get detailed info about a dataset
    print("6. Getting detailed information about human DNA dataset...")
    try:
        info = get_dataset_info("human_dna")
        print(f"Dataset: {info['name']}")
        print(f"Path: {info['path']}")
        print(f"Split: {info['split']}")
        print(f"Description: {info['description']}")
        if "columns" in info:
            print(f"Columns: {info['columns']}")
        if "sample_data" in info and info["sample_data"]:
            print(f"Sample data keys: {list(info['sample_data'].keys())}")

    except Exception as e:
        print(f"❌ Error getting dataset info: {e}")

    print("\n" + "=" * 50)

    # Example 6: Compare different datasets
    print("7. Comparing sequence lengths across datasets...")
    try:
        datasets_to_test = ["human_dna", "drosophila_dna", "dna_promoters"]

        for dataset_name in datasets_to_test:
            print(f"\nTesting {dataset_name}...")
            dataset = load_genomic_dataset(dataset_name, max_samples=10)

            # Check sequence lengths
            lengths = []
            for i in range(min(5, len(dataset))):
                sequence = dataset[i]["sequence"]
                lengths.append(len(sequence))

            print(f"  Sequence lengths: {lengths}")
            print(f"  Average length: {sum(lengths) / len(lengths):.1f}")
            print(f"  Min/Max: {min(lengths)} / {max(lengths)}")

    except Exception as e:
        print(f"❌ Error comparing datasets: {e}")

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\n💡 Tips:")
    print("  - Use these smaller datasets for development and testing")
    print("  - Start with 100-1000 samples to test your models")
    print("  - Scale up to larger subsets as needed")
    print("  - Human DNA, Drosophila DNA, and DNA Promoters are good starting points")


if __name__ == "__main__":
    main()
