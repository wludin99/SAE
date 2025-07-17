"""
RefSeq Parser Module

This module provides functionality to parse RefSeq GenBank format files
using BioPython's SeqIO and integrate them with the SAE pipeline.
"""

from typing import Any, Optional

import numpy as np
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset


class RefSeqDataset(Dataset):
    """
    PyTorch Dataset for RefSeq GenBank files
    """

    def __init__(self,
                 file_path: str,
                 max_length: Optional[int] = None,
                 encode_dna: bool = True,
                 filter_by_type: Optional[str] = None,
                 min_length: int = 100):
        """
        Initialize the RefSeq dataset

        Args:
            file_path: Path to the GenBank file
            max_length: Maximum sequence length (filters sequences longer than this)
            encode_dna: Whether to encode DNA sequences (A=0, C=1, G=2, T=3, N=4)
            filter_by_type: Filter by molecule type (e.g., 'mRNA', 'rRNA', 'tRNA')
            min_length: Minimum sequence length to include
        """
        self.file_path = file_path
        self.max_length = max_length
        self.encode_dna = encode_dna
        self.filter_by_type = filter_by_type
        self.min_length = min_length

        # Load and filter sequences
        self.sequences = self._load_sequences()

    def _load_sequences(self) -> list[dict[str, Any]]:
        """Load sequences from the GenBank file"""
        sequences = []

        print(f"Loading sequences from {self.file_path}...")

        with open(self.file_path) as handle:
            for record in SeqIO.parse(handle, "genbank"):
                # Filter by molecule type if specified
                if self.filter_by_type and record.annotations.get("molecule_type") != self.filter_by_type:
                    continue

                # Filter by length
                if len(record.seq) < self.min_length:
                    continue

                # Filter by maximum length if specified
                if self.max_length is not None and len(record.seq) > self.max_length:
                    continue

                # Extract sequence information
                seq_info = {
                    "id": record.id,
                    "name": record.name,
                    "description": record.description,
                    "sequence": str(record.seq),
                    "length": len(record.seq),
                    "molecule_type": record.annotations.get("molecule_type", "unknown"),
                    "organism": record.annotations.get("organism", "unknown"),
                    "taxonomy": record.annotations.get("taxonomy", []),
                    "features": len(record.features)
                }

                sequences.append(seq_info)

                # Progress indicator
                if len(sequences) % 1000 == 0:
                    print(f"Loaded {len(sequences)} sequences...")

        print(f"✅ Loaded {len(sequences)} sequences from {self.file_path}")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        sequence = item["sequence"]

        if self.encode_dna and isinstance(sequence, str):
            # DNA encoding: A=0, C=1, G=2, T=3, N=4
            encoding = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
            encoded = [encoding.get(base, 4) for base in sequence.upper()]

            # Truncate or pad to max_length
            if self.max_length is not None:
                if len(encoded) > self.max_length:
                    encoded = encoded[:self.max_length]
                else:
                    encoded.extend([4] * (self.max_length - len(encoded)))  # Pad with N

            sequence_tensor = torch.tensor(encoded, dtype=torch.long)
        else:
            # Return raw sequence as tensor
            if isinstance(sequence, str):
                sequence_tensor = torch.tensor([ord(c) for c in sequence], dtype=torch.long)
            else:
                sequence_tensor = torch.tensor(sequence, dtype=torch.float)

        return {
            "input_ids": sequence_tensor,
            "metadata": {
                "id": item["id"],
                "name": item["name"],
                "description": item["description"],
                "length": item["length"],
                "molecule_type": item["molecule_type"],
                "organism": item["organism"]
            }
        }


def load_refseq_dataset(
    file_path: str,
    max_samples: Optional[int] = None,
    max_length: Optional[int] = None,
    filter_by_type: Optional[str] = None,
    min_length: int = 100,
    encode_dna: bool = True
) -> RefSeqDataset:
    """
    Load a RefSeq GenBank file as a dataset

    Args:
        file_path: Path to the GenBank file
        max_samples: Maximum number of samples to load
        max_length: Maximum sequence length (filters sequences longer than this)
        filter_by_type: Filter by molecule type
        min_length: Minimum sequence length
        encode_dna: Whether to encode DNA sequences

    Returns:
        RefSeqDataset object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    dataset = RefSeqDataset(
        file_path=file_path,
        max_length=max_length,
        encode_dna=encode_dna,
        filter_by_type=filter_by_type,
        min_length=min_length
    )

    # Limit samples if specified
    if max_samples is not None and len(dataset) > max_samples:
        dataset.sequences = dataset.sequences[:max_samples]
        print(f"Limited to {max_samples} samples")

    return dataset


def create_refseq_dataloader(
    dataset: RefSeqDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Create a PyTorch DataLoader for RefSeq data

    Args:
        dataset: RefSeqDataset object
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        PyTorch DataLoader
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )

    return dataloader


def analyze_refseq_file(file_path: str) -> dict[str, Any]:
    """
    Analyze a RefSeq GenBank file and return statistics

    Args:
        file_path: Path to the GenBank file

    Returns:
        Dictionary containing analysis results
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Analyzing {file_path}...")

    stats = {
        "total_records": 0,
        "molecule_types": {},
        "organisms": {},
        "length_stats": {
            "min": float("inf"),
            "max": 0,
            "mean": 0,
            "median": 0
        },
        "lengths": []
    }

    with open(file_path) as handle:
        for record in SeqIO.parse(handle, "genbank"):
            stats["total_records"] += 1

            # Count molecule types
            mol_type = record.annotations.get("molecule_type", "unknown")
            stats["molecule_types"][mol_type] = stats["molecule_types"].get(mol_type, 0) + 1

            # Count organisms
            organism = record.annotations.get("organism", "unknown")
            stats["organisms"][organism] = stats["organisms"].get(organism, 0) + 1

            # Track lengths
            length = len(record.seq)
            stats["lengths"].append(length)
            stats["length_stats"]["min"] = min(stats["length_stats"]["min"], length)
            stats["length_stats"]["max"] = max(stats["length_stats"]["max"], length)

            # Progress indicator
            if stats["total_records"] % 1000 == 0:
                print(f"Processed {stats['total_records']} records...")

    # Calculate length statistics
    if stats["lengths"]:
        stats["length_stats"]["mean"] = np.mean(stats["lengths"])
        stats["length_stats"]["median"] = np.median(stats["lengths"])

    print(f"✅ Analysis complete: {stats['total_records']} records found")
    return stats


def print_refseq_analysis(analysis: dict[str, Any]):
    """
    Print a formatted analysis of RefSeq data

    Args:
        analysis: Analysis results from analyze_refseq_file
    """
    print("\n" + "="*60)
    print("REFSEQ FILE ANALYSIS")
    print("="*60)

    print(f"Total Records: {analysis['total_records']:,}")

    print("\nLength Statistics:")
    print(f"  Min: {analysis['length_stats']['min']:,} bp")
    print(f"  Max: {analysis['length_stats']['max']:,} bp")
    print(f"  Mean: {analysis['length_stats']['mean']:.1f} bp")
    print(f"  Median: {analysis['length_stats']['median']:.1f} bp")

    print("\nMolecule Types (top 10):")
    sorted_types = sorted(analysis["molecule_types"].items(),
                         key=lambda x: x[1], reverse=True)[:10]
    for mol_type, count in sorted_types:
        print(f"  {mol_type}: {count:,}")

    print("\nOrganisms (top 10):")
    sorted_orgs = sorted(analysis["organisms"].items(),
                        key=lambda x: x[1], reverse=True)[:10]
    for organism, count in sorted_orgs:
        print(f"  {organism}: {count:,}")

    print("="*60)


if __name__ == "__main__":
    # Example usage
    print("RefSeq Parser Example")
    print("="*50)

    # Example file path (update this to your actual file)
    example_file = "vertebrate_mammalian.1.rna.gbff"

    if os.path.exists(example_file):
        # Analyze the file
        analysis = analyze_refseq_file(example_file)
        print_refseq_analysis(analysis)

        # Load a small dataset
        print(f"\nLoading small dataset from {example_file}...")
        dataset = load_refseq_dataset(
            example_file,
            max_samples=100,
            max_length=1000,
            filter_by_type="mRNA"
        )

        print(f"✅ Loaded {len(dataset)} sequences")

        # Show a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSample sequence:")
            print(f"  ID: {sample['metadata']['id']}")
            print(f"  Description: {sample['metadata']['description']}")
            print(f"  Length: {sample['metadata']['length']} bp")
            print(f"  Tensor shape: {sample['input_ids'].shape}")

        # Create dataloader
        dataloader = create_refseq_dataloader(dataset, batch_size=4)

        for batch in dataloader:
            print(f"\nBatch shape: {batch['input_ids'].shape}")
            print(f"Batch metadata keys: {list(batch['metadata'].keys())}")
            break

    else:
        print(f"File {example_file} not found. Please update the file path.")

    print("\n" + "="*50)
    print("Example completed!")
