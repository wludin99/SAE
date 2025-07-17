"""
RefSeq Preprocessor for Gene Feature Extraction and Codon Processing

This module provides preprocessing functionality specifically for RefSeq GenBank files,
extracting gene features and adding 'E' tokens at codon boundaries for Helical model input.
"""

import logging
from typing import Any, Optional

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

from .codon_utils import CodonPreprocessor

logger = logging.getLogger(__name__)


class RefSeqPreprocessor:
    """
    Preprocessor for RefSeq GenBank files that extracts gene features
    and processes sequences for Helical model input.
    """

    def __init__(self, codon_start_token: str = "E"):
        """
        Initialize the RefSeq preprocessor.

        Args:
            codon_start_token: Token to prepend to each codon (default: 'E')
        """
        self.codon_start_token = codon_start_token
        self.codon_preprocessor = CodonPreprocessor(start_token=codon_start_token)

    def extract_gene_features(self, record: SeqRecord) -> dict[str, Any]:
        """
        Extract gene features from a GenBank record.

        Args:
            record: BioPython SeqRecord object

        Returns:
            Dictionary containing extracted gene features
        """
        features = {
            "id": record.id,
            "name": record.name,
            "description": record.description,
            "sequence": str(record.seq),
            "length": len(record.seq),
            "organism": record.annotations.get("organism", "unknown"),
            "molecule_type": record.annotations.get("molecule_type", "unknown"),
            "gene_features": [],
            "cds_features": [],
            "start_codons": [],
            "stop_codons": []
        }

        # Extract gene and CDS features
        for feature in record.features:
            if feature.type == "gene":
                gene_info = {
                    "type": "gene",
                    "location": str(feature.location),
                    "start": feature.location.start.real if feature.location.start else None,
                    "end": feature.location.end.real if feature.location.end else None,
                    "strand": feature.location.strand,
                    "qualifiers": dict(feature.qualifiers)
                }
                features["gene_features"].append(gene_info)

            elif feature.type == "CDS":
                cds_info = {
                    "type": "CDS",
                    "location": str(feature.location),
                    "start": feature.location.start.real if feature.location.start else None,
                    "end": feature.location.end.real if feature.location.end else None,
                    "strand": feature.location.strand,
                    "qualifiers": dict(feature.qualifiers),
                    "codon_start": feature.qualifiers.get("codon_start", [1])[0]
                }
                features["cds_features"].append(cds_info)

            elif feature.type == "misc_feature":
                # Look for start/stop codons
                note = feature.qualifiers.get("note", [""])[0].lower()
                if "start codon" in note or "atg" in note:
                    features["start_codons"].append({
                        "location": str(feature.location),
                        "start": feature.location.start.real if feature.location.start else None,
                        "end": feature.location.end.real if feature.location.end else None
                    })
                elif "stop codon" in note or "stop" in note:
                    features["stop_codons"].append({
                        "location": str(feature.location),
                        "start": feature.location.start.real if feature.location.start else None,
                        "end": feature.location.end.real if feature.location.end else None
                    })

        return features

    def get_coding_sequence(self, features: dict[str, Any]) -> Optional[str]:
        """
        Extract the coding sequence based on CDS features.

        Args:
            features: Gene features dictionary

        Returns:
            Coding sequence string or None if no CDS found
        """
        if not features["cds_features"]:
            return None

        # Use the first CDS feature
        cds = features["cds_features"][0]
        start = cds["start"]
        end = cds["end"]

        if start is None or end is None:
            return None

        # Extract the coding sequence
        sequence = features["sequence"]
        coding_seq = sequence[start:end]

        # Handle reverse strand
        if cds["strand"] == -1:
            coding_seq = str(Seq(coding_seq).reverse_complement())

        return coding_seq



    def add_codon_start_tokens_to_coding_region(self, full_sequence: str, cds_start: int, cds_end: int,
                                               codon_start: int = 1) -> str:
        """
        Add 'E' tokens before every codon in the coding region only, keeping the full sequence.

        Args:
            full_sequence: Complete DNA/RNA sequence
            cds_start: Start position of CDS (0-based)
            cds_end: End position of CDS (0-based)
            codon_start: Position where coding starts within CDS (1-based)

        Returns:
            Full sequence with 'E' tokens added only in the coding region
        """
        if not full_sequence or cds_start is None or cds_end is None:
            return full_sequence

        # Convert T's to U's in the full sequence
        full_sequence = full_sequence.replace("T", "U").replace("t", "u")

        # Extract the coding region
        coding_region = full_sequence[cds_start:cds_end]

        # Adjust for codon_start within the coding region
        start_offset = codon_start - 1
        if start_offset > 0:
            coding_region = coding_region[start_offset:]

        # Add 'E' tokens before each codon in the coding region
        processed_coding_region = self.codon_preprocessor.add_codon_start_tokens(coding_region)

        # Reconstruct the full sequence with processed coding region
        result = (
            full_sequence[:cds_start] +  # Before CDS
            processed_coding_region +     # Processed coding region
            full_sequence[cds_end:]       # After CDS
        )

        return result

    def process_refseq_record(self, record: SeqRecord, use_cds: bool = True) -> dict[str, Any]:
        """
        Process a single RefSeq record, extracting features and preparing sequence.

        Args:
            record: BioPython SeqRecord object
            use_cds: Whether to use CDS features for sequence extraction

        Returns:
            Dictionary with processed sequence and metadata
        """
        # Extract features
        features = self.extract_gene_features(record)

        # Convert T's to U's in the original sequence
        original_sequence = features["sequence"]
        features["sequence"] = original_sequence.replace("T", "U").replace("t", "u")

        # Determine which sequence to use
        if use_cds and features["cds_features"]:
            # Use CDS features to process coding region
            cds = features["cds_features"][0]
            cds_start = cds["start"]
            cds_end = cds["end"]
            codon_start = int(cds.get("codon_start", 1))

            if cds_start is not None and cds_end is not None:
                # Process coding region while keeping full sequence
                features["sequence_type"] = "CDS_annotated"
                features["final_sequence"] = self.add_codon_start_tokens_to_coding_region(
                    features["sequence"], cds_start, cds_end, codon_start
                )
            else:
                # Fallback to full sequence with codon tokens
                features["sequence_type"] = "full"
                features["final_sequence"] = self.codon_preprocessor.add_codon_start_tokens(features["sequence"])
        else:
            # Use full sequence with codon tokens
            features["sequence_type"] = "full"
            features["final_sequence"] = self.codon_preprocessor.add_codon_start_tokens(features["sequence"])

        return features

    def validate_sequence_for_helical(self, sequence: str) -> bool:
        """
        Validate if a sequence is suitable for Helical model input.

        Args:
            sequence: Processed sequence

        Returns:
            True if sequence is valid
        """
        if not sequence:
            return False

        # Check for valid characters (now including U instead of T)
        valid_chars = set("EACGUN")
        if not all(c in valid_chars for c in sequence.upper()):
            return False

        # For sequences with CDS annotation, we don't require strict codon structure
        # since only the coding region has 'E' tokens
        return True

    def process_refseq_file(self, file_path: str, max_samples: Optional[int] = None,
                           filter_by_type: Optional[str] = None, use_cds: bool = True) -> list[dict[str, Any]]:
        """
        Process a RefSeq GenBank file and prepare sequences for Helical model.

        Args:
            file_path: Path to the GenBank file
            max_samples: Maximum number of samples to process
            filter_by_type: Filter by molecule type
            use_cds: Whether to use CDS features for sequence extraction

        Returns:
            List of processed record dictionaries
        """
        processed_records = []

        logger.info(f"Processing RefSeq file: {file_path}")

        # First, count total records for progress bar
        total_records = 0
        with open(file_path) as handle:
            for _ in SeqIO.parse(handle, "genbank"):
                total_records += 1
                if max_samples and total_records >= max_samples * 2:  # Rough estimate
                    break

        logger.info(f"Found approximately {total_records} records to process")

        with open(file_path) as handle:
            # Use tqdm for progress tracking
            for i, record in enumerate(tqdm(SeqIO.parse(handle, "genbank"),
                                          total=total_records,
                                          desc="Processing records",
                                          unit="record")):
                # Apply filters
                if filter_by_type and record.annotations.get("molecule_type") != filter_by_type:
                    continue

                # Process record
                try:
                    processed_record = self.process_refseq_record(record, use_cds=use_cds)

                    # Validate for Helical
                    if self.validate_sequence_for_helical(processed_record["final_sequence"]):
                        processed_records.append(processed_record)

                    # Check max_samples
                    if max_samples and len(processed_records) >= max_samples:
                        break

                except Exception as e:
                    logger.warning(f"Error processing record {i}: {e}")
                    continue

        logger.info(f"✅ Processed {len(processed_records)} valid records from {file_path}")
        return processed_records

    def get_processing_statistics(self, processed_records: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Get statistics about processed records.

        Args:
            processed_records: List of processed record dictionaries

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            "total_records": len(processed_records),
            "valid_sequences": 0,
            "cds_sequences": 0,
            "full_sequences": 0,
            "length_stats": {
                "min": float("inf"),
                "max": 0,
                "mean": 0,
                "median": 0
            },
            "lengths": [],
            "organisms": {},
            "molecule_types": {}
        }

        for record in processed_records:
            if self.validate_sequence_for_helical(record["final_sequence"]):
                stats["valid_sequences"] += 1

                length = len(record["final_sequence"])
                stats["lengths"].append(length)
                stats["length_stats"]["min"] = min(stats["length_stats"]["min"], length)
                stats["length_stats"]["max"] = max(stats["length_stats"]["max"], length)

                if record["sequence_type"] == "CDS_annotated":
                    stats["cds_sequences"] += 1
                elif record["sequence_type"] == "CDS":
                    stats["cds_sequences"] += 1
                else:
                    stats["full_sequences"] += 1

                # Count organisms
                organism = record["organism"]
                stats["organisms"][organism] = stats["organisms"].get(organism, 0) + 1

                # Count molecule types
                mol_type = record["molecule_type"]
                stats["molecule_types"][mol_type] = stats["molecule_types"].get(mol_type, 0) + 1

        # Calculate length statistics
        if stats["lengths"]:
            stats["length_stats"]["mean"] = np.mean(stats["lengths"])
            stats["length_stats"]["median"] = np.median(stats["lengths"])

        return stats
