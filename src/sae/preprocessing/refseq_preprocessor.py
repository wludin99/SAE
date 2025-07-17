"""
RefSeq Preprocessor for Gene Feature Extraction and Codon Processing

This module provides preprocessing functionality specifically for RefSeq GenBank files,
extracting gene features and adding 'E' tokens at codon boundaries for Helical model input.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import logging

from .base import SequenceModelWrapper, PreprocessingConfig
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
    
    def extract_gene_features(self, record: SeqRecord) -> Dict[str, Any]:
        """
        Extract gene features from a GenBank record.
        
        Args:
            record: BioPython SeqRecord object
            
        Returns:
            Dictionary containing extracted gene features
        """
        features = {
            'id': record.id,
            'name': record.name,
            'description': record.description,
            'sequence': str(record.seq),
            'length': len(record.seq),
            'organism': record.annotations.get('organism', 'unknown'),
            'molecule_type': record.annotations.get('molecule_type', 'unknown'),
            'gene_features': [],
            'cds_features': [],
            'start_codons': [],
            'stop_codons': []
        }
        
        # Extract gene and CDS features
        for feature in record.features:
            if feature.type == 'gene':
                gene_info = {
                    'type': 'gene',
                    'location': str(feature.location),
                    'start': feature.location.start.position if feature.location.start else None,
                    'end': feature.location.end.position if feature.location.end else None,
                    'strand': feature.location.strand,
                    'qualifiers': dict(feature.qualifiers)
                }
                features['gene_features'].append(gene_info)
                
            elif feature.type == 'CDS':
                cds_info = {
                    'type': 'CDS',
                    'location': str(feature.location),
                    'start': feature.location.start.position if feature.location.start else None,
                    'end': feature.location.end.position if feature.location.end else None,
                    'strand': feature.location.strand,
                    'qualifiers': dict(feature.qualifiers),
                    'codon_start': feature.qualifiers.get('codon_start', [1])[0]
                }
                features['cds_features'].append(cds_info)
                
            elif feature.type == 'misc_feature':
                # Look for start/stop codons
                note = feature.qualifiers.get('note', [''])[0].lower()
                if 'start codon' in note or 'atg' in note:
                    features['start_codons'].append({
                        'location': str(feature.location),
                        'start': feature.location.start.position if feature.location.start else None,
                        'end': feature.location.end.position if feature.location.end else None
                    })
                elif 'stop codon' in note or 'stop' in note:
                    features['stop_codons'].append({
                        'location': str(feature.location),
                        'start': feature.location.start.position if feature.location.start else None,
                        'end': feature.location.end.position if feature.location.end else None
                    })
        
        return features
    
    def get_coding_sequence(self, features: Dict[str, Any]) -> Optional[str]:
        """
        Extract the coding sequence based on CDS features.
        
        Args:
            features: Gene features dictionary
            
        Returns:
            Coding sequence string or None if no CDS found
        """
        if not features['cds_features']:
            return None
        
        # Use the first CDS feature
        cds = features['cds_features'][0]
        start = cds['start']
        end = cds['end']
        
        if start is None or end is None:
            return None
        
        # Extract the coding sequence
        sequence = features['sequence']
        coding_seq = sequence[start:end]
        
        # Handle reverse strand
        if cds['strand'] == -1:
            coding_seq = str(Seq(coding_seq).reverse_complement())
        
        return coding_seq
    
    def add_codon_start_tokens(self, sequence: str, codon_start: int = 1) -> str:
        """
        Add 'E' tokens at the beginning of each codon.
        
        Args:
            sequence: DNA/RNA sequence
            codon_start: Position where coding starts (1-based)
            
        Returns:
            Sequence with 'E' tokens added at codon boundaries
        """
        if not sequence:
            return sequence
        
        # Adjust for codon_start (1-based to 0-based)
        start_offset = codon_start - 1
        if start_offset > 0:
            sequence = sequence[start_offset:]
        
        # Process with codon preprocessor
        processed_sequences = self.codon_preprocessor.process_sequences([sequence])
        return processed_sequences[0] if processed_sequences else sequence
    
    def process_refseq_record(self, record: SeqRecord, use_cds: bool = True) -> Dict[str, Any]:
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
        
        # Determine which sequence to use
        if use_cds and features['cds_features']:
            # Use coding sequence
            sequence = self.get_coding_sequence(features)
            if sequence:
                features['processed_sequence'] = sequence
                features['sequence_type'] = 'CDS'
                
                # Get codon start from CDS feature
                cds = features['cds_features'][0]
                codon_start = int(cds.get('codon_start', 1))
                
                # Add codon start tokens
                features['final_sequence'] = self.add_codon_start_tokens(sequence, codon_start)
            else:
                # Fallback to full sequence
                features['processed_sequence'] = features['sequence']
                features['sequence_type'] = 'full'
                features['final_sequence'] = self.add_codon_start_tokens(features['sequence'])
        else:
            # Use full sequence
            features['processed_sequence'] = features['sequence']
            features['sequence_type'] = 'full'
            features['final_sequence'] = self.add_codon_start_tokens(features['sequence'])
        
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
        
        # Check if sequence has proper codon structure (length divisible by 3)
        # Account for 'E' tokens at the beginning of each codon
        if len(sequence) % 4 != 0:  # Each codon is 4 characters: E + 3 bases
            return False
        
        # Check for valid characters
        valid_chars = set('EACGTUN')
        if not all(c in valid_chars for c in sequence.upper()):
            return False
        
        return True
    
    def get_processing_statistics(self, processed_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about processed records.
        
        Args:
            processed_records: List of processed record dictionaries
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total_records': len(processed_records),
            'valid_sequences': 0,
            'cds_sequences': 0,
            'full_sequences': 0,
            'length_stats': {
                'min': float('inf'),
                'max': 0,
                'mean': 0,
                'median': 0
            },
            'lengths': [],
            'organisms': {},
            'molecule_types': {}
        }
        
        for record in processed_records:
            if self.validate_sequence_for_helical(record['final_sequence']):
                stats['valid_sequences'] += 1
                
                length = len(record['final_sequence'])
                stats['lengths'].append(length)
                stats['length_stats']['min'] = min(stats['length_stats']['min'], length)
                stats['length_stats']['max'] = max(stats['length_stats']['max'], length)
                
                if record['sequence_type'] == 'CDS':
                    stats['cds_sequences'] += 1
                else:
                    stats['full_sequences'] += 1
                
                # Count organisms
                organism = record['organism']
                stats['organisms'][organism] = stats['organisms'].get(organism, 0) + 1
                
                # Count molecule types
                mol_type = record['molecule_type']
                stats['molecule_types'][mol_type] = stats['molecule_types'].get(mol_type, 0) + 1
        
        # Calculate length statistics
        if stats['lengths']:
            import numpy as np
            stats['length_stats']['mean'] = np.mean(stats['lengths'])
            stats['length_stats']['median'] = np.median(stats['lengths'])
        
        return stats


class RefSeqWrapper(SequenceModelWrapper):
    """
    Wrapper for RefSeq data processing with Helical model integration.
    """
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)
        self.refseq_preprocessor = RefSeqPreprocessor(
            codon_start_token=config.codon_start_token
        )
        self._helical_config = None
    
    def _load_model(self):
        """Load the Helical mRNA model."""
        try:
            from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig
            
            # Create Helical configuration
            self._helical_config = HelixmRNAConfig(
                batch_size=self.config.batch_size,
                device=self.config.device,
                max_length=self.config.max_length
            )
            
            # Initialize the model
            model = HelixmRNA(configurer=self._helical_config)
            logger.info(f"Helical model loaded successfully on {self.config.device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Helical model: {e}")
            raise
    
    def process_refseq_file(self, file_path: str, max_samples: Optional[int] = None, 
                           filter_by_type: Optional[str] = None, use_cds: bool = True) -> List[Dict[str, Any]]:
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
        
        with open(file_path, 'r') as handle:
            for i, record in enumerate(SeqIO.parse(handle, "genbank")):
                # Apply filters
                if filter_by_type and record.annotations.get('molecule_type') != filter_by_type:
                    continue
                
                # Process record
                try:
                    processed_record = self.refseq_preprocessor.process_refseq_record(record, use_cds=use_cds)
                    
                    # Validate for Helical
                    if self.refseq_preprocessor.validate_sequence_for_helical(processed_record['final_sequence']):
                        processed_records.append(processed_record)
                    
                    # Progress indicator
                    if len(processed_records) % 1000 == 0:
                        logger.info(f"Processed {len(processed_records)} valid records...")
                    
                    # Check max_samples
                    if max_samples and len(processed_records) >= max_samples:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing record {i}: {e}")
                    continue
        
        logger.info(f"✅ Processed {len(processed_records)} valid records from {file_path}")
        return processed_records
    
    def _preprocess_sequences(self, sequences: List[str]) -> Any:
        """
        Preprocess sequences for Helical model.
        
        Args:
            sequences: List of processed sequences with 'E' tokens
            
        Returns:
            Preprocessed data ready for Helical model
        """
        # Validate sequences
        valid_sequences = []
        for i, seq in enumerate(sequences):
            if self.refseq_preprocessor.validate_sequence_for_helical(seq):
                valid_sequences.append(seq)
            else:
                logger.warning(f"Sequence {i} is not valid for Helical model")
        
        if not valid_sequences:
            raise ValueError("No valid sequences found for processing")
        
        # Use Helical's built-in preprocessing
        try:
            processed_data = self.model.process_data(valid_sequences)
            return processed_data
        except Exception as e:
            logger.error(f"Helical preprocessing failed: {e}")
            raise
    
    def _get_embeddings(self, processed_data: Any) -> torch.Tensor:
        """
        Extract embeddings from processed data using Helical model.
        
        Args:
            processed_data: Data processed by Helical's process_data method
            
        Returns:
            Tensor of embeddings
        """
        try:
            embeddings = self.model.get_embeddings(processed_data)
            
            # Ensure embeddings are on the correct device
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.to(self.config.device)
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise


def create_refseq_wrapper(
    device: str = "cuda",
    batch_size: int = 32,
    max_length: Optional[int] = None,
    codon_start_token: str = "E",
    normalize_embeddings: bool = False
) -> RefSeqWrapper:
    """
    Convenience function to create a RefSeq wrapper with default settings.
    
    Args:
        device: Device to run the model on
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        codon_start_token: Token to prepend to each codon
        normalize_embeddings: Whether to normalize embeddings
        
    Returns:
        Configured RefSeqWrapper instance
    """
    config = PreprocessingConfig(
        model_name="helical",
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        codon_start_token=codon_start_token,
        add_codon_start=True,  # Always True for RefSeq processing
        normalize_embeddings=normalize_embeddings
    )
    
    return RefSeqWrapper(config) 