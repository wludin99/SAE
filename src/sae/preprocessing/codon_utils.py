"""
Codon preprocessing utilities for sequence foundation models.
"""

from typing import Optional

import torch


class CodonPreprocessor:
    """
    Utility class for codon-based preprocessing of DNA/RNA sequences.

    Handles the requirement that each codon should start with a specific token (e.g., 'E').
    """

    def __init__(self, start_token: str = "E", codon_length: int = 3):
        """
        Initialize codon preprocessor.

        Args:
            start_token: Token to prepend to each codon
            codon_length: Length of each codon (default: 3 for DNA/RNA)
        """
        self.start_token = start_token
        self.codon_length = codon_length

    def split_into_codons(self, sequence: str) -> list[str]:
        """
        Split a DNA/RNA sequence into codons.

        Args:
            sequence: Input DNA/RNA sequence

        Returns:
            List of codons
        """
        # Remove any whitespace and convert to uppercase
        sequence = sequence.replace(" ", "").upper()

        # Split into codons
        codons = []
        for i in range(0, len(sequence), self.codon_length):
            codon = sequence[i:i + self.codon_length]
            if len(codon) == self.codon_length:
                codons.append(codon)

        return codons

    def add_codon_start_tokens(self, sequence: str) -> str:
        """
        Add start tokens to each codon in a sequence.

        Args:
            sequence: Input DNA/RNA sequence

        Returns:
            Sequence with start tokens added before each codon
        """
        codons = self.split_into_codons(sequence)
        # Add start token before each codon, not between them
        return "".join([self.start_token + codon for codon in codons])

    def process_sequences(self, sequences: list[str]) -> list[str]:
        """
        Process a list of sequences by adding codon start tokens.

        Args:
            sequences: List of DNA/RNA sequences

        Returns:
            List of processed sequences with codon start tokens
        """
        return [self.add_codon_start_tokens(seq) for seq in sequences]

    def extract_codons_with_positions(self, sequence: str) -> list[tuple[str, int]]:
        """
        Extract codons with their positions in the original sequence.

        Args:
            sequence: Input DNA/RNA sequence

        Returns:
            List of (codon, position) tuples
        """
        sequence = sequence.replace(" ", "").upper()
        codons = []

        for i in range(0, len(sequence), self.codon_length):
            codon = sequence[i:i + self.codon_length]
            if len(codon) == self.codon_length:
                codons.append((codon, i))

        return codons

    def validate_sequence(self, sequence: str) -> bool:
        """
        Validate if a sequence can be properly split into codons.

        Args:
            sequence: Input DNA/RNA sequence

        Returns:
            True if sequence is valid for codon processing
        """
        sequence = sequence.replace(" ", "").upper()
        return len(sequence) % self.codon_length == 0

    def get_codon_statistics(self, sequences: list[str]) -> dict:
        """
        Get statistics about codons in a list of sequences.

        Args:
            sequences: List of DNA/RNA sequences

        Returns:
            Dictionary with codon statistics
        """
        all_codons = []
        total_sequences = len(sequences)
        total_length = 0

        for seq in sequences:
            codons = self.split_into_codons(seq)
            all_codons.extend(codons)
            total_length += len(seq.replace(" ", ""))

        codon_counts = {}
        for codon in all_codons:
            codon_counts[codon] = codon_counts.get(codon, 0) + 1

        return {
            "total_sequences": total_sequences,
            "total_codons": len(all_codons),
            "unique_codons": len(codon_counts),
            "average_sequence_length": total_length / total_sequences if total_sequences > 0 else 0,
            "codon_frequencies": codon_counts
        }


def create_codon_aware_tokenizer(vocab: dict, start_token: str = "E") -> dict:
    """
    Create a codon-aware tokenizer vocabulary.

    Args:
        vocab: Original vocabulary
        start_token: Token to use for codon starts

    Returns:
        Updated vocabulary with codon-aware tokens
    """
    # Add start token to vocabulary if not present
    if start_token not in vocab:
        vocab[start_token] = len(vocab)

    return vocab


def batch_codon_preprocessing(sequences: list[str],
                            start_token: str = "E",
                            max_length: Optional[int] = None) -> torch.Tensor:
    """
    Batch process sequences with codon preprocessing.

    Args:
        sequences: List of DNA/RNA sequences
        start_token: Token to prepend to each codon
        max_length: Maximum sequence length (None for no limit)

    Returns:
        Tensor of processed sequences
    """
    preprocessor = CodonPreprocessor(start_token=start_token)
    processed_sequences = preprocessor.process_sequences(sequences)

    # Pad sequences to same length if max_length is specified
    if max_length is not None:
        # Implementation would depend on the specific tokenizer being used
        # This is a placeholder for the actual implementation
        pass

    # Convert to tensor (this is a simplified version)
    # In practice, you'd use the model's tokenizer here
    return torch.tensor([[ord(c) for c in seq] for seq in processed_sequences])
