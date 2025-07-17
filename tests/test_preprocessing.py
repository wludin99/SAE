#!/usr/bin/env python3
"""
Test script for the preprocessing module.
"""

import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all preprocessing modules can be imported."""
    print("Testing imports...")

    try:
        from sae.preprocessing import (
            CodonPreprocessor,
            HelicalWrapper,
            PreprocessingConfig,
            SequenceModelWrapper,
            create_helical_wrapper,
        )
        print("✓ All preprocessing imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_codon_preprocessor():
    """Test codon preprocessing functionality."""
    print("\nTesting codon preprocessor...")

    try:
        from sae.preprocessing import CodonPreprocessor

        # Create preprocessor
        preprocessor = CodonPreprocessor(start_token="E")

        # Test sequences
        sequences = [
            "ATGCGTACGTACGT",  # 15 bases = 5 codons
            "GCTAGCTAGCTAGC"   # 15 bases = 5 codons
        ]

        # Test validation
        for i, seq in enumerate(sequences):
            is_valid = preprocessor.validate_sequence(seq)
            print(f"  Sequence {i}: {seq} (valid: {is_valid})")

        # Test processing
        processed = preprocessor.process_sequences(sequences)
        print(f"  Processed sequences: {processed}")

        # Test statistics
        stats = preprocessor.get_codon_statistics(sequences)
        print(f"  Total codons: {stats['total_codons']}")
        print(f"  Unique codons: {stats['unique_codons']}")

        print("✓ Codon preprocessor tests passed")
        return True

    except Exception as e:
        print(f"✗ Codon preprocessor test failed: {e}")
        return False


def test_config():
    """Test configuration class."""
    print("\nTesting configuration...")

    try:
        from sae.preprocessing import PreprocessingConfig

        # Create config
        config = PreprocessingConfig(
            model_name="helical",
            device="cpu",
            batch_size=16,
            codon_start_token="E",
            add_codon_start=True
        )

        print(f"  Model name: {config.model_name}")
        print(f"  Device: {config.device}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Codon start token: {config.codon_start_token}")
        print(f"  Add codon start: {config.add_codon_start}")

        print("✓ Configuration tests passed")
        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_helical_wrapper_creation():
    """Test Helical wrapper creation (without loading model)."""
    print("\nTesting Helical wrapper creation...")

    try:
        from sae.preprocessing import create_helical_wrapper

        # Create wrapper (should not load model yet)
        wrapper = create_helical_wrapper(
            device="cpu",
            batch_size=16,
            codon_start_token="E",
            add_codon_start=True
        )

        print(f"  Wrapper created: {type(wrapper).__name__}")
        print(f"  Model initialized: {wrapper._is_initialized}")

        # Test model info before initialization
        info = wrapper.get_model_info()
        print(f"  Model info: {info}")

        print("✓ Helical wrapper creation tests passed")
        return True

    except Exception as e:
        print(f"✗ Helical wrapper creation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("SAE Preprocessing Module Tests")
    print("=" * 40)

    tests = [
        test_imports,
        test_codon_preprocessor,
        test_config,
        test_helical_wrapper_creation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
