"""
Example demonstrating the clean separation of concerns:
- RefSeqPreprocessor handles all preprocessing
- HelicalWrapper focuses only on model interaction
"""

import logging

from sae.preprocessing import create_helical_wrapper
from sae.preprocessing.refseq_preprocessor import RefSeqPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Demonstrate the clean architecture."""

    # 1. Create RefSeq preprocessor for all preprocessing
    logger.info("Creating RefSeq preprocessor...")
    refseq_preprocessor = RefSeqPreprocessor(codon_start_token="E")

    # 2. Create Helical wrapper for model interaction only
    logger.info("Creating Helical wrapper...")
    helical_wrapper = create_helical_wrapper(
        device="cuda",
        batch_size=32,
        max_length=1024
    )

    # 3. Example workflow
    logger.info("Example workflow:")
    logger.info("1. RefSeqPreprocessor processes GenBank files and adds codon tokens")
    logger.info("2. HelicalWrapper takes preprocessed sequences and generates embeddings")
    logger.info("3. Clean separation of concerns!")

    # Show component info
    logger.info(f"RefSeqPreprocessor codon token: {refseq_preprocessor.codon_start_token}")
    logger.info(f"HelicalWrapper model info: {helical_wrapper.get_model_info()}")

if __name__ == "__main__":
    main()
