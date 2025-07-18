"""
Embedding Generator Module

This module handles generating embeddings from genomic sequences using the pre-trained
HelicalmRNA model. It provides utilities for batch processing and caching embeddings.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Dict, List

import numpy as np
import torch

# Import helical components and our wrapper
try:
    from sae.preprocessing import (
        HelicalWrapper,
        PreprocessingConfig,
        create_helical_wrapper,
    )
    from sae.preprocessing.refseq_preprocessor import RefSeqPreprocessor
except ImportError:
    print("Warning: Helical not available. Install with: poetry add helical")
    HelicalWrapper = None
    create_helical_wrapper = None
    PreprocessingConfig = None
    RefSeqPreprocessor = None


class EmbeddingGenerator:
    """
    Generates embeddings from genomic sequences using HelicalmRNA model
    """

    def __init__(
        self,
        model_name: str = "helicalmRNA",
        device: Optional[str] = None,
        max_length: int = 1024,
        batch_size: int = 4,
        cache_dir: Optional[str] = "./outputs/embeddings_cache",
        codon_start_token: str = "E",
        normalize_embeddings: bool = False
    ):
        """
        Initialize the embedding generator

        Args:
            model_name: Name of the HelicalmRNA model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length for processing
            batch_size: Batch size for embedding generation
            cache_dir: Directory to cache embeddings
            codon_start_token: Token to prepend to each codon (default: 'E')
            normalize_embeddings: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./outputs/embeddings_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Wrapper configuration
        self.codon_start_token = codon_start_token
        self.normalize_embeddings = normalize_embeddings

        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.wrapper = None
        self.refseq_preprocessor = None
        self._load_model()

    def _load_model(self):
        """Load the HelicalmRNA model using the wrapper"""
        if create_helical_wrapper is None:
            raise ImportError("Helical not available. Please install with: poetry add helical")

        try:
            self.logger.info(f"Loading HelicalmRNA model with wrapper: {self.model_name}")
            self.wrapper = create_helical_wrapper(
                device=str(self.device),
                batch_size=self.batch_size,
                max_length=self.max_length,
                normalize_embeddings=self.normalize_embeddings
            )

            # Initialize RefSeq preprocessor
            self.refseq_preprocessor = RefSeqPreprocessor(
                codon_start_token=self.codon_start_token
            )

            self.logger.info("✅ Model and preprocessor loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def generate_embeddings_from_refseq(
        self,
        refseq_file: str,
        max_samples: Optional[int] = None,
        filter_by_type: str = "mRNA",
        use_cds: bool = True,
        layer_idx: Optional[int] = None,
        use_cache: bool = True,
        dataset_name: Optional[str] = None
    ) -> dict[str, np.ndarray]:
        """
        Generate embeddings from RefSeq GenBank file

        Args:
            refseq_file: Path to the RefSeq GenBank file
            max_samples: Maximum number of samples to process
            filter_by_type: Filter by molecule type (e.g., 'mRNA', 'rRNA', 'tRNA')
            use_cds: Whether to use CDS features for sequence extraction
            layer_idx: Specific layer to extract embeddings from
            use_cache: Whether to use cached embeddings
            dataset_name: Optional name for logging purposes

        Returns:
            Dictionary containing embeddings and metadata
        """
        if create_helical_wrapper is None:
            raise ImportError("Helical wrapper not available. Please install required dependencies.")

        # Check cache first
        if use_cache:
            cache_key = f"refseq_{Path(refseq_file).stem}_{filter_by_type}_{max_samples or 'all'}_{layer_idx or 'last'}"
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                self.logger.info(f"✅ Loaded RefSeq embeddings from cache: {cached_result['embeddings'].shape}")
                return cached_result

        # Use dataset_name for logging if provided
        log_name = dataset_name or f"RefSeq_{Path(refseq_file).stem}"

        # Process RefSeq file using RefSeqPreprocessor
        self.logger.info(f"Processing {log_name} from file: {refseq_file}")
        processed_records = self.refseq_preprocessor.process_refseq_file(
            file_path=refseq_file,
            max_samples=max_samples,
            filter_by_type=filter_by_type,
            use_cds=use_cds
        )

        if not processed_records:
            raise ValueError(f"No valid sequences found in {refseq_file}")

        # Extract sequences for embedding generation
        sequences = [record["final_sequence"] for record in processed_records]

        # Generate embeddings using the wrapper
        self.logger.info(f"Generating embeddings for {len(sequences)} {log_name} sequences...")
        embeddings = self.wrapper(sequences)

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Prepare result
        result = {
            "embeddings": embeddings,
            "metadata": {
                "num_sequences": len(sequences),
                "dataset_name": log_name,
                "refseq_file": refseq_file,
                "filter_by_type": filter_by_type,
                "use_cds": use_cds,
                "processed_records": processed_records
            }
        }

        # Cache result
        if use_cache:
            cache_key = f"refseq_{Path(refseq_file).stem}_{filter_by_type}_{max_samples or 'all'}_{layer_idx or 'last'}"
            self._save_to_cache(cache_key, result)

        self.logger.info(f"✅ Generated {log_name} embeddings shape: {embeddings.shape}")
        return result

    def generate_layer_embeddings_from_refseq(
        self,
        refseq_file: str,
        layer_name: str,
        max_samples: Optional[int] = None,
        filter_by_type: str = "mRNA",
        use_cds: bool = True,
        use_cache: bool = True,
        dataset_name: Optional[str] = None
    ) -> dict[str, np.ndarray]:
        """
        Generate embeddings from a specific layer of the Helical model.

        Args:
            refseq_file: Path to the RefSeq GenBank file
            layer_name: Name of the layer to extract from (see HelicalWrapper.LAYER_MAPPING)
            max_samples: Maximum number of samples to process
            filter_by_type: Filter by molecule type (e.g., 'mRNA', 'rRNA', 'tRNA')
            use_cds: Whether to use CDS features for sequence extraction
            use_cache: Whether to use cached embeddings
            dataset_name: Optional name for logging purposes

        Returns:
            Dictionary containing embeddings and metadata
        """
        if create_helical_wrapper is None:
            raise ImportError("Helical wrapper not available. Please install required dependencies.")

        # Check cache first
        if use_cache:
            cache_key = f"refseq_{Path(refseq_file).stem}_{filter_by_type}_{max_samples or 'all'}_{layer_name}"
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                self.logger.info(f"✅ Loaded {layer_name} embeddings from cache: {cached_result['embeddings'].shape}")
                return cached_result

        # Use dataset_name for logging if provided
        log_name = dataset_name or f"RefSeq_{Path(refseq_file).stem}"

        # Process RefSeq file using RefSeqPreprocessor
        self.logger.info(f"Processing {log_name} from file: {refseq_file}")
        processed_records = self.refseq_preprocessor.process_refseq_file(
            file_path=refseq_file,
            max_samples=max_samples,
            filter_by_type=filter_by_type,
            use_cds=use_cds
        )

        if not processed_records:
            raise ValueError(f"No valid sequences found in {refseq_file}")

        # Extract sequences for embedding generation
        sequences = [record["final_sequence"] for record in processed_records]

        # Generate embeddings from specific layer using the wrapper
        self.logger.info(f"Generating {layer_name} embeddings for {len(sequences)} {log_name} sequences...")
        embeddings = self.wrapper.get_layer_embeddings(sequences, layer_name)

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Prepare result
        result = {
            "embeddings": embeddings,
            "metadata": {
                "num_sequences": len(sequences),
                "dataset_name": log_name,
                "refseq_file": refseq_file,
                "filter_by_type": filter_by_type,
                "use_cds": use_cds,
                "layer_name": layer_name,
                "processed_records": processed_records
            }
        }

        # Cache result
        if use_cache:
            cache_key = f"refseq_{Path(refseq_file).stem}_{filter_by_type}_{max_samples or 'all'}_{layer_name}"
            self._save_to_cache(cache_key, result)

        self.logger.info(f"✅ Generated {layer_name} embeddings shape: {embeddings.shape}")
        return result

    def generate_all_layer_embeddings_from_refseq(
        self,
        refseq_file: str,
        max_samples: Optional[int] = None,
        filter_by_type: str = "mRNA",
        use_cds: bool = True,
        use_cache: bool = True,
        dataset_name: Optional[str] = None
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Generate embeddings from all available layers of the Helical model.

        Args:
            refseq_file: Path to the RefSeq GenBank file
            max_samples: Maximum number of samples to process
            filter_by_type: Filter by molecule type (e.g., 'mRNA', 'rRNA', 'tRNA')
            use_cds: Whether to use CDS features for sequence extraction
            use_cache: Whether to use cached embeddings
            dataset_name: Optional name for logging purposes

        Returns:
            Dictionary containing embeddings for all layers and metadata
        """
        if create_helical_wrapper is None:
            raise ImportError("Helical wrapper not available. Please install required dependencies.")

        # Check cache first
        if use_cache:
            cache_key = f"refseq_{Path(refseq_file).stem}_{filter_by_type}_{max_samples or 'all'}_all_layers"
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                self.logger.info(f"✅ Loaded all layer embeddings from cache")
                return cached_result

        # Use dataset_name for logging if provided
        log_name = dataset_name or f"RefSeq_{Path(refseq_file).stem}"

        # Process RefSeq file using RefSeqPreprocessor
        self.logger.info(f"Processing {log_name} from file: {refseq_file}")
        processed_records = self.refseq_preprocessor.process_refseq_file(
            file_path=refseq_file,
            max_samples=max_samples,
            filter_by_type=filter_by_type,
            use_cds=use_cds
        )

        if not processed_records:
            raise ValueError(f"No valid sequences found in {refseq_file}")

        # Extract sequences for embedding generation
        sequences = [record["final_sequence"] for record in processed_records]

        # Generate embeddings from all layers using the wrapper
        self.logger.info(f"Generating all layer embeddings for {len(sequences)} {log_name} sequences...")
        layer_embeddings = self.wrapper.get_all_layer_embeddings(sequences)

        # Convert to numpy
        numpy_embeddings = {}
        for layer_name, embeddings in layer_embeddings.items():
            if isinstance(embeddings, torch.Tensor):
                numpy_embeddings[layer_name] = embeddings.cpu().numpy()

        # Prepare result
        result = {
            "embeddings": numpy_embeddings,
            "metadata": {
                "num_sequences": len(sequences),
                "dataset_name": log_name,
                "refseq_file": refseq_file,
                "filter_by_type": filter_by_type,
                "use_cds": use_cds,
                "layers": list(numpy_embeddings.keys()),
                "processed_records": processed_records
            }
        }

        # Cache result
        if use_cache:
            cache_key = f"refseq_{Path(refseq_file).stem}_{filter_by_type}_{max_samples or 'all'}_all_layers"
            self._save_to_cache(cache_key, result)

        self.logger.info(f"✅ Generated embeddings for {len(numpy_embeddings)} layers")
        return result

    def get_available_layers(self) -> List[str]:
        """
        Get list of available layers for embedding extraction.

        Returns:
            List of available layer names
        """
        if self.wrapper is None:
            return []
        return list(self.wrapper.LAYER_MAPPING.keys())

    def _get_cache_key(self, sequences: list[str], layer_idx: Optional[int]) -> str:
        return f"embeddings_{hash(tuple(sequences))}_{layer_idx or 'last'}"

    def _save_to_cache(self, cache_key: str, result: dict[str, Any]):
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        self.logger.info(f"✅ Saved embeddings to cache: {cache_path}")

    def _load_from_cache(self, cache_key: str) -> Optional[dict[str, Any]]:
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                result = pickle.load(f)
            return result
        return None

    def get_model_info(self) -> dict[str, Any]:
        if self.wrapper is None:
            return {"status": "Model not loaded"}
        return self.wrapper.get_model_info()


if __name__ == "__main__":
    # Example usage
    print("Embedding Generator Example")
    print("=" * 50)

    # Test with a small dataset
    try:
        generator = EmbeddingGenerator()
        
        # Get available layers
        available_layers = generator.get_available_layers()
        print(f"Available layers: {available_layers}")
        
        # Example: Generate embeddings from a specific layer
        # result = generator.generate_layer_embeddings_from_refseq(
        #     refseq_file="path/to/your/refseq_file.gb",
        #     layer_name="after_mlp_1",
        #     max_samples=10,
        #     dataset_name="test_dataset"
        # )
        
        # Example: Generate embeddings from all layers
        # result = generator.generate_all_layer_embeddings_from_refseq(
        #     refseq_file="path/to/your/refseq_file.gb",
        #     max_samples=10,
        #     dataset_name="test_dataset"
        # )

        print("✅ Embedding generator initialized successfully")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Helical is installed: poetry add helical")
 