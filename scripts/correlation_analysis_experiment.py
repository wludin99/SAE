#!/usr/bin/env python3
"""
Correlation Analysis Experiment: SAE Sparse Activations vs Biological Features

This script uses the SAE pipeline to analyze correlations between
sparse high-dimensional activations and biological features from GenBank records.
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

warnings.filterwarnings("ignore")

import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("⚠️  Joblib not available. Install with: pip install joblib")

# Add the src directory to the path
script_dir = Path(__file__).parent
src_path = script_dir.parent / "src"
sys.path.append(str(src_path))

from sae.metrics.correlation_analysis import cuda_calculate_correlation_matrix
from sae.pipeline import SAETrainingPipeline
from sae.preprocessing.refseq_preprocessor import RefSeqPreprocessor


class CorrelationAnalysisExperiment:
    """
    Experiment to analyze correlations between SAE sparse activations and biological features.
    """

    def __init__(
        self,
        model_path: str,
        refseq_file: str,
        output_dir: str = "./correlation_results",
        max_samples: int = 100,
        batch_size: int = 1,
        max_length: int = 1024
    ):
        """
        Initialize the correlation analysis experiment.
        """
        self.model_path = Path(model_path)
        self.refseq_file = refseq_file
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.max_length = max_length

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.preprocessor = RefSeqPreprocessor()
        self.pipeline = None

        print("🔬 Correlation Analysis Experiment")
        print(f"   Model path: {self.model_path}")
        print(f"   RefSeq file: {self.refseq_file}")
        print(f"   Max samples: {self.max_samples}")
        print(f"   Max length: {self.max_length}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Output directory: {self.output_dir}")
        print()

    def load_pipeline(self) -> SAETrainingPipeline:
        """
        Load the pre-trained SAE pipeline.
        """
        print("📥 Loading pre-trained SAE pipeline...")

        # Load the pipeline from the saved directory
        self.pipeline = SAETrainingPipeline.load_from_checkpoint(self.model_path)

        # Update embedding generator configuration for memory efficiency
        if hasattr(self.pipeline, "embedding_generator"):
            # Update the embedding generator's configuration
            self.pipeline.embedding_generator.batch_size = self.batch_size
            self.pipeline.embedding_generator.max_length = self.max_length
            print(f"   🔧 Updated embedding batch size to: {self.batch_size}")
            print(f"   🔧 Updated embedding max_length to: {self.max_length}")

            # Re-load the model with new configuration
            self.pipeline.embedding_generator._load_model()
            print("   ✅ Re-loaded embedding generator with new config")

        print("   ✅ Pipeline loaded successfully")
        print(f"   Model: {self.pipeline.sae_model.__class__.__name__}")
        print(f"   Device: {self.pipeline.device}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Max length: {self.max_length}")
        print()

        return self.pipeline

    def process_refseq_data(self) -> list[dict[str, Any]]:
        """
        Process RefSeq data and extract biological features.
        """
        print("📊 Processing RefSeq data...")

        processed_records = self.preprocessor.process_refseq_file(
            file_path=self.refseq_file,
            max_samples=self.max_samples,
            use_cds=True,
            max_length=self.max_length
        )

        print(f"   ✅ Processed {len(processed_records)} records")

        # Print feature summary
        feature_stats = self.preprocessor.get_processing_statistics(processed_records)
        print("   📋 Feature Summary:")
        for feature_type, count in feature_stats.get("feature_counts", {}).items():
            print(f"     {feature_type}: {count}")

        # Print alanine statistics
        total_alanine = sum(record["feature_statistics"].get("alanine_count", 0) for record in processed_records)
        records_with_alanine = sum(1 for record in processed_records if record["feature_statistics"].get("has_alanine", False))
        print("   🧬 Alanine Statistics:")
        print(f"     Total alanine positions: {total_alanine}")
        print(f"     Records with alanine: {records_with_alanine}")
        print()

        return processed_records

    def generate_activations_and_alanine_masks(
        self,
        processed_records: list[dict[str, Any]]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Generate sparse activations and alanine position masks using the pipeline.
        """
        print("🧠 Generating sparse activations...")

        # Clear GPU cache before generating embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   🧹 Cleared CUDA cache")

        # Extract sequences
        sequences = [record["final_sequence"] for record in processed_records]

        # Use pipeline to generate embeddings and activations
        embeddings = self.pipeline.embedding_generator.wrapper(sequences)

        # Convert to numpy if it's a tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Use forward method and extract latent activations (encoded)
        with torch.no_grad():
            # Pass embeddings directly to SAE model (preserving sequence length)
            reconstructed, activations = self.pipeline.sae_model.forward(torch.tensor(embeddings, device=self.pipeline.device))
            activations = activations.cpu().numpy()

        # Extract alanine masks from preprocessor
        alanine_masks = []
        alanine_count = 0
        for record in processed_records:
            masks = record["feature_masks"]
            alanine_masks.append(masks["alanine"])
            alanine_count += masks["alanine"].sum()

        print(f"   ✅ Generated activations: {activations.shape}")
        print(f"   ✅ Collected {len(alanine_masks)} alanine masks")
        print(f"   ✅ Found {alanine_count} total alanine positions")

        # Debug: Check alanine distribution
        alanine_per_sequence = [mask.sum() for mask in alanine_masks]
        print(f"   🔍 Alanine per sequence: min={min(alanine_per_sequence)}, max={max(alanine_per_sequence)}, mean={np.mean(alanine_per_sequence):.2f}")
        print(f"   🔍 Sequences with alanine: {sum(1 for count in alanine_per_sequence if count > 0)}/{len(alanine_per_sequence)}")
        print()

        return activations, alanine_masks

    def calculate_correlations(
        self,
        activations: np.ndarray,
        alanine_masks: list[np.ndarray]
    ) -> tuple[dict, dict]:
        """
        Calculate correlations between sparse activations and alanine positions.
        Optimized version with CUDA tensor operations and parallel processing fallback.
        """
        print("📈 Calculating alanine position correlations...")

        # Handle both 2D and 3D activations
        if len(activations.shape) == 3:
            # 3D activations: (n_sequences, seq_len, n_latents)
            n_sequences, seq_len, n_latents = activations.shape
            # Reshape to (n_sequences * seq_len, n_latents) for processing
            activations_2d = activations.reshape(-1, n_latents)
        else:
            # 2D activations: (n_sequences, n_latents)
            n_sequences, n_latents = activations.shape
            activations_2d = activations
            seq_len = 1

        # Pre-process alanine masks into a single array for vectorized operations
        print("   🔄 Preprocessing alanine masks...")

        # Use the actual sequence length from activations, not from masks
        if len(activations.shape) == 3:
            # 3D activations: (n_sequences, seq_len, n_latents)
            actual_seq_len = activations.shape[1]
        else:
            # 2D activations: (n_sequences, n_latents) - no sequence dimension
            actual_seq_len = 1

        print(f"   🔍 Activations shape: {activations.shape}")
        print(f"   🔍 Using sequence length: {actual_seq_len}")

        # Add feature dimension to make it compatible with CUDA function: (n_sequences, 1, actual_seq_len)
        feature_array = np.zeros((n_sequences, 1, actual_seq_len))

        for seq_idx, mask in enumerate(alanine_masks):
            seq_len = min(mask.shape[0], actual_seq_len)  # Don't exceed actual sequence length
            feature_array[seq_idx, 0, :seq_len] = mask[:seq_len]

        print(f"   🔍 Feature array shape: {feature_array.shape}")
        print(f"   🔍 Total alanine positions in feature array: {feature_array.sum()}")
        print(f"   🔍 Non-zero entries: {np.count_nonzero(feature_array)}")

        # Try CUDA tensor operations first (fastest)
        try:
            if torch.cuda.is_available():
                print("   🚀 Using CUDA tensor operations...")
                pearson_corr, pearson_p = cuda_calculate_correlation_matrix(
                    activations_2d, feature_array
                )

                # Convert results to dictionary format
                correlations = {}
                p_values = {}

                for latent_idx in range(n_latents):
                    if pearson_corr[latent_idx, 0] != 0:  # Only include valid correlations
                        correlations[(latent_idx, "alanine")] = {
                            "pearson": float(pearson_corr[latent_idx, 0]),
                            "pearson_p": float(pearson_p[latent_idx, 0])
                        }
                        p_values[(latent_idx, "alanine")] = float(pearson_p[latent_idx, 0])

                print("   ✅ CUDA calculation completed")
                return correlations, p_values

        except Exception as e:
            print(f"   ⚠️  CUDA calculation failed: {e}")
            print("   🔄 Falling back to parallel processing...")

        # Fall back to parallel processing method
        print("   🔄 Using parallel processing fallback...")
        # Create all combinations of latent-alanine pairs for parallel processing
        latent_feature_pairs = []
        for latent_idx in range(n_latents):
            latent_feature_pairs.append((latent_idx, 0, "alanine"))

        print(f"   🔄 Processing {len(latent_feature_pairs)} latent-alanine pairs...")

        # Use parallel processing for correlation calculations
        if JOBLIB_AVAILABLE:
            try:
                n_jobs = min(multiprocessing.cpu_count(), len(latent_feature_pairs))
                print(f"   🔄 Using {n_jobs} parallel workers...")

                # For fallback processing, we need to extract the 2D alanine array
                alanine_array_2d = feature_array[:, 0, :]  # Shape: (n_sequences, max_seq_len)

                results = Parallel(n_jobs=n_jobs, verbose=1)(
                    delayed(self._calculate_single_correlation)(
                        latent_idx, feature_idx, feature_type, activations_2d, alanine_array_2d
                    )
                    for latent_idx, feature_idx, feature_type in latent_feature_pairs
                )

            except Exception as e:
                print(f"   ⚠️  Parallel processing failed: {e}")
                print("   🔄 Using sequential processing...")
                results = []
                # For sequential processing, we need to extract the 2D alanine array
                alanine_array_2d = feature_array[:, 0, :]  # Shape: (n_sequences, max_seq_len)
                for latent_idx, feature_idx, feature_type in tqdm(latent_feature_pairs, desc="Calculating correlations"):
                    result = self._calculate_single_correlation(
                        latent_idx, feature_idx, feature_type, activations_2d, alanine_array_2d
                    )
                    results.append(result)
        else:
            print("   ⚠️  joblib not available, using sequential processing...")
            results = []
            # For sequential processing, we need to extract the 2D alanine array
            alanine_array_2d = feature_array[:, 0, :]  # Shape: (n_sequences, max_seq_len)
            for latent_idx, feature_idx, feature_type in tqdm(latent_feature_pairs, desc="Calculating correlations"):
                result = self._calculate_single_correlation(
                    latent_idx, feature_idx, feature_type, activations_2d, alanine_array_2d
                )
                results.append(result)

        # Process results
        correlations = {}
        p_values = {}

        for result in results:
            if result is not None:
                latent_idx, feature_type, corr_data, p_val = result
                correlations[(latent_idx, feature_type)] = corr_data
                p_values[(latent_idx, feature_type)] = p_val

        print(f"   ✅ Calculated correlations for {len(correlations)} latent-alanine pairs")
        print()

        return correlations, p_values

    def _calculate_single_correlation(
        self,
        latent_idx: int,
        feature_idx: int,
        feature_type: str,
        activations: np.ndarray,
        feature_array: np.ndarray
    ) -> Optional[tuple[int, str, dict, float]]:
        """
        Calculate correlation for a single latent-alanine pair.
        Optimized for vectorized operations.
        """
        try:
            # Get latent activations for this dimension
            latent_activations = activations[:, latent_idx]  # Shape: (n_sequences,)

            # Get alanine values (feature_array is now 2D: n_sequences, seq_len)
            feature_values = feature_array  # Shape: (n_sequences, max_seq_len)

            # Create data points: for each sequence, repeat the activation value for each token
            # This is much more efficient than nested loops
            activation_values = np.repeat(latent_activations, feature_values.shape[1])
            feature_values_flat = feature_values.flatten()

            # Remove any padding (zeros from shorter sequences)
            valid_mask = feature_values_flat != 0  # Assuming 0 is padding
            if valid_mask.sum() < 5:  # Reduced minimum data points for sparse features
                if latent_idx == 0:  # Debug only for first latent
                    print(f"   🔍 Latent {latent_idx}: Only {valid_mask.sum()} valid data points, skipping")
                return None

            activation_values = activation_values[valid_mask]
            feature_values_flat = feature_values_flat[valid_mask]

            if latent_idx == 0:  # Debug only for first latent
                print(f"   🔍 Latent {latent_idx}: {valid_mask.sum()} valid data points, alanine values: {feature_values_flat.sum()}")

            # Calculate correlations
            corr, p_val = pearsonr(activation_values, feature_values_flat)

            corr_data = {
                "pearson": corr,
                "pearson_p": p_val
            }

            return (latent_idx, feature_type, corr_data, p_val)

        except Exception as e:
            print(f"   ⚠️  Error calculating correlation for latent {latent_idx}, feature {feature_type}: {e}")
            return None

    def perform_statistical_analysis(self, p_values: dict) -> dict:
        """
        Perform statistical analysis including multiple testing correction.
        """
        print("📊 Performing statistical analysis...")

        pval_list = list(p_values.values())
        keys = list(p_values.keys())

        if len(pval_list) == 0:
            print("   ⚠️  No p-values to analyze")
            return {}

        try:
            _, corrected_pvals, _, _ = multipletests(pval_list, method="fdr_bh", alpha=0.05)
            corrected_dict = dict(zip(keys, corrected_pvals, strict=False))

            significant = {key: p_val for key, p_val in corrected_dict.items() if p_val < 0.05}

            print(f"   ✅ Corrected {len(pval_list)} p-values")
            print(f"   ✅ Found {len(significant)} significant correlations")
            print()

            return {
                "corrected_p_values": corrected_dict,
                "significant_features": significant,
                "total_tests": len(pval_list)
            }

        except Exception as e:
            print(f"   ❌ Error in statistical analysis: {e}")
            return {}

    def create_visualizations(self, correlations: dict, activations: np.ndarray, alanine_masks: list[np.ndarray]):
        """
        Create visualizations of the alanine position correlation results.
        """
        print("📊 Creating alanine position visualizations...")

        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("SAE Sparse Activations vs Alanine Positions", fontsize=16)

        # 1. Alanine correlation distribution
        ax1 = axes[0, 0]
        n_latents = activations.shape[1]

        correlations_list = []
        for i in range(n_latents):
            key = (i, "alanine")
            if key in correlations:
                correlations_list.append(correlations[key]["pearson"])
            else:
                correlations_list.append(0)

        ax1.hist(correlations_list, bins=30, alpha=0.7, color="skyblue")
        ax1.set_xlabel("Pearson Correlation")
        ax1.set_ylabel("Number of Latent Dimensions")
        ax1.set_title("Alanine Position Correlation Distribution")
        ax1.axvline(x=0, color="red", linestyle="--", alpha=0.7)

        # 2. Top alanine correlations
        ax2 = axes[0, 1]
        top_correlations = []
        for key, corr_data in correlations.items():
            top_correlations.append({
                "latent": key[0],
                "correlation": abs(corr_data["pearson"]),
                "p_value": corr_data["pearson_p"]
            })

        if top_correlations:
            top_correlations.sort(key=lambda x: x["correlation"], reverse=True)
            top_10 = top_correlations[:10]

            latents = [f"Latent {item['latent']}" for item in top_10]
            corrs = [item["correlation"] for item in top_10]

            ax2.barh(range(len(latents)), corrs, color="lightcoral")
            ax2.set_yticks(range(len(latents)))
            ax2.set_yticklabels(latents)
            ax2.set_xlabel("Absolute Correlation")
            ax2.set_title("Top 10 Alanine Position Correlations")

        # 3. Sparsity analysis
        ax3 = axes[1, 0]
        # Calculate sparsity per latent dimension (mean across sequences)
        if len(activations.shape) == 3:
            # 3D activations: (n_sequences, seq_len, n_latents)
            sparsity = (activations == 0).mean(axis=(0, 1))  # Mean across sequences and positions
        else:
            # 2D activations: (n_sequences, n_latents)
            sparsity = (activations == 0).mean(axis=0)  # Mean across sequences

        ax3.hist(sparsity, bins=30, alpha=0.7, color="lightgreen")
        ax3.set_xlabel("Sparsity (fraction of zeros)")
        ax3.set_ylabel("Number of Latent Dimensions")
        ax3.set_title("Sparsity Distribution")

        # 4. Alanine position distribution
        ax4 = axes[1, 1]
        alanine_counts = [mask.sum() for mask in alanine_masks]

        ax4.hist(alanine_counts, bins=20, alpha=0.7, color="gold")
        ax4.set_xlabel("Alanine Positions per Sequence")
        ax4.set_ylabel("Number of Sequences")
        ax4.set_title("Alanine Position Distribution")

        plt.tight_layout()

        plot_path = self.output_dir / "alanine_position_correlation_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"   📈 Plot saved to: {plot_path}")

        plt.show()

    def save_results(self, correlations: dict, statistical_results: dict):
        """
        Save all results to files.
        """
        print("💾 Saving results...")

        # Convert tuple keys to strings for JSON serialization
        correlations_json = {}
        for (latent_idx, feature_type), corr_data in correlations.items():
            key = f"{latent_idx}_{feature_type}"
            correlations_json[key] = corr_data

        # Save correlations
        correlations_path = self.output_dir / "alanine_position_correlations.json"
        with open(correlations_path, "w") as f:
            json.dump(correlations_json, f, indent=2)

        # Save statistical results
        stats_path = self.output_dir / "alanine_position_statistical_results.json"

        # Convert tuple keys in statistical results as well (including nested dictionaries)
        def convert_tuple_keys(obj):
            if isinstance(obj, dict):
                converted = {}
                for key, value in obj.items():
                    if isinstance(key, tuple):
                        new_key = f"{key[0]}_{key[1]}"
                    else:
                        new_key = key
                    converted[new_key] = convert_tuple_keys(value)
                return converted
            elif isinstance(obj, list):
                return [convert_tuple_keys(item) for item in obj]
            else:
                return obj

        stats_json = convert_tuple_keys(statistical_results)

        with open(stats_path, "w") as f:
            json.dump(stats_json, f, indent=2)

        # Save summary CSV
        summary_data = []
        for (latent_idx, feature_type), corr_data in correlations.items():
            summary_data.append({
                "latent_dimension": latent_idx,
                "pearson_correlation": corr_data["pearson"],
                "pearson_p_value": corr_data["pearson_p"]
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "alanine_position_correlation_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"   📄 Alanine position correlations saved to: {correlations_path}")
        print(f"   📄 Statistical results saved to: {stats_path}")
        print(f"   📄 Summary CSV saved to: {summary_path}")
        print()

    def print_summary(self, correlations: dict, statistical_results: dict):
        """
        Print a summary of the correlation analysis results.
        """
        print("\n" + "=" * 60)
        print("📋 CORRELATION ANALYSIS SUMMARY")
        print("=" * 60)

        print("\n🔬 Experiment Details:")
        print(f"   Model: {self.model_path}")
        print(f"   Samples analyzed: {len(correlations)}")
        print(f"   Latent dimensions: {self.pipeline.hidden_dim}")

        print("\n📊 Alanine Position Correlation Results:")
        print(f"   Total correlations calculated: {len(correlations)}")
        print(f"   Significant correlations: {len(statistical_results.get('significant_features', {}))}")
        print(f"   Total statistical tests: {statistical_results.get('total_tests', 0)}")

        # Top correlations
        if correlations:
            top_correlations = []
            for key, corr_data in correlations.items():
                top_correlations.append({
                    "latent": key[0],
                    "correlation": corr_data["pearson"],
                    "p_value": corr_data["pearson_p"]
                })

            top_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            print("\n🏆 Top 5 Alanine Position Correlations:")
            for i, corr in enumerate(top_correlations[:5]):
                print(f"   {i+1}. Latent {corr['latent']}: {corr['correlation']:.4f} (p={corr['p_value']:.4f})")

        print(f"\n✅ Results saved to: {self.output_dir}")

    def run_experiment(self):
        """
        Run the complete correlation analysis experiment.
        """
        print("🚀 Starting Correlation Analysis Experiment")
        print("=" * 60)

        try:
            # Step 1: Load pre-trained pipeline
            self.load_pipeline()

            # Step 2: Process RefSeq data
            processed_records = self.process_refseq_data()

            # Step 3: Generate activations and alanine masks
            activations, alanine_masks = self.generate_activations_and_alanine_masks(processed_records)

            # Step 4: Calculate correlations
            correlations, p_values = self.calculate_correlations(activations, alanine_masks)

            # Step 5: Statistical analysis
            statistical_results = self.perform_statistical_analysis(p_values)

            # Step 6: Create visualizations
            self.create_visualizations(correlations, activations, alanine_masks)

            # Step 7: Save results
            self.save_results(correlations, statistical_results)

            # Step 8: Print summary
            self.print_summary(correlations, statistical_results)

            print("\n🎉 Experiment completed successfully!")

        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the correlation analysis experiment."""

    # Configuration
    model_path = "outputs/ablation_study_reconstruction/regular_sae_latent_1000"
    refseq_file = "../data/vertebrate_mammalian.1.rna.gbff"
    output_dir = "outputs/correlation_analysis"
    max_samples = 50  # Start with smaller sample for testing

    # Create and run experiment
    experiment = CorrelationAnalysisExperiment(
        model_path=model_path,
        refseq_file=refseq_file,
        output_dir=output_dir,
        max_samples=max_samples
    )

    experiment.run_experiment()


if __name__ == "__main__":
    main()
