"""
Ablation Study: Reconstruction Loss Comparison

This script performs an ablation study comparing reconstruction loss between
regular SAE and BatchTopK SAE models across different latent dimensions.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from sae.pipeline import run_complete_batchtopk_pipeline, run_complete_pipeline


class AblationStudy:
    """
    Ablation study comparing regular SAE vs BatchTopK SAE across different latent dimensions
    """

    def __init__(
        self,
        refseq_file: str,
        max_samples: int = 500,
        embedding_dim: int = None,
        latent_dims: list[int] = [100, 250, 500, 1000, 2000],
        epochs: int = 20,
        batch_size: int = 8,
        topk_values: list[int] = [5, 10, 20],
        layer_idx: int = None,
        layer_name: str = "final",
        output_dir: str = "./outputs/ablation_study"
    ):
        """
        Initialize the ablation study

        Args:
            refseq_file: Path to RefSeq file
            max_samples: Number of samples to use
            embedding_dim: Embedding dimension (auto-detected if None)
            latent_dims: List of latent dimensions to test
            epochs: Number of training epochs
            batch_size: Batch size for training
            topk_values: List of topk values to test for BatchTopK
            layer_idx: Layer index to extract embeddings from
            layer_name: Layer name for identification
            output_dir: Directory to save results
        """
        self.refseq_file = refseq_file
        self.max_samples = max_samples
        self.embedding_dim = embedding_dim
        self.latent_dims = latent_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.topk_values = topk_values
        self.layer_idx = layer_idx
        self.layer_name = layer_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = {
            "regular_sae": {},
            "batchtopk_sae": {}
        }

        # L0 sparsity values storage
        self.l0_values = {
            "regular_sae": {},
            "batchtopk_sae": {}
        }

        # Training times
        self.training_times = {
            "regular_sae": {},
            "batchtopk_sae": {}
        }

        print("🔬 Ablation Study Configuration:")
        print(f"   RefSeq file: {refseq_file}")
        print(f"   Max samples: {max_samples}")
        print(f"   Latent dimensions: {latent_dims}")
        print(f"   Epochs: {epochs}")
        print(f"   Layer: {layer_name}")
        print(f"   Output directory: {output_dir}")
        print()

    def run_regular_sae_experiments(self) -> dict[int, float]:
        """Run experiments for regular SAE across different latent dimensions"""
        print("🚀 Running Regular SAE Experiments")
        print("=" * 50)

        results = {}

        for latent_dim in tqdm(self.latent_dims, desc="Regular SAE"):
            print(f"\n📊 Testing Regular SAE with latent_dim={latent_dim}")

            try:
                start_time = time.time()

                # Run regular SAE pipeline
                pipeline = run_complete_pipeline(
                    refseq_file=self.refseq_file,
                    max_samples=self.max_samples,
                    embedding_dim=self.embedding_dim,
                    hidden_dim=latent_dim,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    layer_idx=self.layer_idx,
                    layer_name=self.layer_name,
                    cache_dir=str(self.output_dir / "regular_sae_cache"),
                    model_save_dir=str(self.output_dir / f"regular_sae_latent_{latent_dim}")
                )

                training_time = time.time() - start_time
                self.training_times["regular_sae"][latent_dim] = training_time

                # Get final validation reconstruction loss and L0 sparsity
                final_val_reconstruction_loss = None
                final_val_l0 = None

                # Try to get from stored training history first (works for both regular SAE and BatchTopK SAE)
                if hasattr(pipeline, "_last_training_history") and pipeline._last_training_history:
                    if pipeline._last_training_history.get("val_reconstruction_loss"):
                        final_val_reconstruction_loss = pipeline._last_training_history["val_reconstruction_loss"][-1]
                    if pipeline._last_training_history.get("val_l0_sparsity"):
                        final_val_l0 = pipeline._last_training_history["val_l0_sparsity"][-1]

                # Fallback: try to get from trainer training history
                if final_val_reconstruction_loss is None and hasattr(pipeline, "trainer") and pipeline.trainer and hasattr(pipeline.trainer, "training_history"):
                    if pipeline.trainer.training_history:
                        final_val_reconstruction_loss = pipeline.trainer.training_history[-1].get("val_reconstruction_loss")
                        final_val_l0 = pipeline.trainer.training_history[-1].get("val_l0_sparsity")

                # Fallback: try to get from trainer logger
                if final_val_reconstruction_loss is None and hasattr(pipeline, "trainer") and pipeline.trainer and hasattr(pipeline.trainer, "logger"):
                    if hasattr(pipeline.trainer.logger, "val_losses") and pipeline.trainer.logger.val_losses:
                        final_val_reconstruction_loss = pipeline.trainer.logger.val_losses[-1]

                if final_val_reconstruction_loss is not None:
                    results[latent_dim] = final_val_reconstruction_loss
                    if final_val_l0 is not None:
                        self.l0_values["regular_sae"][latent_dim] = final_val_l0
                    print(f"   ✅ Final validation reconstruction loss: {final_val_reconstruction_loss:.6f}")
                    if final_val_l0 is not None:
                        print(f"   📊 Final validation L0 sparsity: {final_val_l0:.1f}")
                    print(f"   ⏱️  Training time: {training_time:.2f}s")
                else:
                    print("   ❌ No validation reconstruction loss recorded")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[latent_dim] = None

        self.results["regular_sae"] = results
        return results

    def run_batchtopk_sae_experiments(self) -> dict[tuple[int, int], float]:
        """Run experiments for BatchTopK SAE across different latent dimensions and topk values"""
        print("\n🚀 Running BatchTopK SAE Experiments")
        print("=" * 50)

        results = {}

        for latent_dim in tqdm(self.latent_dims, desc="BatchTopK SAE"):
            for topk in self.topk_values:
                print(f"\n📊 Testing BatchTopK SAE with latent_dim={latent_dim}, topk={topk}")

                try:
                    start_time = time.time()

                    # Run BatchTopK SAE pipeline
                    pipeline = run_complete_batchtopk_pipeline(
                        refseq_file=self.refseq_file,
                        max_samples=self.max_samples,
                        embedding_dim=self.embedding_dim,
                        hidden_dim=latent_dim,
                        topk=topk,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        layer_idx=self.layer_idx,
                        layer_name=self.layer_name,
                        cache_dir=str(self.output_dir / "batchtopk_sae_cache"),
                        model_save_dir=str(self.output_dir / f"batchtopk_sae_latent_{latent_dim}_topk_{topk}")
                    )

                    training_time = time.time() - start_time
                    self.training_times["batchtopk_sae"][(latent_dim, topk)] = training_time

                    # Get final validation reconstruction loss from training history
                    final_val_reconstruction_loss = None

                    # Try to get from trainer training history first (for regular SAE)
                    if hasattr(pipeline, "trainer") and pipeline.trainer and hasattr(pipeline.trainer, "training_history"):
                        if pipeline.trainer.training_history:
                            final_val_reconstruction_loss = pipeline.trainer.training_history[-1].get("val_reconstruction_loss")

                    # If not found, try to get from custom training history (for BatchTopK)
                    if final_val_reconstruction_loss is None and hasattr(pipeline, "_last_training_history") and pipeline._last_training_history:
                        if pipeline._last_training_history.get("val_reconstruction_loss"):
                            final_val_reconstruction_loss = pipeline._last_training_history["val_reconstruction_loss"][-1]

                    # Fallback: try to get from trainer logger
                    if final_val_reconstruction_loss is None and hasattr(pipeline, "trainer") and pipeline.trainer and hasattr(pipeline.trainer, "logger"):
                        if hasattr(pipeline.trainer.logger, "val_losses") and pipeline.trainer.logger.val_losses:
                            final_val_reconstruction_loss = pipeline.trainer.logger.val_losses[-1]

                    if final_val_reconstruction_loss is not None:
                        results[(latent_dim, topk)] = final_val_reconstruction_loss
                        print(f"   ✅ Final validation reconstruction loss: {final_val_reconstruction_loss:.6f}")
                        print(f"   ⏱️  Training time: {training_time:.2f}s")
                    else:
                        print("   ❌ No validation reconstruction loss recorded")

                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    results[(latent_dim, topk)] = None

        self.results["batchtopk_sae"] = results
        return results

    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame for easy analysis and plotting"""
        data = []

        # Add regular SAE results
        for latent_dim, loss in self.results["regular_sae"].items():
            if loss is not None:
                data.append({
                    "Model": "Regular SAE",
                    "Latent_Dim": latent_dim,
                    "TopK": None,
                    "Reconstruction_Loss": loss,
                    "Training_Time": self.training_times["regular_sae"].get(latent_dim, None)
                })

        # Add BatchTopK SAE results
        for (latent_dim, topk), loss in self.results["batchtopk_sae"].items():
            if loss is not None:
                data.append({
                    "Model": "BatchTopK SAE",
                    "Latent_Dim": latent_dim,
                    "TopK": topk,
                    "Reconstruction_Loss": loss,
                    "Training_Time": self.training_times["batchtopk_sae"].get((latent_dim, topk), None)
                })

        return pd.DataFrame(data)

    def plot_results(self, df: pd.DataFrame):
        """Create comprehensive plots of the ablation study results"""
        print("\n📊 Creating visualization plots...")

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Ablation Study: Regular SAE vs BatchTopK SAE\nLayer: {self.layer_name}, Samples: {self.max_samples}", fontsize=16)

        # 1. Reconstruction Loss vs Latent Dimension
        ax1 = axes[0, 0]
        for model in ["Regular SAE", "BatchTopK SAE"]:
            model_data = df[df["Model"] == model]
            if model == "Regular SAE":
                line = ax1.plot(model_data["Latent_Dim"], model_data["Reconstruction_Loss"],
                        "o-", label=model, linewidth=2, markersize=8)

                # Add L0 annotations for Regular SAE points
                for idx, row in model_data.iterrows():
                    latent_dim = row["Latent_Dim"]
                    loss = row["Reconstruction_Loss"]
                    l0_val = self.l0_values["regular_sae"].get(latent_dim)
                    if l0_val is not None:
                        ax1.annotate(f"L0: {l0_val:.1f}",
                                   xy=(latent_dim, loss),
                                   xytext=(10, 10),
                                   textcoords="offset points",
                                   fontsize=8,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
            else:
                # For BatchTopK, plot each topk value separately
                for topk in self.topk_values:
                    topk_data = model_data[model_data["TopK"] == topk]
                    if not topk_data.empty:
                        ax1.plot(topk_data["Latent_Dim"], topk_data["Reconstruction_Loss"],
                                "s-", label=f"{model} (TopK={topk})", linewidth=2, markersize=8)

        ax1.set_xlabel("Latent Dimension")
        ax1.set_ylabel("Reconstruction Loss")
        ax1.set_title("Reconstruction Loss vs Latent Dimension")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")

        # 2. Training Time vs Latent Dimension
        ax2 = axes[0, 1]
        for model in ["Regular SAE", "BatchTopK SAE"]:
            model_data = df[df["Model"] == model]
            if model == "Regular SAE":
                ax2.plot(model_data["Latent_Dim"], model_data["Training_Time"],
                        "o-", label=model, linewidth=2, markersize=8)
            else:
                # For BatchTopK, plot each topk value separately
                for topk in self.topk_values:
                    topk_data = model_data[model_data["TopK"] == topk]
                    if not topk_data.empty:
                        ax2.plot(topk_data["Latent_Dim"], topk_data["Training_Time"],
                                "s-", label=f"{model} (TopK={topk})", linewidth=2, markersize=8)

        ax2.set_xlabel("Latent Dimension")
        ax2.set_ylabel("Training Time (seconds)")
        ax2.set_title("Training Time vs Latent Dimension")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")

        # 3. Heatmap of BatchTopK results
        ax3 = axes[1, 0]
        batchtopk_data = df[df["Model"] == "BatchTopK SAE"]
        if not batchtopk_data.empty:
            pivot_data = batchtopk_data.pivot(index="TopK", columns="Latent_Dim", values="Reconstruction_Loss")
            sns.heatmap(pivot_data, annot=True, fmt=".4f", cmap="viridis", ax=ax3)
            ax3.set_title("BatchTopK SAE: Reconstruction Loss Heatmap")
            ax3.set_xlabel("Latent Dimension")
            ax3.set_ylabel("TopK Value")

        # 4. L0 Sparsity vs Latent Dimension (Regular SAE only)
        ax4 = axes[1, 1]
        regular_sae_data = df[df["Model"] == "Regular SAE"]
        if not regular_sae_data.empty:
            l0_values = []
            latent_dims = []
            for latent_dim in self.latent_dims:
                if latent_dim in self.l0_values["regular_sae"]:
                    l0_values.append(self.l0_values["regular_sae"][latent_dim])
                    latent_dims.append(latent_dim)

            if l0_values:
                ax4.plot(latent_dims, l0_values, "o-", color="blue", linewidth=2, markersize=8, label="Regular SAE L0")
                ax4.set_xlabel("Latent Dimension")
                ax4.set_ylabel("L0 Sparsity (Non-zero features per token)")
                ax4.set_title("L0 Sparsity vs Latent Dimension")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_xscale("log")

                # Add value annotations
                for latent_dim, l0_val in zip(latent_dims, l0_values, strict=False):
                    ax4.annotate(f"{l0_val:.1f}",
                               xy=(latent_dim, l0_val),
                               xytext=(0, 5),
                               textcoords="offset points",
                               ha="center", fontsize=8)
            else:
                ax4.text(0.5, 0.5, "No L0 data available", ha="center", va="center", transform=ax4.transAxes)
                ax4.set_title("L0 Sparsity vs Latent Dimension")
        else:
            ax4.text(0.5, 0.5, "No Regular SAE data available", ha="center", va="center", transform=ax4.transAxes)
            ax4.set_title("L0 Sparsity vs Latent Dimension")

        plt.tight_layout()

        # Save the plot
        plot_path = self.output_dir / "ablation_study_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"   📈 Plot saved to: {plot_path}")

        return plot_path

    def save_results(self, df: pd.DataFrame):
        """Save results to CSV and JSON files"""
        print("\n💾 Saving results...")

        # Save DataFrame to CSV
        csv_path = self.output_dir / "ablation_study_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"   📄 CSV results saved to: {csv_path}")

        # Convert tuple keys to strings for JSON serialization
        def convert_tuple_keys_to_strings(data_dict):
            """Convert tuple keys to string keys for JSON serialization"""
            converted = {}
            for key, value in data_dict.items():
                if isinstance(key, tuple):
                    # Convert tuple key to string: (latent_dim, topk) -> "latent_dim_topk"
                    str_key = f"{key[0]}_{key[1]}"
                    converted[str_key] = value
                else:
                    converted[str(key)] = value
            return converted

        # Convert results and training_times for JSON serialization
        json_results = {
            "regular_sae": self.results["regular_sae"],
            "batchtopk_sae": convert_tuple_keys_to_strings(self.results["batchtopk_sae"])
        }

        json_training_times = {
            "regular_sae": self.training_times["regular_sae"],
            "batchtopk_sae": convert_tuple_keys_to_strings(self.training_times["batchtopk_sae"])
        }

        # Save detailed results to JSON
        json_path = self.output_dir / "ablation_study_results.json"
        with open(json_path, "w") as f:
            json.dump({
                "results": json_results,
                "training_times": json_training_times,
                "configuration": {
                    "refseq_file": self.refseq_file,
                    "max_samples": self.max_samples,
                    "latent_dims": self.latent_dims,
                    "epochs": self.epochs,
                    "topk_values": self.topk_values,
                    "layer_idx": self.layer_idx,
                    "layer_name": self.layer_name
                }
            }, f, indent=2)
        print(f"   📄 JSON results saved to: {json_path}")

    def print_summary(self, df: pd.DataFrame):
        """Print a summary of the ablation study results"""
        print("\n" + "=" * 60)
        print("📋 ABLATION STUDY SUMMARY")
        print("=" * 60)

        # Check if we have results for each model type
        regular_sae_results = df[df["Model"] == "Regular SAE"]
        batchtopk_sae_results = df[df["Model"] == "BatchTopK SAE"]

        # Best performing models
        if not regular_sae_results.empty:
            best_regular = regular_sae_results["Reconstruction_Loss"].min()
            print(f"🏆 Best Regular SAE Loss: {best_regular:.6f}")
        else:
            best_regular = None
            print("🏆 Best Regular SAE Loss: No results available")

        if not batchtopk_sae_results.empty:
            best_batchtopk = batchtopk_sae_results["Reconstruction_Loss"].min()
            print(f"🏆 Best BatchTopK SAE Loss: {best_batchtopk:.6f}")
        else:
            best_batchtopk = None
            print("🏆 Best BatchTopK SAE Loss: No results available")

        # Calculate improvement if both models have results
        if best_regular is not None and best_batchtopk is not None:
            improvement = ((best_regular - best_batchtopk) / best_regular * 100)
            print(f"📊 Improvement: {improvement:.2f}%")
        else:
            print("📊 Improvement: Cannot calculate (missing results)")

        # Best configurations
        if not regular_sae_results.empty:
            best_regular_idx = regular_sae_results["Reconstruction_Loss"].idxmin()
            best_regular_config = regular_sae_results.loc[best_regular_idx]
            print("\n🔧 Best Regular SAE Config:")
            print(f"   Latent Dim: {best_regular_config['Latent_Dim']}")
            print(f"   Loss: {best_regular_config['Reconstruction_Loss']:.6f}")
            print(f"   Training Time: {best_regular_config['Training_Time']:.2f}s")
        else:
            print("\n🔧 Best Regular SAE Config: No results available")

        if not batchtopk_sae_results.empty:
            best_batchtopk_idx = batchtopk_sae_results["Reconstruction_Loss"].idxmin()
            best_batchtopk_config = batchtopk_sae_results.loc[best_batchtopk_idx]
            print("\n🔧 Best BatchTopK SAE Config:")
            print(f"   Latent Dim: {best_batchtopk_config['Latent_Dim']}")
            print(f"   TopK: {best_batchtopk_config['TopK']}")
            print(f"   Loss: {best_batchtopk_config['Reconstruction_Loss']:.6f}")
            print(f"   Training Time: {best_batchtopk_config['Training_Time']:.2f}s")
        else:
            print("\n🔧 Best BatchTopK SAE Config: No results available")

        # Performance by latent dimension
        print("\n📊 Performance by Latent Dimension:")
        for latent_dim in self.latent_dims:
            latent_data = df[df["Latent_Dim"] == latent_dim]
            if not latent_data.empty:
                regular_sae_data = latent_data[latent_data["Model"] == "Regular SAE"]
                batchtopk_sae_data = latent_data[latent_data["Model"] == "BatchTopK SAE"]

                print(f"   Latent Dim {latent_dim}:")
                if not regular_sae_data.empty:
                    regular_loss = regular_sae_data["Reconstruction_Loss"].iloc[0]
                    print(f"     Regular SAE: {regular_loss:.6f}")
                else:
                    print("     Regular SAE: No results")

                if not batchtopk_sae_data.empty:
                    batchtopk_losses = batchtopk_sae_data["Reconstruction_Loss"].values
                    print(f"     BatchTopK SAE: {batchtopk_losses.min():.6f} (min), {batchtopk_losses.mean():.6f} (avg)")
                else:
                    print("     BatchTopK SAE: No results")
            else:
                print(f"   Latent Dim {latent_dim}: No results for any model")

    def run_study(self):
        """Run the complete ablation study"""
        print("🔬 Starting Ablation Study")
        print("=" * 60)

        # Run experiments
        self.run_regular_sae_experiments()
        self.run_batchtopk_sae_experiments()

        # Create DataFrame
        df = self.create_comparison_dataframe()

        # Create plots
        self.plot_results(df)

        # Save results
        self.save_results(df)

        # Print summary
        self.print_summary(df)

        print(f"\n✅ Ablation study completed! Results saved to: {self.output_dir}")
        return df


def main():
    """Run the ablation study"""
    # Configuration
    refseq_file = "../data/vertebrate_mammalian.1.rna.gbff"

    # Create ablation study
    study = AblationStudy(
        refseq_file=refseq_file,
        max_samples=200,  # Smaller for faster experimentation
        latent_dims=[100, 250, 500, 1000],  # Test different latent dimensions
        epochs=15,  # Fewer epochs for faster experimentation
        batch_size=8,
        topk_values=[5, 10, 20],  # Test different topk values
        layer_idx=None,  # Final layer
        layer_name="final",
        output_dir="./outputs/ablation_study_reconstruction"
    )

    # Run the study
    results_df = study.run_study()

    return results_df


if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 Ablation study completed successfully!")
        print(f"📊 Results shape: {results.shape}")
        print(f"📈 Models tested: {results['Model'].nunique()}")
        print(f"🔢 Latent dimensions tested: {results['Latent_Dim'].nunique()}")

    except Exception as e:
        print(f"❌ Error running ablation study: {e}")
        import traceback
        traceback.print_exc()
