#!/usr/bin/env python3
"""
CoffeeRL Model Performance Analysis Script

This script provides comprehensive analysis of trained Qwen2 models for coffee brewing
recommendations, including performance evaluation against success criteria and
trade-off analysis between different model sizes.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set up matplotlib for better plots
plt.style.use("default")
try:
    sns.set_palette("husl")
except Exception:
    pass  # Fallback if seaborn not available


class ModelPerformanceAnalyzer:
    """Analyzes and compares model performance against success criteria."""

    def __init__(self, output_dir: str = "analysis_results"):
        """Initialize the analyzer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Success criteria from PRD
        self.success_criteria = {
            "minimum": {
                "grind_direction_accuracy": 0.70,  # 70%
                "extraction_category_accuracy": 0.60,  # 60%
                "brew_time_mae_threshold": 30.0,  # 30 seconds
            },
            "stretch": {
                "grind_direction_accuracy": 0.90,  # 90%
                "extraction_category_accuracy": 0.85,  # 85%
                "brew_time_mae_threshold": 15.0,  # 15 seconds
            },
        }

        # Model size information (in billions of parameters)
        self.model_sizes = {
            "Qwen2-0.5B": 0.5,
            "Qwen2-1.5B": 1.5,
            "qwen2-0.5b": 0.5,
            "qwen2-1.5b": 1.5,
        }

    def load_evaluation_results(self, results_file: str) -> Dict:
        """Load evaluation results from JSON file."""
        try:
            with open(results_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Results file not found: {results_file}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in results file: {results_file}")
            return {}

    def extract_model_name(self, results: Dict) -> str:
        """Extract model name from results, with fallback logic."""
        if "evaluation_info" in results and "model_path" in results["evaluation_info"]:
            model_path = results["evaluation_info"]["model_path"]
            if "qwen2-0.5b" in model_path.lower() or "0.5b" in model_path.lower():
                return "Qwen2-0.5B"
            elif "qwen2-1.5b" in model_path.lower() or "1.5b" in model_path.lower():
                return "Qwen2-1.5B"

        # Fallback to filename analysis
        return "Unknown Model"

    def analyze_single_model(self, results: Dict, model_name: str = None) -> Dict:
        """Analyze performance of a single model."""
        if not results or "metrics" not in results:
            return {}

        metrics = results["metrics"]

        # Extract or infer model name
        if model_name is None:
            model_name = self.extract_model_name(results)

        # Calculate additional metrics
        analysis = {
            "model_name": model_name,
            "model_size_b": self.model_sizes.get(model_name, 0),
            "grind_accuracy": metrics.get("grind_accuracy", 0),
            "extraction_accuracy": metrics.get("extraction_accuracy", 0),
            "average_accuracy": metrics.get("average_accuracy", 0),
            "total_samples": results.get("evaluation_info", {}).get("total_samples", 0),
            "evaluation_time": results.get("evaluation_info", {}).get(
                "evaluation_time_seconds", 0
            ),
        }

        # Check success criteria
        analysis["meets_minimum_grind"] = (
            analysis["grind_accuracy"]
            >= self.success_criteria["minimum"]["grind_direction_accuracy"]
        )
        analysis["meets_stretch_grind"] = (
            analysis["grind_accuracy"]
            >= self.success_criteria["stretch"]["grind_direction_accuracy"]
        )
        analysis["meets_minimum_extraction"] = (
            analysis["extraction_accuracy"]
            >= self.success_criteria["minimum"]["extraction_category_accuracy"]
        )
        analysis["meets_stretch_extraction"] = (
            analysis["extraction_accuracy"]
            >= self.success_criteria["stretch"]["extraction_category_accuracy"]
        )

        # Calculate efficiency metrics
        if analysis["model_size_b"] > 0:
            analysis["accuracy_per_param"] = (
                analysis["average_accuracy"] / analysis["model_size_b"]
            )
            analysis["grind_accuracy_per_param"] = (
                analysis["grind_accuracy"] / analysis["model_size_b"]
            )
        else:
            analysis["accuracy_per_param"] = 0
            analysis["grind_accuracy_per_param"] = 0

        return analysis

    def compare_models(self, model_analyses: List[Dict]) -> Dict:
        """Compare multiple models and generate insights."""
        if not model_analyses:
            return {}

        comparison = {
            "total_models": len(model_analyses),
            "models_meeting_minimum": sum(
                1 for m in model_analyses if m.get("meets_minimum_grind", False)
            ),
            "models_meeting_stretch": sum(
                1 for m in model_analyses if m.get("meets_stretch_grind", False)
            ),
        }

        # Find best models
        if model_analyses:
            best_accuracy = max(
                model_analyses, key=lambda x: x.get("grind_accuracy", 0)
            )
            best_efficiency = max(
                model_analyses, key=lambda x: x.get("accuracy_per_param", 0)
            )

            comparison["best_accuracy_model"] = best_accuracy["model_name"]
            comparison["best_accuracy_score"] = best_accuracy["grind_accuracy"]
            comparison["best_efficiency_model"] = best_efficiency["model_name"]
            comparison["best_efficiency_score"] = best_efficiency["accuracy_per_param"]

        return comparison

    def plot_performance_comparison(self, model_analyses: List[Dict]) -> None:
        """Create performance comparison plots."""
        if not model_analyses:
            print("No model analyses to plot")
            return

        # Create DataFrame for easier plotting
        df = pd.DataFrame(model_analyses)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Accuracy Comparison Bar Chart
        ax1 = axes[0, 0]
        x = np.arange(len(df))
        width = 0.35

        ax1.bar(
            x - width / 2,
            df["grind_accuracy"],
            width,
            label="Grind Direction",
            alpha=0.8,
        )
        ax1.bar(
            x + width / 2,
            df["extraction_accuracy"],
            width,
            label="Extraction Category",
            alpha=0.8,
        )

        # Add success criteria lines
        ax1.axhline(
            y=self.success_criteria["minimum"]["grind_direction_accuracy"],
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Minimum Goal (70%)",
        )
        ax1.axhline(
            y=self.success_criteria["stretch"]["grind_direction_accuracy"],
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Stretch Goal (90%)",
        )

        ax1.set_xlabel("Models")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Model Accuracy Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["model_name"], rotation=45)
        ax1.legend()
        ax1.set_ylim(0, 1.0)

        # 2. Model Size vs Accuracy Scatter Plot
        ax2 = axes[0, 1]
        scatter = ax2.scatter(
            df["model_size_b"],
            df["grind_accuracy"],
            s=100,
            alpha=0.7,
            c=df["grind_accuracy"],
            cmap="viridis",
        )

        for i, model in enumerate(df["model_name"]):
            ax2.annotate(
                model,
                (df["model_size_b"].iloc[i], df["grind_accuracy"].iloc[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        ax2.set_xlabel("Model Size (Billion Parameters)")
        ax2.set_ylabel("Grind Direction Accuracy")
        ax2.set_title("Model Size vs Accuracy Trade-off")
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label="Accuracy")

        # 3. Efficiency Analysis
        ax3 = axes[1, 0]
        ax3.bar(df["model_name"], df["accuracy_per_param"], alpha=0.8, color="skyblue")
        ax3.set_xlabel("Models")
        ax3.set_ylabel("Accuracy per Billion Parameters")
        ax3.set_title("Model Efficiency (Accuracy per Parameter)")
        ax3.tick_params(axis="x", rotation=45)

        # 4. Success Criteria Achievement
        ax4 = axes[1, 1]
        criteria_data = {
            "Minimum Grind": df["meets_minimum_grind"].sum(),
            "Stretch Grind": df["meets_stretch_grind"].sum(),
            "Minimum Extraction": df["meets_minimum_extraction"].sum(),
            "Stretch Extraction": df["meets_stretch_extraction"].sum(),
        }

        bars = ax4.bar(
            criteria_data.keys(), criteria_data.values(), alpha=0.8, color="lightcoral"
        )
        ax4.set_ylabel("Number of Models")
        ax4.set_title("Success Criteria Achievement")
        ax4.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "model_performance_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_report(self, model_analyses: List[Dict], comparison: Dict) -> None:
        """Generate a comprehensive text report."""
        report_path = self.output_dir / "performance_analysis_report.md"

        with open(report_path, "w") as f:
            f.write("# CoffeeRL Model Performance Analysis Report\n\n")
            f.write(
                f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(
                f"- **Total Models Analyzed:** {comparison.get('total_models', 0)}\n"
            )
            f.write(
                f"- **Models Meeting Minimum Criteria:** {comparison.get('models_meeting_minimum', 0)}\n"
            )
            f.write(
                f"- **Models Meeting Stretch Goals:** {comparison.get('models_meeting_stretch', 0)}\n"
            )

            if comparison.get("best_accuracy_model"):
                f.write(
                    f"- **Best Performing Model:** {comparison['best_accuracy_model']} "
                    f"({comparison['best_accuracy_score']:.1%} accuracy)\n"
                )

            if comparison.get("best_efficiency_model"):
                f.write(
                    f"- **Most Efficient Model:** {comparison['best_efficiency_model']} "
                    f"({comparison['best_efficiency_score']:.4f} accuracy/param)\n"
                )

            f.write("\n")

            # Success Criteria Analysis
            f.write("## Success Criteria Analysis\n\n")
            f.write("### Minimum Success Criteria (70% grind accuracy)\n")
            for analysis in model_analyses:
                status = (
                    "✅ PASSED"
                    if analysis.get("meets_minimum_grind", False)
                    else "❌ FAILED"
                )
                f.write(
                    f"- **{analysis['model_name']}:** {analysis['grind_accuracy']:.1%} - {status}\n"
                )

            f.write("\n### Stretch Goals (90% grind accuracy)\n")
            for analysis in model_analyses:
                status = (
                    "✅ PASSED"
                    if analysis.get("meets_stretch_grind", False)
                    else "❌ FAILED"
                )
                f.write(
                    f"- **{analysis['model_name']}:** {analysis['grind_accuracy']:.1%} - {status}\n"
                )

            f.write("\n")

            # Detailed Model Performance
            f.write("## Detailed Model Performance\n\n")
            for analysis in model_analyses:
                f.write(f"### {analysis['model_name']}\n")
                f.write(
                    f"- **Model Size:** {analysis['model_size_b']:.1f}B parameters\n"
                )
                f.write(
                    f"- **Grind Direction Accuracy:** {analysis['grind_accuracy']:.1%}\n"
                )
                f.write(
                    f"- **Extraction Category Accuracy:** {analysis['extraction_accuracy']:.1%}\n"
                )
                f.write(f"- **Average Accuracy:** {analysis['average_accuracy']:.1%}\n")
                f.write(
                    f"- **Efficiency (Accuracy/Param):** {analysis['accuracy_per_param']:.4f}\n"
                )
                f.write(f"- **Evaluation Samples:** {analysis['total_samples']}\n")
                if analysis["evaluation_time"] > 0:
                    f.write(
                        f"- **Evaluation Time:** {analysis['evaluation_time']:.1f} seconds\n"
                    )
                f.write("\n")

    def run_analysis(
        self, results_files: List[str], model_names: Optional[List[str]] = None
    ) -> None:
        """Run complete performance analysis."""
        print("🔍 Starting CoffeeRL Model Performance Analysis...")

        model_analyses = []

        for i, results_file in enumerate(results_files):
            print(f"📊 Analyzing results from: {results_file}")

            results = self.load_evaluation_results(results_file)
            if not results:
                continue

            model_name = (
                model_names[i] if model_names and i < len(model_names) else None
            )
            analysis = self.analyze_single_model(results, model_name)

            if analysis:
                model_analyses.append(analysis)
                print(
                    f"✅ Analysis complete for {analysis['model_name']}: "
                    f"{analysis['grind_accuracy']:.1%} grind accuracy"
                )

        if not model_analyses:
            print("❌ No valid model analyses found. Please check your results files.")
            return

        print(
            f"\n📈 Generating comparison analysis for {len(model_analyses)} models..."
        )

        # Generate comparison
        comparison = self.compare_models(model_analyses)

        # Create visualizations
        print("📊 Creating performance comparison plots...")
        self.plot_performance_comparison(model_analyses)

        # Generate report
        print("📝 Generating comprehensive report...")
        self.generate_report(model_analyses, comparison)

        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
        print(
            f"📄 View the full report: {self.output_dir / 'performance_analysis_report.md'}"
        )
        print(f"📊 View plots: {self.output_dir / 'model_performance_comparison.png'}")


def main():
    """Main function to run the performance analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze CoffeeRL model performance against success criteria"
    )

    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to evaluation results JSON file(s)",
    )

    parser.add_argument(
        "--model-names",
        type=str,
        nargs="*",
        help="Optional model names (if not specified, will be inferred from results)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Output directory for analysis results (default: analysis_results)",
    )

    args = parser.parse_args()

    # Validate input files
    for results_file in args.results:
        if not os.path.exists(results_file):
            print(f"Error: Results file not found: {results_file}")
            sys.exit(1)

    # Validate model names count
    if args.model_names and len(args.model_names) != len(args.results):
        print("Error: Number of model names must match number of results files")
        sys.exit(1)

    # Run analysis
    analyzer = ModelPerformanceAnalyzer(args.output_dir)
    analyzer.run_analysis(args.results, args.model_names)


if __name__ == "__main__":
    main()
