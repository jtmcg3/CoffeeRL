#!/usr/bin/env python3
"""Model Performance Comparison System for CoffeeRL.

This module provides functionality to compare the performance of trained models
against reference models using standardized metrics and evaluation datasets.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from hf_model_manager import HFModelManager  # noqa: E402
from reward_calculator import RewardCalculator  # noqa: E402
from train_rl import RLTrainingLoop  # noqa: E402


class ModelComparator:
    """Compares performance between trained and reference models."""

    def __init__(
        self,
        reward_calculator: Optional[RewardCalculator] = None,
        hf_manager: Optional[HFModelManager] = None,
    ):
        """Initialize the model comparator.

        Args:
            reward_calculator: RewardCalculator instance for performance metrics
            hf_manager: HFModelManager for loading model versions
        """
        self.reward_calculator = reward_calculator or RewardCalculator()
        self.hf_manager = hf_manager or HFModelManager()

    def load_model_for_comparison(
        self, model_path_or_version: str, is_hf_version: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a model and tokenizer for comparison.

        Args:
            model_path_or_version: Local path or HF Hub version tag
            is_hf_version: Whether to load from HF Hub or local path

        Returns:
            Tuple of (model, tokenizer)
        """
        if is_hf_version:
            print(f"🔄 Loading model version '{model_path_or_version}' from HF Hub...")
            model, tokenizer = self.hf_manager.load_model_and_tokenizer(
                version_tag=model_path_or_version
            )
            if model is None or tokenizer is None:
                raise ValueError(
                    f"Failed to load model version '{model_path_or_version}' from HF Hub"
                )
        else:
            print(f"🔄 Loading model from local path: {model_path_or_version}")
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_version)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Determine device and dtype
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and torch.backends.mps.is_available():
                device = "mps"

            torch_dtype = torch.float32 if device == "cpu" else torch.float16

            model = AutoModelForCausalLM.from_pretrained(
                model_path_or_version,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
            )

        print("✅ Model loaded successfully")
        return model, tokenizer

    def evaluate_model_on_dataset(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        max_samples: Optional[int] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Evaluate a model on a dataset and calculate performance metrics.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            dataset: Evaluation dataset
            max_samples: Maximum number of samples to evaluate
            temperature: Generation temperature

        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"🔄 Evaluating model on {len(dataset)} samples...")

        # Limit samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            print(f"📊 Limited evaluation to {max_samples} samples")

        # Create RL training loop for evaluation utilities
        rl_loop = RLTrainingLoop(
            model=model,
            ref_model=None,  # Not needed for evaluation
            tokenizer=tokenizer,
            ppo_trainer=None,  # Not needed for evaluation
            reward_calculator=self.reward_calculator,
        )

        rewards = []
        successful_predictions = 0
        total_samples = len(dataset)

        for i, sample in enumerate(dataset):
            try:
                # Extract coffee data from sample
                coffee_data = self._extract_coffee_data_from_sample(sample)

                # Generate model response
                prompt = rl_loop.format_coffee_prompt(coffee_data)
                response = self._generate_model_response(
                    model, tokenizer, prompt, temperature
                )

                # Parse predictions
                predictions = rl_loop.parse_model_response(response)

                # Calculate reward if we have ground truth
                if "actual_time" in coffee_data or "user_rating" in coffee_data:
                    reward = rl_loop.calculate_reward_for_episode(
                        predictions, coffee_data
                    )
                    rewards.append(reward)

                # Check if prediction was successful (contains required fields)
                if (
                    predictions.get("yield_percentage") is not None
                    or predictions.get("predicted_time") is not None
                ):
                    successful_predictions += 1

                if (i + 1) % 10 == 0:
                    print(f"📊 Processed {i + 1}/{total_samples} samples...")

            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
                continue

        # Calculate metrics
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        task_completion_rate = successful_predictions / total_samples
        reward_std = torch.tensor(rewards).std().item() if len(rewards) > 1 else 0.0

        metrics = {
            "average_reward": avg_reward,
            "task_completion_rate": task_completion_rate,
            "total_samples": total_samples,
            "successful_predictions": successful_predictions,
            "reward_samples": len(rewards),
            "reward_std": reward_std,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
        }

        print(f"✅ Evaluation complete: {metrics}")
        return metrics

    def _extract_coffee_data_from_sample(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract coffee brewing data from a dataset sample.

        Args:
            sample: Dataset sample

        Returns:
            Dictionary with coffee brewing parameters
        """
        # Handle different sample formats
        if "text" in sample:
            # Parse from text format (common in training datasets)
            text = sample["text"]
            coffee_data = self._parse_coffee_data_from_text(text)
        else:
            # Direct dictionary format
            coffee_data = {
                "bean_type": sample.get("bean_type", "Unknown"),
                "grind_size": sample.get("grind_size", "Medium"),
                "water_temp": sample.get("water_temp", 93),
                "brew_method": sample.get("brew_method", "Pour Over"),
                "actual_time": sample.get("actual_time"),
                "user_rating": sample.get("user_rating"),
                "yield_percentage": sample.get("yield_percentage"),
            }

        return coffee_data

    def _parse_coffee_data_from_text(self, text: str) -> Dict[str, Any]:
        """Parse coffee brewing data from text format.

        Args:
            text: Text containing coffee brewing information

        Returns:
            Dictionary with extracted coffee parameters
        """
        # Simple parsing logic - can be enhanced based on actual data format
        coffee_data = {
            "bean_type": "Unknown",
            "grind_size": "Medium",
            "water_temp": 93,
            "brew_method": "Pour Over",
        }

        # Extract basic parameters from text
        lines = text.lower().split("\n")
        for line in lines:
            if "bean" in line or "coffee" in line:
                # Extract bean type
                parts = line.split(":")
                if len(parts) > 1:
                    coffee_data["bean_type"] = parts[1].strip().title()

            elif "grind" in line:
                if "fine" in line:
                    coffee_data["grind_size"] = "Fine"
                elif "coarse" in line:
                    coffee_data["grind_size"] = "Coarse"

            elif "temperature" in line or "temp" in line:
                import re

                temp_match = re.search(r"(\d+)", line)
                if temp_match:
                    coffee_data["water_temp"] = int(temp_match.group(1))

            elif "method" in line:
                if "v60" in line or "pour" in line:
                    coffee_data["brew_method"] = "Pour Over"
                elif "french" in line:
                    coffee_data["brew_method"] = "French Press"

        return coffee_data

    def _generate_model_response(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """Generate response from model for a given prompt.

        Args:
            model: Model to use for generation
            tokenizer: Tokenizer for the model
            prompt: Input prompt
            temperature: Generation temperature

        Returns:
            Generated response text
        """
        # Get model device
        device = next(model.parameters()).device

        # Tokenize prompt
        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt) :].strip()

        return response

    def compare_models(
        self,
        model1_path: str,
        model2_path: str,
        dataset: Dataset,
        model1_is_hf: bool = False,
        model2_is_hf: bool = False,
        max_samples: Optional[int] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Compare performance between two models.

        Args:
            model1_path: Path or version tag for first model
            model2_path: Path or version tag for second model
            dataset: Evaluation dataset
            model1_is_hf: Whether model1 is from HF Hub
            model2_is_hf: Whether model2 is from HF Hub
            max_samples: Maximum samples to evaluate
            temperature: Generation temperature

        Returns:
            Dictionary containing comparison results
        """
        print("🔄 Starting model comparison...")

        # Load models
        print("\n📥 Loading Model 1...")
        model1, tokenizer1 = self.load_model_for_comparison(model1_path, model1_is_hf)

        print("\n📥 Loading Model 2...")
        model2, tokenizer2 = self.load_model_for_comparison(model2_path, model2_is_hf)

        # Evaluate both models
        print("\n📊 Evaluating Model 1...")
        metrics1 = self.evaluate_model_on_dataset(
            model1, tokenizer1, dataset, max_samples, temperature
        )

        print("\n📊 Evaluating Model 2...")
        metrics2 = self.evaluate_model_on_dataset(
            model2, tokenizer2, dataset, max_samples, temperature
        )

        # Calculate comparison metrics
        comparison = {
            "model1": {
                "path": model1_path,
                "is_hf_version": model1_is_hf,
                "metrics": metrics1,
            },
            "model2": {
                "path": model2_path,
                "is_hf_version": model2_is_hf,
                "metrics": metrics2,
            },
            "comparison": {
                "reward_improvement": metrics2["average_reward"]
                - metrics1["average_reward"],
                "completion_rate_improvement": metrics2["task_completion_rate"]
                - metrics1["task_completion_rate"],
                "better_model": (
                    "model2"
                    if metrics2["average_reward"] > metrics1["average_reward"]
                    else "model1"
                ),
            },
            "evaluation_config": {
                "max_samples": max_samples,
                "temperature": temperature,
                "dataset_size": len(dataset),
            },
        }

        return comparison

    def print_comparison_report(self, comparison: Dict[str, Any]) -> None:
        """Print a formatted comparison report.

        Args:
            comparison: Comparison results from compare_models
        """
        print("\n" + "=" * 60)
        print("🔍 MODEL PERFORMANCE COMPARISON REPORT")
        print("=" * 60)

        # Model information
        print(f"\n📊 Model 1: {comparison['model1']['path']}")
        print(f"📊 Model 2: {comparison['model2']['path']}")

        # Performance metrics
        m1 = comparison["model1"]["metrics"]
        m2 = comparison["model2"]["metrics"]

        print("\n📈 PERFORMANCE METRICS:")
        print("{'Metric':<25} {'Model 1':<15} {'Model 2':<15} {'Improvement':<15}")
        print("-" * 70)

        print(
            f"{'Average Reward':<25} {m1['average_reward']:<15.4f} "
            f"{m2['average_reward']:<15.4f} "
            f"{comparison['comparison']['reward_improvement']:<15.4f}"
        )

        print(
            f"{'Task Completion Rate':<25} {m1['task_completion_rate']:<15.4f} "
            f"{m2['task_completion_rate']:<15.4f} "
            f"{comparison['comparison']['completion_rate_improvement']:<15.4f}"
        )

        print(
            f"{'Reward Std Dev':<25} {m1['reward_std']:<15.4f} "
            f"{m2['reward_std']:<15.4f} "
            f"{m2['reward_std'] - m1['reward_std']:<15.4f}"
        )

        # Summary
        better_model = comparison["comparison"]["better_model"]
        print(f"\n🏆 WINNER: {better_model.upper()}")

        if better_model == "model2":
            improvement = comparison["comparison"]["reward_improvement"]
            print(f"   Model 2 shows {improvement:.4f} reward improvement")
        else:
            improvement = -comparison["comparison"]["reward_improvement"]
            print(f"   Model 1 shows {improvement:.4f} reward advantage")

        print("=" * 60)


def load_evaluation_dataset(dataset_path: str) -> Dataset:
    """Load evaluation dataset from disk.

    Args:
        dataset_path: Path to the dataset

    Returns:
        Loaded dataset
    """
    print(f"📂 Loading evaluation dataset from {dataset_path}...")

    try:
        if Path(dataset_path).is_dir():
            dataset = load_from_disk(dataset_path)
        else:
            # Handle JSON file format
            with open(dataset_path, "r") as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)

        print(f"✅ Loaded dataset with {len(dataset)} samples")
        return dataset

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CoffeeRL Model Performance Comparison Tool"
    )

    # Model paths
    parser.add_argument(
        "--model1", type=str, required=True, help="Path or HF version for model 1"
    )
    parser.add_argument(
        "--model2", type=str, required=True, help="Path or HF version for model 2"
    )

    # Model source flags
    parser.add_argument(
        "--model1-hf",
        action="store_true",
        help="Model 1 is from Hugging Face Hub",
    )
    parser.add_argument(
        "--model2-hf",
        action="store_true",
        help="Model 2 is from Hugging Face Hub",
    )

    # Dataset and evaluation parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/coffee_validation_dataset",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save comparison results to JSON file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick comparison with 50 samples",
    )

    return parser.parse_args()


def main() -> int:
    """Main function for command-line usage."""
    args = parse_arguments()

    try:
        # Quick mode settings
        if args.quick:
            args.max_samples = 50
            print("🚀 Quick comparison mode: using 50 samples")

        # Load dataset
        dataset = load_evaluation_dataset(args.dataset)

        # Initialize comparator
        comparator = ModelComparator()

        # Run comparison
        comparison = comparator.compare_models(
            model1_path=args.model1,
            model2_path=args.model2,
            dataset=dataset,
            model1_is_hf=args.model1_hf,
            model2_is_hf=args.model2_hf,
            max_samples=args.max_samples,
            temperature=args.temperature,
        )

        # Print report
        comparator.print_comparison_report(comparison)

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")

        return 0

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
