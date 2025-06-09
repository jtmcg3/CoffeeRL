#!/usr/bin/env python3
"""Batch Training System for CoffeeRL.

This module handles data accumulation and batch training for the RL system.
It can be triggered manually or scheduled to run periodically.
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, concatenate_datasets, load_from_disk

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from hf_model_manager import HFModelManager  # noqa: E402
from reward_calculator import RewardCalculator  # noqa: E402
from rl_environment import (  # noqa: E402
    create_ppo_trainer,
    setup_rl_environment,
)
from train_rl import RLTrainingLoop, create_dummy_rl_dataset  # noqa: E402


class BatchTrainingManager:
    """Manages data accumulation and batch training for CoffeeRL."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the batch training manager.

        Args:
            config: Configuration dictionary for batch training
        """
        # Default configuration
        default_config = {
            "data_accumulation_dir": "./data/accumulated",
            "batch_training_dir": "./data/batch_training",
            "checkpoint_dir": "./checkpoints/batch_training",
            "results_dir": "./results/batch_training",
            "min_batch_size": 50,  # Minimum samples needed for training
            "max_batch_size": 1000,  # Maximum samples per batch
            "training_episodes_per_batch": 500,  # Episodes to train per batch
            "backup_previous_batches": True,
            "max_backup_batches": 5,
        }

        self.config = {**default_config, **(config or {})}

        # Create necessary directories
        for dir_key in [
            "data_accumulation_dir",
            "batch_training_dir",
            "checkpoint_dir",
            "results_dir",
        ]:
            Path(self.config[dir_key]).mkdir(parents=True, exist_ok=True)

        # Initialize Hugging Face model manager
        self.hf_manager = HFModelManager(
            repo_id=self.config.get("hf_repo_id", "JTMCG3/coffeerl-qwen2-0.5b"),
            private=self.config.get("hf_private", False),
        )

        self.batch_count = 0
        self.load_batch_metadata()

    def load_batch_metadata(self) -> None:
        """Load batch training metadata from disk."""
        metadata_path = Path(self.config["results_dir"]) / "batch_metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.batch_count = metadata.get("batch_count", 0)
        else:
            self.batch_count = 0

    def save_batch_metadata(self) -> None:
        """Save batch training metadata to disk."""
        metadata_path = Path(self.config["results_dir"]) / "batch_metadata.json"

        metadata = {
            "batch_count": self.batch_count,
            "last_training_time": datetime.now().isoformat(),
            "config": self.config,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def add_training_data(self, data: List[Dict[str, Any]]) -> None:
        """Add new training data to the accumulation directory.

        Args:
            data: List of training examples to add
        """
        if not data:
            return

        # Create dataset from new data
        new_dataset = Dataset.from_list(data)

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(self.config["data_accumulation_dir"]) / f"data_{timestamp}"

        new_dataset.save_to_disk(save_path)

        print(f"✅ Added {len(data)} training examples to {save_path}")

    def get_accumulated_data_size(self) -> int:
        """Get the total number of accumulated training examples.

        Returns:
            Total number of accumulated examples
        """
        accumulation_dir = Path(self.config["data_accumulation_dir"])
        total_size = 0

        for data_dir in accumulation_dir.glob("data_*"):
            if data_dir.is_dir():
                try:
                    dataset = load_from_disk(data_dir)
                    total_size += len(dataset)
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {data_dir}: {e}")

        return total_size

    def consolidate_accumulated_data(self) -> Optional[Dataset]:
        """Consolidate all accumulated data into a single dataset.

        Returns:
            Consolidated dataset or None if no data available
        """
        accumulation_dir = Path(self.config["data_accumulation_dir"])
        datasets = []

        for data_dir in sorted(accumulation_dir.glob("data_*")):
            if data_dir.is_dir():
                try:
                    dataset = load_from_disk(data_dir)
                    datasets.append(dataset)
                    print(f"📂 Loaded {len(dataset)} examples from {data_dir.name}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {data_dir}: {e}")

        if not datasets:
            print("📭 No accumulated data found")
            return None

        # Concatenate all datasets
        consolidated = concatenate_datasets(datasets)

        # Limit batch size if needed
        max_size = self.config["max_batch_size"]
        if len(consolidated) > max_size:
            print(f"🔄 Limiting batch size from {len(consolidated)} to {max_size}")
            consolidated = consolidated.select(range(max_size))

        print(f"✅ Consolidated {len(consolidated)} training examples")
        return consolidated

    def backup_accumulated_data(self) -> None:
        """Backup accumulated data before training."""
        if not self.config["backup_previous_batches"]:
            return

        accumulation_dir = Path(self.config["data_accumulation_dir"])
        backup_dir = Path(self.config["results_dir"]) / "data_backups"
        backup_dir.mkdir(exist_ok=True)

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_backup_dir = backup_dir / f"batch_{self.batch_count}_{timestamp}"

        if accumulation_dir.exists() and any(accumulation_dir.iterdir()):
            shutil.copytree(accumulation_dir, batch_backup_dir)
            print(f"💾 Backed up accumulated data to {batch_backup_dir}")

            # Clean up old backups
            self.cleanup_old_backups(backup_dir)

    def cleanup_old_backups(self, backup_dir: Path) -> None:
        """Clean up old backup directories."""
        max_backups = self.config["max_backup_batches"]
        backup_dirs = sorted(
            backup_dir.glob("batch_*"), key=lambda x: x.stat().st_mtime
        )

        if len(backup_dirs) > max_backups:
            for old_backup in backup_dirs[:-max_backups]:
                shutil.rmtree(old_backup)
                print(f"🗑️ Removed old backup: {old_backup.name}")

    def clear_accumulated_data(self) -> None:
        """Clear accumulated data after successful training."""
        accumulation_dir = Path(self.config["data_accumulation_dir"])

        if accumulation_dir.exists():
            for item in accumulation_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            print("🧹 Cleared accumulated data")

    def run_batch_training(self, dataset: Dataset) -> Dict[str, Any]:
        """Run batch training on the provided dataset.

        Args:
            dataset: Dataset to train on

        Returns:
            Training results dictionary
        """
        print(f"🚀 Starting batch training #{self.batch_count + 1}...")

        # Setup RL environment
        print("Setting up RL environment...")
        model, ref_model, tokenizer, ppo_config = setup_rl_environment()

        # Create PPO trainer
        ppo_trainer = create_ppo_trainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            ppo_config=ppo_config,
        )

        # Create reward calculator
        reward_calculator = RewardCalculator()

        # Setup training configuration
        training_config = {
            "max_episodes": self.config["training_episodes_per_batch"],
            "save_freq": max(50, self.config["training_episodes_per_batch"] // 10),
            "eval_freq": max(25, self.config["training_episodes_per_batch"] // 20),
            "log_freq": max(10, self.config["training_episodes_per_batch"] // 50),
        }

        # Create training loop
        rl_trainer = RLTrainingLoop(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            ppo_trainer=ppo_trainer,
            reward_calculator=reward_calculator,
            config=training_config,
        )

        # Run training
        start_time = time.time()
        training_results = rl_trainer.train(dataset)
        training_time = time.time() - start_time

        # Save batch results
        batch_results = {
            "batch_number": self.batch_count + 1,
            "dataset_size": len(dataset),
            "training_time": training_time,
            "training_results": training_results,
            "timestamp": datetime.now().isoformat(),
        }

        # Save model checkpoint locally
        checkpoint_path = (
            Path(self.config["checkpoint_dir"]) / f"batch_{self.batch_count + 1}"
        )
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

        # Upload model to Hugging Face Hub
        version_tag = f"batch-{self.batch_count + 1}"
        hf_upload_success = self.hf_manager.upload_model(
            model_path=str(checkpoint_path),
            version_tag=version_tag,
            training_results=training_results,
        )

        # Save batch results
        results_path = (
            Path(self.config["results_dir"])
            / f"batch_{self.batch_count + 1}_results.json"
        )
        batch_results["hf_upload_success"] = hf_upload_success
        batch_results["hf_version_tag"] = version_tag if hf_upload_success else None
        batch_results["hf_repo_id"] = (
            self.hf_manager.repo_id if hf_upload_success else None
        )

        with open(results_path, "w") as f:
            json.dump(batch_results, f, indent=2)

        print(f"✅ Batch training #{self.batch_count + 1} completed!")
        print(f"📊 Final average reward: {training_results['final_avg_reward']:.4f}")
        print(f"💾 Model saved locally to: {checkpoint_path}")
        if hf_upload_success:
            print(
                f"🤗 Model uploaded to HF Hub: https://huggingface.co/{self.hf_manager.repo_id}/tree/{version_tag}"
            )
        else:
            print("⚠️ HF Hub upload failed - model only saved locally")
        print(f"📄 Results saved to: {results_path}")

        return batch_results

    def can_run_batch_training(self) -> Tuple[bool, str]:
        """Check if batch training can be run.

        Returns:
            Tuple of (can_run, reason)
        """
        data_size = self.get_accumulated_data_size()
        min_size = self.config["min_batch_size"]

        if data_size < min_size:
            return (
                False,
                f"Insufficient data: {data_size} < {min_size} (minimum required)",
            )

        return True, f"Ready to train on {data_size} examples"

    def run_batch_training_cycle(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """Run a complete batch training cycle.

        Args:
            force: Force training even if minimum data requirements aren't met

        Returns:
            Training results or None if training was skipped
        """
        print("🔍 Checking batch training readiness...")

        can_run, reason = self.can_run_batch_training()

        if not can_run and not force:
            print(f"⏸️ Skipping batch training: {reason}")
            return None

        if force and not can_run:
            print(f"⚠️ Forcing batch training despite: {reason}")
        else:
            print(f"✅ {reason}")

        # Consolidate accumulated data
        dataset = self.consolidate_accumulated_data()

        if dataset is None:
            print("❌ No data available for training")
            return None

        # Backup data before training
        self.backup_accumulated_data()

        try:
            # Run training
            results = self.run_batch_training(dataset)

            # Update batch count
            self.batch_count += 1
            self.save_batch_metadata()

            # Clear accumulated data after successful training
            self.clear_accumulated_data()

            return results

        except Exception as e:
            print(f"❌ Batch training failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get the history of batch training runs.

        Returns:
            List of training result dictionaries
        """
        results_dir = Path(self.config["results_dir"])
        history = []

        for results_file in sorted(results_dir.glob("batch_*_results.json")):
            try:
                with open(results_file, "r") as f:
                    results = json.load(f)
                    history.append(results)
            except Exception as e:
                print(f"⚠️ Warning: Could not load {results_file}: {e}")

        return history

    def load_best_model(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Load the best performing model from Hugging Face Hub.

        Returns:
            Tuple of (model, tokenizer) or (None, None) if failed
        """
        return self.hf_manager.load_model_and_tokenizer("main")

    def load_model_version(
        self, version_tag: str
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Load a specific model version from Hugging Face Hub.

        Args:
            version_tag: Version tag to load (e.g., "batch-1")

        Returns:
            Tuple of (model, tokenizer) or (None, None) if failed
        """
        return self.hf_manager.load_model_and_tokenizer(version_tag)

    def print_status(self) -> None:
        """Print current batch training status."""
        print("\n📊 Batch Training Status")
        print("=" * 50)

        data_size = self.get_accumulated_data_size()
        can_run, reason = self.can_run_batch_training()

        print(f"Accumulated data: {data_size} examples")
        print(f"Batch count: {self.batch_count}")
        print(f"Training readiness: {reason}")

        # Show recent training history
        history = self.get_training_history()
        if history:
            print("\nRecent training runs:")
            for result in history[-3:]:  # Show last 3 runs
                batch_num = result["batch_number"]
                reward = result["training_results"]["final_avg_reward"]
                timestamp = result["timestamp"][:19]  # Remove microseconds
                hf_status = "🤗" if result.get("hf_upload_success") else "💾"
                print(
                    f"  Batch {batch_num}: Reward={reward:.4f} {hf_status} ({timestamp})"
                )
        else:
            print("\nNo previous training runs found")

        # Show HF Hub status
        print("\n🤗 Hugging Face Hub:")
        print(f"Repository: {self.hf_manager.repo_id}")
        print(f"Exists: {'✅ Yes' if self.hf_manager.repo_exists else '❌ No'}")

        print("=" * 50)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for batch training."""
    parser = argparse.ArgumentParser(
        description="CoffeeRL Batch Training Manager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Commands
    parser.add_argument(
        "command",
        choices=[
            "status",
            "train",
            "add-dummy-data",
            "clear-data",
            "history",
            "hf-status",
            "hf-versions",
        ],
        help="Command to execute",
    )

    # Training options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force training even if minimum data requirements aren't met",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes per batch",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=50,
        help="Minimum batch size for training",
    )

    # Data options
    parser.add_argument(
        "--dummy-data-size",
        type=int,
        default=100,
        help="Number of dummy examples to add",
    )

    return parser.parse_args()


def main() -> int:
    """Main function for batch training management."""
    args = parse_arguments()

    # Setup configuration
    config = {
        "training_episodes_per_batch": args.episodes,
        "min_batch_size": args.min_batch_size,
    }

    # Create batch training manager
    manager = BatchTrainingManager(config)

    try:
        if args.command == "status":
            manager.print_status()

        elif args.command == "train":
            print("🚀 Starting manual batch training...")
            results = manager.run_batch_training_cycle(force=args.force)

            if results:
                print("\n🎉 Batch training completed successfully!")
                print(
                    f"Batch #{results['batch_number']} - Final reward: {results['training_results']['final_avg_reward']:.4f}"
                )
            else:
                print("\n⏸️ Batch training was skipped or failed")
                return 1

        elif args.command == "add-dummy-data":
            print(f"📝 Adding {args.dummy_data_size} dummy training examples...")
            dummy_dataset = create_dummy_rl_dataset()

            # Convert to list format expected by add_training_data
            dummy_data = []
            for i in range(min(args.dummy_data_size, len(dummy_dataset))):
                dummy_data.append(dummy_dataset[i])

            manager.add_training_data(dummy_data)
            print(f"✅ Added {len(dummy_data)} dummy examples")

        elif args.command == "clear-data":
            print("🧹 Clearing accumulated data...")
            manager.clear_accumulated_data()
            print("✅ Accumulated data cleared")

        elif args.command == "history":
            history = manager.get_training_history()

            if not history:
                print("📭 No training history found")
                return 0

            print(f"\n📊 Training History ({len(history)} batches)")
            print("=" * 80)

            for result in history:
                batch_num = result["batch_number"]
                dataset_size = result["dataset_size"]
                reward = result["training_results"]["final_avg_reward"]
                training_time = result["training_time"]
                timestamp = result["timestamp"][:19]

                print(
                    f"Batch {batch_num:2d}: {dataset_size:3d} samples, "
                    f"Reward={reward:7.4f}, Time={training_time:6.1f}s ({timestamp})"
                )

        elif args.command == "hf-status":
            print("🤗 Hugging Face Hub Status")
            manager.hf_manager.print_model_status()

        elif args.command == "hf-versions":
            versions = manager.hf_manager.list_model_versions()

            if not versions:
                print("📭 No model versions found on Hugging Face Hub")
                return 0

            print(f"\n🤗 Hugging Face Model Versions ({len(versions)})")
            print("=" * 80)

            for version in versions:
                version_tag = version["version_tag"]
                reward = version.get("final_reward", 0.0)
                dataset_size = version.get("dataset_size", 0)
                timestamp = version.get("upload_timestamp", "")[:19]

                print(
                    f"{version_tag:12} | Reward: {reward:7.4f} | Data: {dataset_size:3d} | {timestamp}"
                )

            print(
                f"\n🔗 Repository: https://huggingface.co/{manager.hf_manager.repo_id}"
            )

        return 0

    except Exception as e:
        print(f"❌ Command failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
