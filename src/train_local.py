#!/usr/bin/env python3
"""Local training script for CoffeeRL-Lite with Qwen2-0.5B.

This script is optimized for local training environments with enhanced
command-line arguments, better logging, and local-specific optimizations.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from datasets import load_from_disk
from trl import SFTTrainer

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from qlora_config import setup_training_environment  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for local training."""
    parser = argparse.ArgumentParser(
        description="CoffeeRL-Lite Local Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/coffee_training_dataset",
        help="Path to training dataset directory",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="data/processed/coffee_validation_dataset",
        help="Path to evaluation dataset directory",
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Model name or path (default optimized for local training)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/coffee-qwen2-local",
        help="Output directory for trained model",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (auto-detected if not specified)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=None,
        help="Gradient accumulation steps (auto-detected if not specified)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    # Development and debugging
    parser.add_argument(
        "--dev-mode",
        action="store_true",
        help="Enable development mode with subset of data",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=100,
        help="Maximum training samples in dev mode",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=20,
        help="Maximum evaluation samples in dev mode",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation during training",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (default: save per epoch)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log training metrics every N steps",
    )

    # Platform overrides
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU training even if GPU is available",
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable quantization even if bitsandbytes is available",
    )

    return parser.parse_args()


def setup_environment_variables(args: argparse.Namespace) -> None:
    """Set environment variables based on command line arguments."""
    # Set development mode if requested
    if args.dev_mode:
        os.environ["COFFEE_DEV_MODE"] = "true"
        os.environ["COFFEE_MAX_TRAIN_SAMPLES"] = str(args.max_train_samples)
        os.environ["COFFEE_MAX_EVAL_SAMPLES"] = str(args.max_eval_samples)

    # Set output directory
    os.environ["COFFEE_OUTPUT_DIR"] = args.output_dir


def load_datasets_with_args(args: argparse.Namespace) -> Tuple[Any, Any]:
    """Load training and evaluation datasets based on arguments."""
    print(f"Loading training dataset from: {args.train_data}")
    print(f"Loading evaluation dataset from: {args.eval_data}")

    try:
        train_dataset = load_from_disk(args.train_data)
        eval_dataset = load_from_disk(args.eval_data)

        if args.dev_mode:
            print(
                f"🔧 Development mode: using {args.max_train_samples} train, "
                f"{args.max_eval_samples} eval samples"
            )
            train_dataset = train_dataset.select(
                range(min(len(train_dataset), args.max_train_samples))
            )
            eval_dataset = eval_dataset.select(
                range(min(len(eval_dataset), args.max_eval_samples))
            )

        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Evaluation samples: {len(eval_dataset)}")

        return train_dataset, eval_dataset
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        print("Please ensure datasets are properly formatted and saved.")
        print("Run data preparation scripts first if needed.")
        raise


def create_local_training_args(args: argparse.Namespace) -> Any:
    """Create training arguments optimized for local training."""
    from transformers import TrainingArguments

    from config.platform_config import (
        get_gradient_accumulation_steps,
        get_training_batch_size,
    )

    # Use provided values or auto-detect
    batch_size = args.batch_size or get_training_batch_size()
    gradient_accumulation = (
        args.gradient_accumulation or get_gradient_accumulation_steps()
    )

    # Adjust learning rate based on effective batch size
    effective_batch_size = batch_size * gradient_accumulation
    learning_rate = args.learning_rate * (effective_batch_size / 16)

    # Determine save strategy
    save_strategy = "steps" if args.save_steps else "epoch"
    save_steps = args.save_steps or 500

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        # Evaluation settings
        eval_strategy="epoch" if not args.no_eval else "no",
        eval_steps=None,
        # Saving settings
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=not args.no_eval,
        # Logging
        logging_steps=args.logging_steps,
        report_to="tensorboard",
        # Optimization
        warmup_ratio=0.05,
        weight_decay=0.01,
        # Platform-specific settings
        fp16=torch.cuda.is_available() and not args.force_cpu,
        bf16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        # Local training optimizations
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        disable_tqdm=False,  # Keep progress bars for local monitoring
    )

    return training_args


def format_coffee_example(example: Dict[str, Any]) -> str:
    """Format coffee brewing example for training."""
    input_text = example["input"]
    output_json = json.dumps(example["output"])
    return f"### Coffee Brewing Input:\n{input_text}\n\n### Recommended Adjustment:\n{output_json}"


def train_local_model(args: argparse.Namespace) -> Tuple[SFTTrainer, Dict[str, Any]]:
    """Train the model locally with enhanced monitoring."""
    start_time = time.time()

    print("🚀 Starting local CoffeeRL-Lite training...")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🤖 Model: {args.model_name}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    train_dataset, eval_dataset = load_datasets_with_args(args)

    # Setup training environment (model, tokenizer, etc.)
    print("⚙️  Setting up training environment...")
    model, tokenizer, _ = setup_training_environment()

    # Create local training arguments
    training_args = create_local_training_args(args)

    print("\n📊 Training Configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Max sequence length: {args.max_seq_length}")
    print(f"  Evaluation: {'Enabled' if not args.no_eval else 'Disabled'}")
    print(
        f"  Device: {'GPU' if torch.cuda.is_available() and not args.force_cpu else 'CPU'}"
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if not args.no_eval else None,
        args=training_args,
        processing_class=tokenizer,
        peft_config=None,  # PEFT config handled in model setup
        formatting_func=format_coffee_example,
    )

    # Start training
    print("\n🏃 Starting training...")
    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("💾 Saving current model state...")
        trainer.save_model()
        raise
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

    # Calculate training time
    training_time = (time.time() - start_time) / 60
    print(f"⏱️  Training completed in {training_time:.2f} minutes")

    # Save the final model
    print(f"💾 Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save training metrics
    training_metrics = {}
    if trainer.state.log_history:
        metrics_path = os.path.join(args.output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"📈 Training metrics saved to: {metrics_path}")
        training_metrics = trainer.state.log_history

    # Run evaluation if enabled
    eval_results = {}
    if not args.no_eval:
        print("📊 Running final evaluation...")
        eval_results = trainer.evaluate()
        eval_path = os.path.join(args.output_dir, "eval_results.json")
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"📊 Evaluation results saved to: {eval_path}")
        if "eval_loss" in eval_results:
            print(f"📊 Final evaluation loss: {eval_results['eval_loss']:.4f}")

    # Save training metadata
    metadata = {
        "training_time_minutes": training_time,
        "model_name": args.model_name,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "arguments": vars(args),
        "training_args": {
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_path = os.path.join(args.output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Training metadata saved to: {metadata_path}")

    # Combine results
    results = {
        "training_time_minutes": training_time,
        "training_metrics": training_metrics,
        "eval_results": eval_results,
    }

    return trainer, results


def main() -> int:
    """Main function for local training."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Setup environment
        setup_environment_variables(args)

        # Print platform summary
        from config.platform_config import print_platform_summary

        print_platform_summary()

        # Train model
        trainer, results = train_local_model(args)

        # Print summary
        print("\n🎉 === Training Summary ===")
        print("✅ Training completed successfully!")
        print(f"⏱️  Training time: {results['training_time_minutes']:.2f} minutes")
        print(f"📁 Model saved to: {args.output_dir}")

        if results["eval_results"]:
            eval_loss = results["eval_results"].get("eval_loss", "N/A")
            print(f"📊 Final evaluation loss: {eval_loss}")

        print("\n🚀 Next steps:")
        print(
            f"  1. Test the model: python src/evaluate_local.py --model-path {args.output_dir}"
        )
        print(
            "  2. Run inference: python src/inference.py --model-path {args.output_dir}"
        )
        print("  3. View training logs: tensorboard --logdir {args.output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
