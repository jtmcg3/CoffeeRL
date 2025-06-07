"""Training script for QLoRA fine-tuning of Qwen2 models.

This script provides a complete training pipeline for fine-tuning Qwen2 models
using QLoRA with platform awareness and automatic model selection.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from datasets import load_from_disk
from trl import SFTTrainer

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config.platform_config import (  # noqa: E402
    get_optimal_qwen_model,
    print_platform_summary,
)
from qlora_config import setup_training_environment  # noqa: E402


def load_datasets(
    train_path: str = "data/processed/coffee_training_dataset",
    eval_path: str = "data/processed/coffee_validation_dataset",
    use_subset: bool = False,
    max_train_samples: int = 100,
    max_eval_samples: int = 20,
) -> Tuple[Any, Any]:
    """Load training and evaluation datasets with optional subset for testing."""
    print(f"Loading training dataset from: {train_path}")
    print(f"Loading evaluation dataset from: {eval_path}")

    try:
        train_dataset = load_from_disk(train_path)
        eval_dataset = load_from_disk(eval_path)

        if use_subset:
            print(
                f"Using subset for development: {max_train_samples} train, {max_eval_samples} eval"
            )
            train_dataset = train_dataset.select(
                range(min(len(train_dataset), max_train_samples))
            )
            eval_dataset = eval_dataset.select(
                range(min(len(eval_dataset), max_eval_samples))
            )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Evaluation samples: {len(eval_dataset)}")

        return train_dataset, eval_dataset
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure datasets are properly formatted and saved.")
        raise


def evaluate_model_performance(
    trainer: SFTTrainer,
    eval_dataset: Any,
    output_dir: str,
) -> Dict[str, float]:
    """Evaluate model performance on coffee brewing tasks."""
    print("Evaluating model performance...")

    # Get evaluation metrics from trainer
    eval_results = trainer.evaluate()

    # Save evaluation results
    eval_results_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"Evaluation results saved to: {eval_results_path}")

    # Print key metrics
    if "eval_loss" in eval_results:
        print(f"Evaluation Loss: {eval_results['eval_loss']:.4f}")

    return eval_results


def save_training_metadata(
    output_dir: str,
    training_time: float,
    model_name: str,
    train_samples: int,
    eval_samples: int,
    training_args: Any,
) -> None:
    """Save training metadata for reproducibility."""
    metadata = {
        "training_time_minutes": training_time,
        "model_name": model_name,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "training_args": {
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "fp16": training_args.fp16,
            "bf16": training_args.bf16,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training metadata saved to: {metadata_path}")


def train_model(
    train_dataset: Any,
    eval_dataset: Any,
    output_dir: str = "./models/coffee-qwen2-qlora",
    max_seq_length: int = 512,
    run_evaluation: bool = True,
) -> Tuple[SFTTrainer, Dict[str, Any]]:
    """Train the model using QLoRA with comprehensive logging and evaluation."""
    start_time = time.time()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup training environment
    model, tokenizer, training_args = setup_training_environment()

    # Update output directory in training args
    training_args.output_dir = output_dir

    # Get model name for metadata
    model_name = get_optimal_qwen_model()

    print("Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Evaluation samples: {len(eval_dataset)}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")

    # Format function to convert input/output to text format
    def format_coffee_example(example):
        """Format coffee brewing example for training."""
        import json

        input_text = example["input"]
        output_json = json.dumps(example["output"])
        return f"{input_text}\n{output_json}"

    # Create trainer with updated TRL API
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        peft_config=None,  # PEFT config is handled in model setup
        formatting_func=format_coffee_example,  # Format function for coffee data
    )

    # Start training
    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        raise

    # Calculate training time
    training_time = (time.time() - start_time) / 60
    print(f"Training completed in {training_time:.2f} minutes")

    # Save the final model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    training_metrics = {}
    if trainer.state.log_history:
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"Training metrics saved to: {metrics_path}")
        training_metrics = trainer.state.log_history

    # Run evaluation if requested
    eval_results = {}
    if run_evaluation:
        eval_results = evaluate_model_performance(trainer, eval_dataset, output_dir)

    # Save training metadata
    save_training_metadata(
        output_dir,
        training_time,
        model_name,
        len(train_dataset),
        len(eval_dataset),
        training_args,
    )

    # Combine all results
    results = {
        "training_time_minutes": training_time,
        "training_metrics": training_metrics,
        "eval_results": eval_results,
    }

    return trainer, results


def main() -> int:
    """Main training function with enhanced error handling and options."""
    print("=== CoffeeRL-Lite QLoRA Training ===")

    # Print platform summary
    print_platform_summary()

    # Check for development mode (subset training)
    use_subset = os.getenv("COFFEE_DEV_MODE", "false").lower() == "true"
    max_train_samples = int(os.getenv("COFFEE_MAX_TRAIN_SAMPLES", "100"))
    max_eval_samples = int(os.getenv("COFFEE_MAX_EVAL_SAMPLES", "20"))

    if use_subset:
        print("🔧 Development mode enabled - using subset of data")
        print(f"   Max train samples: {max_train_samples}")
        print(f"   Max eval samples: {max_eval_samples}")

    # Load datasets
    try:
        train_dataset, eval_dataset = load_datasets(
            use_subset=use_subset,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
        )
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        print("Please run data preparation scripts first.")
        return 1

    # Set output directory
    output_dir = os.getenv("COFFEE_OUTPUT_DIR", "./models/coffee-qwen2-qlora")

    # Train model
    try:
        trainer, results = train_model(
            train_dataset,
            eval_dataset,
            output_dir=output_dir,
            run_evaluation=True,
        )

        print("\n=== Training Summary ===")
        print("✅ Training completed successfully!")
        print(f"⏱️  Training time: {results['training_time_minutes']:.2f} minutes")
        print(f"📁 Model saved to: {output_dir}")

        if results["eval_results"]:
            print(
                f"📊 Evaluation loss: {results['eval_results'].get('eval_loss', 'N/A')}"
            )

        return 0

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
