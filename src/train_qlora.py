"""Training script for QLoRA fine-tuning of Qwen2 models.

This script provides a complete training pipeline for fine-tuning Qwen2 models
using QLoRA with platform awareness and automatic model selection.
"""

import os
import sys
from pathlib import Path

from datasets import load_from_disk
from trl import SFTTrainer

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config.platform_config import print_platform_summary
from qlora_config import setup_training_environment


def load_datasets(
    train_path: str = "data/processed/coffee_training_dataset",
    eval_path: str = "data/processed/coffee_validation_dataset",
):
    """Load training and evaluation datasets."""
    print(f"Loading training dataset from: {train_path}")
    print(f"Loading evaluation dataset from: {eval_path}")

    try:
        train_dataset = load_from_disk(train_path)
        eval_dataset = load_from_disk(eval_path)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Evaluation samples: {len(eval_dataset)}")

        return train_dataset, eval_dataset
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure datasets are properly formatted and saved.")
        raise


def train_model(
    train_dataset,
    eval_dataset,
    output_dir: str = "./models/coffee-qwen2-qlora",
    max_seq_length: int = 512,
):
    """Train the model using QLoRA."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup training environment
    model, tokenizer, training_args = setup_training_environment()

    # Update output directory in training args
    training_args.output_dir = output_dir

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=max_seq_length,
        dataset_text_field="text",  # Assuming datasets have a 'text' field
        packing=False,  # Disable packing for simplicity
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the final model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    if trainer.state.log_history:
        import json

        with open(f"{output_dir}/training_metrics.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)

    print("Training completed!")
    return trainer


def main():
    """Main training function."""
    print("=== CoffeeRL-Lite QLoRA Training ===")

    # Print platform summary
    print_platform_summary()

    # Load datasets
    try:
        train_dataset, eval_dataset = load_datasets()
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        print("Please run data preparation scripts first.")
        return 1

    # Train model
    try:
        trainer = train_model(train_dataset, eval_dataset)
        print("Training completed successfully!")
        return 0
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
