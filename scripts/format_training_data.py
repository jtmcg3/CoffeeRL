#!/usr/bin/env python3
"""
Format training data for QLoRA fine-tuning of Qwen2 models.

This script processes the collected coffee brewing data and formats it
for fine-tuning with QLoRA on Qwen2-0.5B (local) or Qwen2-1.5B (cloud).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(
    data_dir: Path = Path("data/processed"),
) -> Tuple[List[Dict], List[Dict]]:
    """Load training and validation examples from JSON files."""
    training_file = data_dir / "training_examples.json"
    validation_file = data_dir / "validation_examples.json"

    logger.info("Loading training data from %s", training_file)
    with open(training_file, "r", encoding="utf-8") as f:
        training_examples = json.load(f)

    logger.info("Loading validation data from %s", validation_file)
    with open(validation_file, "r", encoding="utf-8") as f:
        validation_examples = json.load(f)

    logger.info("Loaded %d training examples", len(training_examples))
    logger.info("Loaded %d validation examples", len(validation_examples))

    return training_examples, validation_examples


def format_example_for_qwen2(example: Dict[str, Any]) -> str:
    """Format a single example for Qwen2 fine-tuning."""
    input_text = example["input"]
    output_data = example["output"]

    # Format output as JSON string
    output_json = json.dumps(output_data, ensure_ascii=False)

    # Create prompt template optimized for Qwen2
    prompt = f"""Analyze this V60 brew: {input_text}

Provide grind adjustment, expected brew time, extraction assessment, and reasoning:"""

    # Format for Qwen2 training (simple concatenation)
    formatted_text = f"{prompt}\n{output_json}"

    return formatted_text


def create_qwen2_dataset(examples: List[Dict[str, Any]]) -> Dataset:
    """Convert examples to Hugging Face Dataset format for Qwen2."""
    formatted_data = []

    for example in examples:
        formatted_text = format_example_for_qwen2(example)
        formatted_data.append({"text": formatted_text})

    # Convert to DataFrame then Dataset
    df = pd.DataFrame(formatted_data)
    dataset = Dataset.from_pandas(df)

    return dataset


def validate_token_lengths(
    dataset: Dataset, model_size: str = "0.5B"
) -> Dict[str, Any]:
    """Validate token lengths using Qwen2 tokenizer."""
    # Use appropriate model name for tokenizer
    model_name = (
        f"Qwen/Qwen2-{model_size}"
        if model_size in ["0.5B", "1.5B"]
        else "Qwen/Qwen2-0.5B"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Loaded tokenizer for %s", model_name)
    except Exception as e:
        logger.warning("Could not load tokenizer for %s: %s", model_name, e)
        logger.info("Using fallback tokenizer estimation")
        return {"status": "estimated", "max_length": 8192}

    token_lengths: List[int] = []
    max_length = 8192  # Qwen2 context length

    for example in dataset:
        tokens = tokenizer.encode(example["text"])
        token_lengths.append(len(tokens))

    stats = {
        "total_examples": len(token_lengths),
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "avg_tokens": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
        "over_limit": sum(1 for length in token_lengths if length > max_length),
        "model_max_length": max_length,
        "status": "validated",
    }

    logger.info("Token length stats: %s", stats)

    if stats["over_limit"] > 0:
        logger.warning("%d examples exceed %d tokens", stats["over_limit"], max_length)

    return stats


def save_formatted_dataset(
    dataset: Dataset, output_dir: Path, dataset_type: str, model_size: str
) -> None:
    """Save formatted dataset to disk."""
    output_path = output_dir / f"coffee_{dataset_type}_qwen2_{model_size}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving %s dataset to %s", dataset_type, output_path)
    dataset.save_to_disk(str(output_path))

    # Also save as JSON for inspection
    json_path = output_path.parent / f"coffee_{dataset_type}_qwen2_{model_size}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset.to_list(), f, ensure_ascii=False, indent=2)

    logger.info("Also saved as JSON: %s", json_path)


def format_training_data_for_qwen2(
    model_size: str = "0.5B",
    data_dir: Path = Path("data/processed"),
    output_dir: Path = Path("data/processed"),
) -> Dict[str, Any]:
    """
    Main function to format training data for Qwen2 fine-tuning.

    Args:
        model_size: "0.5B" for local development, "1.5B" for cloud training
        data_dir: Directory containing input JSON files
        output_dir: Directory to save formatted datasets

    Returns:
        Dictionary with processing statistics
    """
    logger.info("Formatting training data for Qwen2-%s", model_size)

    # Load raw data
    training_examples, validation_examples = load_training_data(data_dir)

    # Format datasets
    logger.info("Formatting training dataset...")
    training_dataset = create_qwen2_dataset(training_examples)

    logger.info("Formatting validation dataset...")
    validation_dataset = create_qwen2_dataset(validation_examples)

    # Validate token lengths
    logger.info("Validating token lengths...")
    training_stats = validate_token_lengths(training_dataset, model_size)
    validation_stats = validate_token_lengths(validation_dataset, model_size)

    # Save formatted datasets
    save_formatted_dataset(training_dataset, output_dir, "training", model_size)
    save_formatted_dataset(validation_dataset, output_dir, "validation", model_size)

    # Print example
    logger.info("Example formatted text:")
    logger.info("%s...", training_dataset[0]["text"][:500])

    results = {
        "model_size": model_size,
        "training_examples": len(training_dataset),
        "validation_examples": len(validation_dataset),
        "training_stats": training_stats,
        "validation_stats": validation_stats,
        "output_dir": str(output_dir),
    }

    logger.info("Formatting complete: %s", results)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Format training data for Qwen2 fine-tuning"
    )
    parser.add_argument(
        "--model-size",
        choices=["0.5B", "1.5B"],
        default="0.5B",
        help="Qwen2 model size (0.5B for local, 1.5B for cloud)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing input JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save formatted datasets",
    )

    args = parser.parse_args()

    results = format_training_data_for_qwen2(
        model_size=args.model_size, data_dir=args.data_dir, output_dir=args.output_dir
    )

    print("\nFormatting completed successfully!")
    print(f"Training examples: {results['training_examples']}")
    print(f"Validation examples: {results['validation_examples']}")
    print(f"Output directory: {results['output_dir']}")
