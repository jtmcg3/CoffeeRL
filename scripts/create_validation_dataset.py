#!/usr/bin/env python3
"""
Create Validation Dataset Script

This script creates a balanced validation dataset of 100 high-quality examples
from all available coffee brewing data sources for model evaluation.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import Dataset


def load_all_data_sources(data_dir: Path) -> List[Dict[str, Any]]:
    """Load and combine all available data sources."""
    all_examples = []

    # Load raw dataset (GPT-3.5 generated)
    raw_data_path = data_dir / "coffee_dataset_raw.json"
    if raw_data_path.exists():
        with open(raw_data_path, "r") as f:
            raw_data = json.load(f)
            all_examples.extend(raw_data)
            print(f"Loaded {len(raw_data)} examples from raw dataset")

    # Load manual examples
    manual_data_path = data_dir / "manual_examples.json"
    if manual_data_path.exists():
        with open(manual_data_path, "r") as f:
            manual_data = json.load(f)
            all_examples.extend(manual_data)
            print(f"Loaded {len(manual_data)} examples from manual dataset")

    # Load community examples
    community_dir = data_dir / "community"
    if community_dir.exists():
        for community_file in community_dir.glob("*_examples*.json"):
            with open(community_file, "r") as f:
                community_data = json.load(f)
                if isinstance(community_data, list):
                    all_examples.extend(community_data)
                    print(
                        f"Loaded {len(community_data)} examples from {community_file.name}"
                    )

    print(f"Total examples loaded: {len(all_examples)}")
    return all_examples


def categorize_by_extraction(
    examples: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize examples by extraction type for balanced sampling."""
    categories = {"under": [], "good": [], "over": []}

    for example in examples:
        output = example.get("output", {})
        extraction = output.get("extraction", "")

        if extraction in categories:
            categories[extraction].append(example)
        else:
            # Fallback: determine extraction from grind change and reasoning
            grind_change = output.get("grind_change", "")
            reasoning = output.get("reasoning", "").lower()

            if "finer" in grind_change or "under-extraction" in reasoning:
                categories["under"].append(example)
            elif "coarser" in grind_change or "over-extraction" in reasoning:
                categories["over"].append(example)
            else:
                categories["good"].append(example)

    return categories


def select_high_quality_examples(
    categories: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Select high-quality examples based on confidence and diversity."""
    high_quality = []

    for category, examples in categories.items():
        # Sort by confidence score (descending)
        sorted_examples = sorted(
            examples,
            key=lambda x: x.get("output", {}).get("confidence", 0.0),
            reverse=True,
        )

        # Select diverse examples (avoid duplicates)
        selected = []
        seen_inputs = set()

        for example in sorted_examples:
            input_text = example.get("input", "")
            # Create a simplified version for duplicate detection
            simplified = input_text.lower().replace(" ", "")

            if simplified not in seen_inputs:
                selected.append(example)
                seen_inputs.add(simplified)

        high_quality.extend(selected)
        print(f"Category '{category}': {len(selected)} high-quality examples")

    return high_quality


def create_balanced_validation_set(
    examples: List[Dict[str, Any]], validation_size: int = 100
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create a balanced validation set and return remaining training examples."""

    # Categorize examples by extraction type
    categories = categorize_by_extraction(examples)

    # Select high-quality examples
    high_quality_examples = select_high_quality_examples(categories)

    # Re-categorize high-quality examples
    hq_categories = categorize_by_extraction(high_quality_examples)

    validation_examples = []
    target_per_category = validation_size // 3

    # Select balanced examples for validation
    for category, category_examples in hq_categories.items():
        # Shuffle to ensure randomness
        random.shuffle(category_examples)

        # Take up to target_per_category examples
        selected = category_examples[:target_per_category]
        validation_examples.extend(selected)
        print(
            f"Selected {len(selected)} examples from '{category}' category for validation"
        )

    # Add remaining examples to reach validation_size if needed
    remaining_needed = validation_size - len(validation_examples)
    if remaining_needed > 0:
        # Collect remaining high-quality examples
        validation_ids = {id(ex) for ex in validation_examples}
        remaining_hq = [
            ex for ex in high_quality_examples if id(ex) not in validation_ids
        ]

        # Shuffle and take what we need
        random.shuffle(remaining_hq)
        validation_examples.extend(remaining_hq[:remaining_needed])
        print(
            f"Added {min(remaining_needed, len(remaining_hq))} additional examples to reach target size"
        )

    # Create training set from all remaining examples
    validation_ids = {id(ex) for ex in validation_examples}
    training_examples = [ex for ex in examples if id(ex) not in validation_ids]

    print(f"Final validation set size: {len(validation_examples)}")
    print(f"Final training set size: {len(training_examples)}")

    return validation_examples, training_examples


def save_datasets(
    validation_examples: List[Dict[str, Any]],
    training_examples: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save validation and training datasets in multiple formats."""

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON files
    validation_json_path = output_dir / "validation_examples.json"
    training_json_path = output_dir / "training_examples.json"

    with open(validation_json_path, "w") as f:
        json.dump(validation_examples, f, indent=2)

    with open(training_json_path, "w") as f:
        json.dump(training_examples, f, indent=2)

    # Save as Hugging Face datasets
    validation_df = pd.DataFrame(validation_examples)
    training_df = pd.DataFrame(training_examples)

    validation_dataset = Dataset.from_pandas(validation_df)
    training_dataset = Dataset.from_pandas(training_df)

    validation_dataset.save_to_disk(str(output_dir / "coffee_validation_dataset"))
    training_dataset.save_to_disk(str(output_dir / "coffee_training_dataset"))

    print(f"Datasets saved to {output_dir}")
    print(f"Validation JSON: {validation_json_path}")
    print(f"Training JSON: {training_json_path}")
    print(f"Validation HF Dataset: {output_dir / 'coffee_validation_dataset'}")
    print(f"Training HF Dataset: {output_dir / 'coffee_training_dataset'}")


def analyze_dataset_balance(examples: List[Dict[str, Any]], dataset_name: str) -> None:
    """Analyze and report dataset balance across different dimensions."""
    print(f"\n=== {dataset_name} Analysis ===")

    # Extraction categories
    categories = categorize_by_extraction(examples)
    for category, category_examples in categories.items():
        print(f"{category.capitalize()} extraction: {len(category_examples)} examples")

    # Confidence distribution
    confidences = [ex.get("output", {}).get("confidence", 0.0) for ex in examples]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    print(f"Average confidence: {avg_confidence:.3f}")

    # Grind change distribution
    grind_changes = {}
    for ex in examples:
        grind_change = ex.get("output", {}).get("grind_change", "unknown")
        grind_changes[grind_change] = grind_changes.get(grind_change, 0) + 1

    print("Grind change distribution:")
    for change, count in sorted(grind_changes.items()):
        print(f"  {change}: {count}")


def main() -> None:
    """Main function to create validation dataset."""
    # Set random seed for reproducibility
    random.seed(42)

    # Define paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = data_dir / "processed"

    print("Creating validation dataset for CoffeeRL...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load all data sources
    all_examples = load_all_data_sources(data_dir)

    if not all_examples:
        print("No data found! Please ensure data files exist in the data directory.")
        return

    # Create balanced validation set
    validation_examples, training_examples = create_balanced_validation_set(
        all_examples
    )

    # Analyze datasets
    analyze_dataset_balance(validation_examples, "Validation Dataset")
    analyze_dataset_balance(training_examples, "Training Dataset")

    # Save datasets
    save_datasets(validation_examples, training_examples, output_dir)

    print("\nValidation dataset creation completed successfully!")


if __name__ == "__main__":
    main()
