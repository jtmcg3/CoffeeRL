#!/usr/bin/env python3
"""
Demonstration of the data collection templates for V60 brewing scenarios.

This script shows how to use the various functions in the data_collection module
to create, validate, and work with brewing data examples.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_collection import (  # noqa: E402
    create_manual_example,
    extract_brewing_parameters,
    get_gpt_generation_prompt,
    validate_batch_examples,
    validate_example,
)


def demo_manual_example_creation() -> List[Dict[str, Any]]:
    """Demonstrate creating manual examples with different scenarios."""
    print("=== Manual Example Creation Demo ===\n")

    # Example 1: Balanced brew
    print("1. Balanced brew example:")
    example1 = create_manual_example(
        coffee_amount=15.0,
        water_amount=250.0,
        grind_size="medium",
        brew_time="2:30",
        taste_notes="balanced and sweet",
    )
    print(f"Input: {example1['input']}")
    print(f"Grind change: {example1['output']['grind_change']}")
    print(f"Reasoning: {example1['output']['reasoning']}\n")

    # Example 2: Over-extracted (bitter)
    print("2. Over-extracted (bitter) example:")
    example2 = create_manual_example(
        coffee_amount=20.0,
        water_amount=300.0,
        grind_size="fine",
        brew_time="4:30",
        taste_notes="bitter and harsh",
    )
    print(f"Input: {example2['input']}")
    print(f"Grind change: {example2['output']['grind_change']}")
    print(f"Reasoning: {example2['output']['reasoning']}\n")

    # Example 3: Under-extracted (sour)
    print("3. Under-extracted (sour) example:")
    example3 = create_manual_example(
        coffee_amount=12.0,
        water_amount=200.0,
        grind_size="coarse",
        brew_time="1:45",
        taste_notes="sour and weak",
    )
    print(f"Input: {example3['input']}")
    print(f"Grind change: {example3['output']['grind_change']}")
    print(f"Reasoning: {example3['output']['reasoning']}\n")

    return [example1, example2, example3]


def demo_gpt_prompt_generation() -> None:
    """Demonstrate GPT prompt generation for different scenarios."""
    print("=== GPT Prompt Generation Demo ===\n")

    print("1. Default prompt (with issue):")
    prompt1 = get_gpt_generation_prompt()
    print(prompt1[:200] + "...\n")

    print("2. Balanced scenario prompt:")
    prompt2 = get_gpt_generation_prompt("balanced")
    print(prompt2[-100:] + "\n")

    print("3. Random scenario prompt:")
    prompt3 = get_gpt_generation_prompt("random")
    print(prompt3[-100:] + "\n")


def demo_validation() -> None:
    """Demonstrate example validation."""
    print("=== Validation Demo ===\n")

    # Valid example
    valid_example = {
        "input": "V60, 15g coffee, 250g water, medium grind, 2:30 time, tastes balanced",
        "output": {
            "grind_change": "none",
            "reasoning": "Current parameters produce balanced extraction.",
        },
    }

    # Invalid example (missing required field)
    invalid_example = {
        "input": "V60, 15g coffee, medium grind, tastes balanced",  # Missing water and time
        "output": {"grind_change": "none", "reasoning": "Test reasoning"},
    }

    print("1. Valid example validation:")
    print(f"Is valid: {validate_example(valid_example)}\n")

    print("2. Invalid example validation:")
    print(f"Is valid: {validate_example(invalid_example)}\n")

    # Batch validation
    examples = [valid_example, invalid_example, valid_example]
    print("3. Batch validation results:")
    results = validate_batch_examples(examples)
    print(f"Total: {results['total']}")
    print(f"Valid: {results['valid']}")
    print(f"Invalid: {results['invalid']}")
    print(f"Invalid indices: {results['invalid_indices']}\n")


def demo_parameter_extraction() -> None:
    """Demonstrate parameter extraction from text."""
    print("=== Parameter Extraction Demo ===\n")

    test_texts = [
        "V60, 15g coffee, 250g water, medium grind, 2:30 brew time, tastes balanced",
        "V60 with 18.5g coffee, fine grind, 3 minutes total, tastes bitter",
        "Coarse grind, 20g coffee, tastes sour and weak",
        "V60, 280g water, extra fine grind, 4:15 time, over-extracted",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"{i}. Text: {text}")
        params = extract_brewing_parameters(text)
        print(f"   Extracted parameters: {params}\n")


def demo_integration_workflow() -> None:
    """Demonstrate a complete workflow using multiple functions."""
    print("=== Integration Workflow Demo ===\n")

    # Step 1: Extract parameters from user input
    user_input = (
        "V60, 17g coffee, 270g water, fine grind, 3:45 brew time, tastes bitter"
    )
    print(f"1. User input: {user_input}")

    params = extract_brewing_parameters(user_input)
    print(f"2. Extracted parameters: {params}")

    # Step 2: Create structured example
    if all(
        key in params
        for key in [
            "coffee_amount",
            "water_amount",
            "grind_size",
            "brew_time",
            "taste_notes",
        ]
    ):
        example = create_manual_example(
            coffee_amount=float(params["coffee_amount"]),
            water_amount=float(params["water_amount"]),
            grind_size=params["grind_size"],
            brew_time=params["brew_time"],
            taste_notes=params["taste_notes"],
        )
        print(f"3. Created example: {example}")

        # Step 3: Validate the example
        is_valid = validate_example(example)
        print(f"4. Example is valid: {is_valid}")

        # Step 4: Show recommendation
        print(f"5. Recommendation: {example['output']['grind_change']}")
        print(f"6. Reasoning: {example['output']['reasoning']}")
    else:
        print("3. Could not create example - missing required parameters")


def main() -> None:
    """Run all demonstrations."""
    print("CoffeeRL Data Collection Templates Demo")
    print("=" * 50)
    print()

    # Run all demos
    demo_manual_example_creation()
    demo_gpt_prompt_generation()
    demo_validation()
    demo_parameter_extraction()
    demo_integration_workflow()

    print("Demo completed! 🎉")
    print("\nThe data collection templates are ready for use in the CoffeeRL project.")


if __name__ == "__main__":
    main()
