"""
Data collection templates for V60 brewing scenarios.

This module provides templates and utilities for collecting V60 brewing data
from various sources including manual creation, GPT-3.5 generation, and
community contributions.
"""

import re
from typing import Any, Dict, List, Union


def create_manual_example(
    coffee_amount: float,
    water_amount: float,
    grind_size: str,
    brew_time: str,
    taste_notes: str,
) -> Dict[str, Any]:
    """
    Create a manual data collection example for V60 brewing.

    Args:
        coffee_amount: Amount of coffee in grams
        water_amount: Amount of water in grams
        grind_size: Description of grind size (e.g., "medium", "fine")
        brew_time: Total brew time (e.g., "2:30", "3:00")
        taste_notes: Taste description (e.g., "bitter", "sour", "balanced")

    Returns:
        Dictionary with input string and output assessment
    """
    input_text = (
        f"V60, {coffee_amount}g coffee, {water_amount}g water, "
        f"{grind_size} grind, {brew_time} brew time, tastes {taste_notes}"
    )

    return {
        "input": input_text,
        "output": {
            "grind_change": _determine_grind_change(grind_size, brew_time, taste_notes),
            "reasoning": _generate_reasoning(grind_size, brew_time, taste_notes),
        },
    }


def _determine_grind_change(grind_size: str, brew_time: str, taste_notes: str) -> str:
    """Determine recommended grind adjustment based on brewing parameters."""
    # Convert brew time to seconds for analysis
    brew_seconds = _parse_brew_time(brew_time)

    # Basic logic for grind adjustments
    if "bitter" in taste_notes.lower() or "over-extracted" in taste_notes.lower():
        return "coarser"
    elif "sour" in taste_notes.lower() or "under-extracted" in taste_notes.lower():
        return "finer"
    elif brew_seconds > 300:  # Over 5 minutes
        return "coarser"
    elif brew_seconds < 120:  # Under 2 minutes
        return "finer"
    else:
        return "none"


def _generate_reasoning(grind_size: str, brew_time: str, taste_notes: str) -> str:
    """Generate reasoning for the grind change recommendation."""
    brew_seconds = _parse_brew_time(brew_time)

    if "bitter" in taste_notes.lower():
        return "Bitter taste indicates over-extraction. A coarser grind will slow extraction."
    elif "sour" in taste_notes.lower():
        return "Sour taste indicates under-extraction. A finer grind will increase extraction."
    elif brew_seconds > 300:
        return (
            f"Brew time of {brew_time} is too long. Coarser grind will speed up flow."
        )
    elif brew_seconds < 120:
        return f"Brew time of {brew_time} is too fast. Finer grind will slow flow."
    else:
        return "Current parameters produce balanced extraction. No grind change needed."


def _parse_brew_time(brew_time: str) -> int:
    """Parse brew time string (e.g., '2:30') to seconds."""
    if ":" in brew_time:
        parts = brew_time.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    else:
        # Assume it's already in seconds or minutes
        try:
            return int(float(brew_time))
        except ValueError:
            return 180  # Default to 3 minutes


# GPT-3.5 generation template
PROMPT_TEMPLATE = """
You are a coffee expert. Generate a realistic V60 brewing scenario.

Example format:
Input: "V60, 15g coffee, 250g water, medium grind, 2:30 total time, tastes balanced"
Output: {
    "assessment": "good extraction",
    "suggestion": "maintain current grind",
    "grind_change": "none",
    "expected_time": "2:30",
    "extraction": "good",
    "confidence": 0.9,
    "reasoning": "The brew time and taste indicate proper extraction."
}

Now generate a new scenario with an issue:
"""


def get_gpt_generation_prompt(scenario_type: str = "issue") -> str:
    """
    Get the prompt template for GPT-3.5 generation.

    Args:
        scenario_type: Type of scenario to generate ("issue", "balanced", "random")

    Returns:
        Formatted prompt string for GPT-3.5
    """
    base_prompt = PROMPT_TEMPLATE

    if scenario_type == "balanced":
        return base_prompt.replace("with an issue", "that is well-balanced")
    elif scenario_type == "random":
        return base_prompt.replace("with an issue", "with random parameters")
    else:
        return base_prompt


def validate_example(example: Any) -> bool:
    """
    Validate that a brewing example contains all required fields and data.

    Args:
        example: Data to validate (should be a dictionary)

    Returns:
        True if example is valid, False otherwise
    """
    # Check basic structure
    if not isinstance(example, dict):
        return False

    if "input" not in example or "output" not in example:
        return False

    # Check input contains required brewing parameters
    input_text = example["input"].lower()
    required_input_fields = ["coffee", "water", "time", "tastes"]

    for field in required_input_fields:
        if field not in input_text:
            return False

    # Check output structure
    output_data = example["output"]
    if not isinstance(output_data, dict):
        return False

    required_output_fields = ["grind_change", "reasoning"]
    for field in required_output_fields:
        if field not in output_data:
            return False

    # Validate grind_change values (allow both simple and detailed formats)
    grind_change = output_data["grind_change"]
    if "_" in grind_change:
        # Format: "finer_2" or "coarser_1"
        direction = grind_change.split("_")[0]
        valid_directions = ["finer", "coarser", "none"]
        if direction not in valid_directions:
            return False
    else:
        # Simple format: "finer", "coarser", "none"
        valid_grind_changes = ["finer", "coarser", "none"]
        if grind_change not in valid_grind_changes:
            return False

    # Validate optional fields if present
    if "confidence" in output_data:
        confidence = output_data["confidence"]
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            return False

    if "extraction" in output_data:
        valid_extractions = ["under", "good", "over"]
        if output_data["extraction"] not in valid_extractions:
            return False

    return True


def validate_batch_examples(
    examples: List[Dict[str, Any]],
) -> Dict[str, Union[int, List[int]]]:
    """
    Validate a batch of examples and return validation results.

    Args:
        examples: List of example dictionaries to validate

    Returns:
        Dictionary with validation statistics and invalid example indices
    """
    invalid_indices: List[int] = []
    valid_count = 0
    invalid_count = 0

    for i, example in enumerate(examples):
        if validate_example(example):
            valid_count += 1
        else:
            invalid_count += 1
            invalid_indices.append(i)

    return {
        "total": len(examples),
        "valid": valid_count,
        "invalid": invalid_count,
        "invalid_indices": invalid_indices,
    }


def extract_brewing_parameters(input_text: str) -> Dict[str, str]:
    """
    Extract brewing parameters from input text using regex patterns.

    Args:
        input_text: Raw input text describing brewing scenario

    Returns:
        Dictionary with extracted parameters
    """
    parameters = {}

    # Extract coffee amount
    coffee_match = re.search(r"(\d+(?:\.\d+)?)g?\s*coffee", input_text.lower())
    if coffee_match:
        parameters["coffee_amount"] = coffee_match.group(1)

    # Extract water amount
    water_match = re.search(r"(\d+(?:\.\d+)?)g?\s*water", input_text.lower())
    if water_match:
        parameters["water_amount"] = water_match.group(1)

    # Extract grind size
    grind_match = re.search(
        r"(fine|medium|coarse|extra fine|extra coarse)\s*grind", input_text.lower()
    )
    if grind_match:
        parameters["grind_size"] = grind_match.group(1)

    # Extract brew time
    time_match = re.search(r"(\d+:\d+|\d+\s*(?:minutes?|min))", input_text.lower())
    if time_match:
        parameters["brew_time"] = time_match.group(1)

    # Extract taste notes
    taste_match = re.search(r"tastes?\s+([^,]+)", input_text.lower())
    if taste_match:
        parameters["taste_notes"] = taste_match.group(1).strip()

    return parameters
