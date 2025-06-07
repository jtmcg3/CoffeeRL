#!/usr/bin/env python3
"""
Test script for community data processing functionality.

This script demonstrates how to process community-contributed V60 brewing data
and validates the processing pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from community_collection import CommunityDataProcessor  # noqa: E402


def main():
    """Test the community data processing functionality."""
    print("🧪 Testing Community Data Processing...")

    # Initialize processor
    processor = CommunityDataProcessor()
    print(f"✅ Initialized processor with output directory: {processor.output_dir}")

    # Test with example CSV
    example_csv = "data/community/example_responses.csv"
    print(f"\n📊 Processing example CSV: {example_csv}")

    try:
        # Process the example responses
        examples = processor.process_csv_responses(example_csv)
        print(f"✅ Processed {len(examples)} examples from CSV")

        # Display processed examples
        print("\n📋 Processed Examples:")
        for i, example in enumerate(examples, 1):
            print(f"\n--- Example {i} ---")
            print(f"Input: {example['input']}")
            print(f"Grind Change: {example['output']['grind_change']}")
            print(f"Extraction: {example['output']['extraction']}")
            print(f"Expected Time: {example['output']['expected_time']}")
            print(f"Reasoning: {example['output']['reasoning'][:100]}...")

        # Validate examples
        print("\n🔍 Validating examples...")
        valid_examples = processor.validate_and_filter(examples)
        print(f"✅ Validation complete: {len(valid_examples)} valid examples")

        # Save processed examples
        print("\n💾 Saving processed examples...")
        output_file = processor.save_examples(
            valid_examples, "test_processed_examples.json"
        )
        print(f"✅ Saved examples to: {output_file}")

        # Test individual processing methods
        test_individual_methods(processor)

        print("\n🎉 All tests passed! Community data processing is working correctly.")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False

    return True


def test_individual_methods(processor):
    """Test individual processing methods."""
    print("\n🔧 Testing individual processing methods...")

    # Test extraction determination
    test_cases = [
        ("bitter and harsh", "over"),
        ("sour and weak", "under"),
        ("balanced and sweet", "good"),
        ("fruity and complex", "good"),
        ("astringent and dry", "over"),
        ("grassy and thin", "under"),
    ]

    print("Testing extraction determination:")
    for taste_notes, expected in test_cases:
        result = processor._determine_extraction(taste_notes)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{taste_notes}' -> {result} (expected: {expected})")

    # Test time calculation
    print("\nTesting time calculation:")
    time_cases = [
        ("3:00", "finer", "3:30"),
        ("2:30", "coarser", "2:00"),
        ("4:00", "much_finer", "4:45"),
        ("3:30", "no_change", "3:30"),
    ]

    for current_time, adjustment, expected in time_cases:
        result = processor._calculate_expected_time(current_time, adjustment)
        status = "✅" if result == expected else "❌"
        print(
            f"  {status} {current_time} + {adjustment} -> {result} (expected: {expected})"
        )

    # Test time parsing
    print("\nTesting time parsing:")
    parse_cases = [("2:30", (2, 30)), ("4:15", (4, 15)), ("180", (3, 0))]

    for time_str, expected in parse_cases:
        result = processor._parse_time(time_str)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{time_str}' -> {result} (expected: {expected})")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
