#!/usr/bin/env python3
"""
Standalone script to generate V60 brewing training data using GPT-3.5.

Usage:
    python scripts/generate_data.py --examples 600 --output data/
    python scripts/generate_data.py --test  # Generate test batch only
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generation import DataGenerator, create_training_dataset  # noqa: E402


def main() -> None:
    """Main entry point for data generation script."""
    parser = argparse.ArgumentParser(description="Generate V60 brewing training data")
    parser.add_argument(
        "--examples",
        type=int,
        default=600,
        help="Number of examples to generate (default: 600)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory for generated data (default: data/)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Generate test batch only (10 examples)"
    )
    parser.add_argument(
        "--api-key", type=str, help="OpenAI API key (overrides OPENAI_API_KEY env var)"
    )

    args = parser.parse_args()

    # Check for API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api-key"
        )
        sys.exit(1)

    try:
        if args.test:
            print("Generating test batch (10 examples)...")
            generator = DataGenerator(api_key=api_key)
            examples = generator.generate_batch(10)

            if examples:
                print(f"✅ Test successful! Generated {len(examples)} examples")
                print("\nSample example:")
                import json

                print(json.dumps(examples[0], indent=2))
            else:
                print("❌ Test failed. No valid examples generated.")
                sys.exit(1)
        else:
            print(f"Generating {args.examples} training examples...")
            dataset = create_training_dataset(
                n_examples=args.examples, output_dir=args.output
            )
            print(f"✅ Successfully created dataset with {len(dataset)} examples")
            print(f"📁 Data saved to: {args.output}/")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
