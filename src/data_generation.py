"""
Data generation script for V60 brewing scenarios using GPT-3.5.

This module generates training data by calling OpenAI's GPT-3.5 API
to create realistic V60 brewing problems and solutions.
"""

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from datasets import Dataset
from openai import OpenAI

try:
    from .data_collection import validate_example
except ImportError:
    from data_collection import validate_example  # type: ignore


class DataGenerator:
    """Generates V60 brewing training data using GPT-3.5."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key."""
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.generation_template = self._create_generation_template()

    def _create_generation_template(self) -> str:
        """Create the prompt template for GPT-3.5 generation."""
        return """Generate a realistic V60 brewing problem and solution.

Create a scenario where someone has brewing parameters and a taste issue, then provide the correct grind adjustment.

Format your response as JSON with this exact structure:
{
  "input": "V60, [coffee]g coffee, [water]g water, [grind] grind, [time] brew time, tastes [taste]",
  "output": {
    "grind_change": "[finer/coarser/none]_[number]",
    "reasoning": "[explanation of why this adjustment helps]",
    "expected_time": "[expected brew time after adjustment]",
    "extraction": "[under/good/over]",
    "confidence": [0.0-1.0]
  }
}

Make the scenario realistic with:
- Coffee: 15-25g
- Water: 250-400g
- Grind: fine/medium-fine/medium/medium-coarse/coarse
- Time: 1:30-5:00
- Taste: bitter, sour, weak, astringent, balanced, etc.

Generate one complete example now:"""

    def generate_single_example(
        self, max_retries: int = 3, delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """Generate a single V60 brewing example using GPT-3.5."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": self.generation_template}],
                    temperature=0.7,
                    max_tokens=200,
                )

                content = response.choices[0].message.content
                if not content:
                    continue

                parsed = self._parse_response(content)
                if parsed and validate_example(parsed):
                    return parsed

            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)

        return None

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse GPT-3.5 response into structured data."""
        try:
            # Extract JSON from response
            content = response.strip()

            # Handle code block formatting
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed_json: Dict[str, Any] = json.loads(content)
            return parsed_json

        except json.JSONDecodeError:
            # Fallback: try to extract structured data from text
            return self._parse_text_response(response)

    def _parse_text_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Fallback parser for non-JSON responses."""
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        input_text = ""
        output_data: Dict[str, Any] = {}

        for line in lines:
            if line.startswith("Input:"):
                input_text = line.replace("Input:", "").strip().strip('"')
            elif line.startswith("Output:"):
                try:
                    output_json = line.replace("Output:", "").strip()
                    output_data = json.loads(output_json)
                except json.JSONDecodeError:
                    continue

        if input_text and output_data:
            return {"input": input_text, "output": output_data}

        return None

    def generate_batch(
        self,
        n_examples: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a batch of V60 brewing examples."""
        examples = []
        failed_attempts = 0
        max_failed = n_examples // 2  # Allow some failures

        for i in range(n_examples):
            if progress_callback:
                progress_callback(i + 1, n_examples)
            else:
                print(f"Generating example {i + 1}/{n_examples}")

            example = self.generate_single_example()
            if example:
                examples.append(example)
            else:
                failed_attempts += 1
                if failed_attempts > max_failed:
                    print(
                        f"Too many failed attempts ({failed_attempts}). Stopping generation."
                    )
                    break

        print(f"Generated {len(examples)} valid examples ({failed_attempts} failed)")
        return examples

    def save_dataset(
        self, examples: List[Dict[str, Any]], output_dir: str = "data"
    ) -> Dataset:
        """Save examples as both JSON and HuggingFace dataset."""
        os.makedirs(output_dir, exist_ok=True)

        # Save raw JSON
        json_path = os.path.join(output_dir, "coffee_dataset_raw.json")
        with open(json_path, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"Saved raw data to {json_path}")

        # Create HuggingFace dataset
        df = pd.DataFrame(examples)
        dataset = Dataset.from_pandas(df)

        dataset_path = os.path.join(output_dir, "coffee_dataset")
        dataset.save_to_disk(dataset_path)
        print(f"Saved HuggingFace dataset to {dataset_path}")

        return dataset


def create_training_dataset(n_examples: int = 600, output_dir: str = "data") -> Dataset:
    """Create a complete training dataset with GPT-3.5 generated examples."""
    generator = DataGenerator()

    print(f"Generating {n_examples} V60 brewing examples...")
    examples = generator.generate_batch(n_examples)

    if not examples:
        raise RuntimeError("Failed to generate any valid examples")

    dataset = generator.save_dataset(examples, output_dir)
    print(f"Dataset created with {len(dataset)} examples")

    return dataset


if __name__ == "__main__":
    # Generate test batch first
    print("Generating test batch...")
    test_generator = DataGenerator()
    test_examples = test_generator.generate_batch(10)

    if test_examples:
        print(f"Test successful! Generated {len(test_examples)} examples")
        print("Sample example:")
        print(json.dumps(test_examples[0], indent=2))

        # Generate full dataset
        dataset = create_training_dataset()
    else:
        print("Test failed. Check your OpenAI API key and try again.")
