#!/usr/bin/env python3
"""
Setup script for the community contribution collection system.

This script initializes the community collection system by:
1. Creating the necessary directory structure
2. Generating the web form HTML
3. Creating outreach templates
4. Setting up example CSV processing
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from community_collection import (
    CommunityDataProcessor,
    save_outreach_templates,
    save_web_form,
    create_outreach_templates,
    generate_web_form_html
)


def main():
    """Set up the community collection system."""
    print("🚀 Setting up Community Contribution Collection System...")
    
    # Initialize processor (creates directories)
    processor = CommunityDataProcessor()
    print(f"✅ Created output directory: {processor.output_dir}")
    
    # Generate and save web form
    save_web_form()
    print("✅ Generated web form HTML")
    
    # Generate and save outreach templates
    save_outreach_templates()
    print("✅ Generated outreach templates")
    
    # Create example CSV structure
    create_example_csv()
    print("✅ Created example CSV structure")
    
    # Create usage instructions
    create_usage_instructions()
    print("✅ Created usage instructions")
    
    print("\n🎉 Community collection system setup complete!")
    print("\nNext steps:")
    print("1. Review the generated files in data/community/")
    print("2. Deploy the web form or create a Google Form with the same fields")
    print("3. Share outreach templates with coffee communities")
    print("4. Process responses using the CommunityDataProcessor")


def create_example_csv():
    """Create an example CSV file showing the expected format."""
    import pandas as pd
    
    # Example data matching the form fields
    example_data = [
        {
            "coffee_amount": 20.0,
            "water_amount": 300.0,
            "grind_size": "medium",
            "brew_time": "3:30",
            "taste_notes": "bitter and harsh",
            "adjustment": "coarser",
            "reasoning": "The bitter taste indicates over-extraction. A coarser grind will reduce extraction and improve balance."
        },
        {
            "coffee_amount": 18.0,
            "water_amount": 270.0,
            "grind_size": "medium_fine",
            "brew_time": "2:45",
            "taste_notes": "sour and weak",
            "adjustment": "finer",
            "reasoning": "Sour taste suggests under-extraction. A finer grind will increase extraction and improve flavor."
        },
        {
            "coffee_amount": 22.0,
            "water_amount": 330.0,
            "grind_size": "medium",
            "brew_time": "3:15",
            "taste_notes": "balanced and sweet",
            "adjustment": "no_change",
            "reasoning": "Perfect balance indicates optimal extraction. Current parameters are working well."
        }
    ]
    
    df = pd.DataFrame(example_data)
    output_path = Path("data/community/example_responses.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Created example CSV: {output_path}")


def create_usage_instructions():
    """Create detailed usage instructions."""
    instructions = """# Community Contribution Collection System

## Overview
This system collects and processes V60 brewing data from the coffee community to improve AI brewing recommendations.

## Files Generated

### Web Form
- `contribution_form.html` - Complete HTML form for data collection
- Can be deployed directly or used as a template for Google Forms

### Outreach Templates
- `reddit_post_template.txt` - Post template for r/coffee and r/espresso
- `discord_message_template.txt` - Message for Discord coffee communities
- `coffee_shop_email_template.txt` - Email template for coffee shops
- `slack_message_template.txt` - Message for Slack coffee communities

### Example Data
- `example_responses.csv` - Sample CSV showing expected data format

## Usage

### 1. Data Collection
Deploy the web form or create a Google Form with these fields:
- coffee_amount (number, 10-30g)
- water_amount (number, 150-500g)
- grind_size (select: very_fine, fine, medium_fine, medium, medium_coarse, coarse)
- brew_time (text, format: mm:ss)
- taste_notes (textarea)
- adjustment (select: much_finer, finer, slightly_finer, no_change, slightly_coarser, coarser, much_coarser)
- reasoning (textarea)

### 2. Data Processing
```python
from src.community_collection import CommunityDataProcessor

# Initialize processor
processor = CommunityDataProcessor()

# Process CSV responses from Google Forms
examples = processor.process_csv_responses("path/to/responses.csv")

# Validate and filter examples
valid_examples = processor.validate_and_filter(examples)

# Save processed examples
output_file = processor.save_examples(valid_examples)
print(f"Saved {len(valid_examples)} examples to {output_file}")
```

### 3. Community Outreach
Use the templates in the following order:
1. Start with Discord/Slack for quick responses
2. Post to Reddit for broader reach
3. Email coffee shops for professional contributions

### 4. Data Integration
Processed examples can be merged with existing training data:
```python
import json

# Load existing data
with open("data/coffee_dataset_raw.json", "r") as f:
    existing_data = json.load(f)

# Load community data
with open("data/community/community_examples_YYYYMMDD_HHMMSS.json", "r") as f:
    community_data = json.load(f)

# Combine datasets
combined_data = existing_data + community_data

# Save combined dataset
with open("data/coffee_dataset_combined.json", "w") as f:
    json.dump(combined_data, f, indent=2)
```

## Quality Control
The system includes validation to ensure:
- All required fields are present
- Grind adjustments are valid
- Extraction categories are correctly assigned
- Time formats are consistent
- Confidence scores are reasonable

## Expected Output Format
Each processed example follows this structure:
```json
{
  "input": "V60, 20g coffee, 300g water, medium grind, 3:30 brew time, tastes bitter",
  "output": {
    "grind_change": "coarser_2",
    "reasoning": "The bitter taste indicates over-extraction...",
    "expected_time": "3:00",
    "extraction": "over",
    "confidence": 0.8
  }
}
```

## Target: 200 Community Examples
Goal is to collect 200 high-quality examples from:
- Reddit coffee communities (50-75 examples)
- Discord/Slack groups (25-50 examples)
- Coffee shop baristas (75-100 examples)
- Other coffee forums (25-50 examples)
"""
    
    output_path = Path("data/community/README.md")
    with open(output_path, 'w') as f:
        f.write(instructions)
    print(f"✅ Created usage instructions: {output_path}")


if __name__ == "__main__":
    main() 