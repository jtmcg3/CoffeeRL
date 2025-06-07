"""
Community contribution collection system for V60 brewing data.

This module provides tools for collecting and processing V60 brewing data
from the coffee community through web forms and standardized data processing.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    from .data_collection import validate_example
except ImportError:
    # Fallback for direct execution
    from data_collection import validate_example


class CommunityDataProcessor:
    """Process and validate community-contributed V60 brewing data."""

    def __init__(self, output_dir: str = "data/community"):
        """Initialize the processor with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Mapping from form selections to standardized format
        self.grind_change_map = {
            "much_finer": "finer_4",
            "finer": "finer_2",
            "slightly_finer": "finer_1",
            "no_change": "none",
            "slightly_coarser": "coarser_1",
            "coarser": "coarser_2",
            "much_coarser": "coarser_4",
        }

    def process_csv_responses(self, csv_file: str) -> List[Dict[str, Any]]:
        """
        Process Google Form responses from CSV file.

        Args:
            csv_file: Path to CSV file with form responses

        Returns:
            List of processed examples in standard format
        """
        df = pd.read_csv(csv_file)
        examples = []

        for _, row in df.iterrows():
            try:
                example = self._convert_row_to_example(row)
                if validate_example(example):
                    examples.append(example)
                else:
                    print(f"Invalid example skipped: {row.to_dict()}")
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        return examples

    def _convert_row_to_example(self, row: pd.Series) -> Dict[str, Any]:
        """Convert a form response row to standard example format."""
        # Extract and clean data
        coffee_amount = float(row["coffee_amount"])
        water_amount = float(row["water_amount"])
        grind_size = str(row["grind_size"]).replace("_", "-")
        brew_time = str(row["brew_time"])
        taste_notes = str(row["taste_notes"]).strip()
        adjustment = str(row["adjustment"])
        reasoning = str(row["reasoning"]).strip()

        # Create input text
        input_text = (
            f"V60, {coffee_amount}g coffee, {water_amount}g water, "
            f"{grind_size} grind, {brew_time} brew time, tastes {taste_notes}"
        )

        # Determine extraction category
        extraction = self._determine_extraction(taste_notes)

        # Calculate expected time
        expected_time = self._calculate_expected_time(brew_time, adjustment)

        return {
            "input": input_text,
            "output": {
                "grind_change": self.grind_change_map[adjustment],
                "reasoning": reasoning,
                "expected_time": expected_time,
                "extraction": extraction,
                "confidence": 0.8,  # Default confidence for community examples
            },
        }

    def _determine_extraction(self, taste_notes: str) -> str:
        """Determine extraction category from taste notes."""
        taste_lower = taste_notes.lower()

        # Under-extraction indicators
        under_indicators = ["sour", "weak", "thin", "grassy", "vegetal", "hollow"]
        if any(indicator in taste_lower for indicator in under_indicators):
            return "under"

        # Over-extraction indicators
        over_indicators = ["bitter", "harsh", "astringent", "dry", "burnt"]
        if any(indicator in taste_lower for indicator in over_indicators):
            return "over"

        # Good extraction indicators
        good_indicators = ["balanced", "sweet", "complex", "fruity", "bright"]
        if any(indicator in taste_lower for indicator in good_indicators):
            return "good"

        return "good"  # Default to good if unclear

    def _calculate_expected_time(self, current_time: str, adjustment: str) -> str:
        """Calculate expected brew time after grind adjustment."""
        # Parse current time
        minutes, seconds = self._parse_time(current_time)
        total_seconds = minutes * 60 + seconds

        # Adjust based on grind change (finer = longer, coarser = shorter)
        if "finer" in adjustment:
            if "much" in adjustment:
                total_seconds += 45  # 45 seconds for much finer
            elif "slightly" in adjustment:
                total_seconds += 15  # 15 seconds for slightly finer
            else:
                total_seconds += 30  # 30 seconds for finer
        elif "coarser" in adjustment:
            if "much" in adjustment:
                total_seconds -= 45  # 45 seconds for much coarser
            elif "slightly" in adjustment:
                total_seconds -= 15  # 15 seconds for slightly coarser
            else:
                total_seconds -= 30  # 30 seconds for coarser

        # Convert back to mm:ss format
        new_minutes = total_seconds // 60
        new_seconds = total_seconds % 60
        return f"{new_minutes}:{new_seconds:02d}"

    def _parse_time(self, time_str: str) -> Tuple[int, int]:
        """Parse time string to minutes and seconds."""
        if ":" in time_str:
            parts = time_str.split(":")
            return int(parts[0]), int(parts[1])
        else:
            # Assume it's in seconds
            total_seconds = int(float(time_str))
            return total_seconds // 60, total_seconds % 60

    def save_examples(
        self, examples: List[Dict[str, Any]], filename: str = None
    ) -> str:
        """Save processed examples to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"community_examples_{timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)

        print(f"Saved {len(examples)} examples to {output_path}")
        return str(output_path)

    def validate_and_filter(
        self, examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate examples and filter out invalid ones."""
        valid_examples = []
        invalid_count = 0

        for example in examples:
            if validate_example(example):
                valid_examples.append(example)
            else:
                invalid_count += 1

        print(
            f"Validation complete: {len(valid_examples)} valid, {invalid_count} invalid"
        )
        return valid_examples


def generate_web_form_html() -> str:
    """Generate HTML for the community contribution web form."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V60 Brewing Experience Contribution</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .form-container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h2 {
            color: #8B4513;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }
        input, select, textarea {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        input:focus, select:focus, textarea:focus {
            border-color: #8B4513;
            outline: none;
        }
        textarea {
            height: 80px;
            resize: vertical;
        }
        button {
            background-color: #8B4513;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
        }
        button:hover {
            background-color: #A0522D;
        }
        .help-text {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <h2>☕ V60 Brewing Experience Contribution</h2>
        <p>Help us build a better coffee brewing assistant by sharing your V60 experiences!</p>

        <form id="brewingForm" action="#" method="post">
            <div class="form-group">
                <label for="coffee_amount">Coffee amount (g):</label>
                <input type="number" id="coffee_amount" name="coffee_amount"
                       min="10" max="30" step="0.1" required>
                <div class="help-text">Typical range: 15-25g</div>
            </div>

            <div class="form-group">
                <label for="water_amount">Water amount (g):</label>
                <input type="number" id="water_amount" name="water_amount"
                       min="150" max="500" step="1" required>
                <div class="help-text">Typical range: 200-400g</div>
            </div>

            <div class="form-group">
                <label for="grind_size">Grind size:</label>
                <select id="grind_size" name="grind_size" required>
                    <option value="">Select grind size...</option>
                    <option value="very_fine">Very Fine</option>
                    <option value="fine">Fine</option>
                    <option value="medium_fine">Medium-Fine</option>
                    <option value="medium">Medium</option>
                    <option value="medium_coarse">Medium-Coarse</option>
                    <option value="coarse">Coarse</option>
                </select>
            </div>

            <div class="form-group">
                <label for="brew_time">Brew time (mm:ss):</label>
                <input type="text" id="brew_time" name="brew_time"
                       pattern="[0-9]:[0-5][0-9]" placeholder="2:30" required>
                <div class="help-text">Format: minutes:seconds (e.g., 2:30)</div>
            </div>

            <div class="form-group">
                <label for="taste_notes">Taste notes:</label>
                <textarea id="taste_notes" name="taste_notes" required
                          placeholder="Describe the taste (e.g., sour, bitter, balanced, sweet, fruity)"></textarea>
                <div class="help-text">Be specific about flavors and mouthfeel</div>
            </div>

            <div class="form-group">
                <label for="adjustment">What adjustment would you recommend?</label>
                <select id="adjustment" name="adjustment" required>
                    <option value="">Select adjustment...</option>
                    <option value="much_finer">Much Finer</option>
                    <option value="finer">Finer</option>
                    <option value="slightly_finer">Slightly Finer</option>
                    <option value="no_change">No Change</option>
                    <option value="slightly_coarser">Slightly Coarser</option>
                    <option value="coarser">Coarser</option>
                    <option value="much_coarser">Much Coarser</option>
                </select>
            </div>

            <div class="form-group">
                <label for="reasoning">Why do you recommend this adjustment?</label>
                <textarea id="reasoning" name="reasoning" required
                          placeholder="Explain your reasoning (e.g., 'Too bitter, indicating over-extraction')"></textarea>
                <div class="help-text">Help others understand your thought process</div>
            </div>

            <button type="submit">Submit Brewing Experience</button>
        </form>
    </div>

    <script>
        document.getElementById('brewingForm').addEventListener('submit', function(e) {
            e.preventDefault();
            alert('Thank you for your contribution! Your data will help improve coffee brewing for everyone.');
            // In a real implementation, this would submit to a backend service
        });
    </script>
</body>
</html>"""


def create_outreach_templates() -> Dict[str, str]:
    """Create templates for community outreach messages."""
    return {
        "reddit_post": """# 🔬 Help Build a Better Coffee Brewing Assistant! [Research Project]

Hi r/coffee! I'm working on an AI-powered V60 brewing assistant and need your expertise to make it better.

## What I'm Building
A system that can recommend grind adjustments based on your brewing parameters and taste results. Think of it as having an experienced barista help you dial in your V60.

## How You Can Help
I'm collecting real brewing experiences from the community. If you've brewed a V60 recently and can remember the details, I'd love to hear about it!

**What I need:**
- Coffee amount (g)
- Water amount (g)
- Grind size
- Total brew time
- How it tasted
- What you'd adjust next time

## Why This Matters
Most brewing guides are generic. This project aims to create personalized recommendations based on real experiences from coffee lovers like you.

**Contribute here:** [Form Link]

Thanks for helping make better coffee for everyone! ☕

*This is for a research project on coffee brewing optimization. All data will be anonymized and used solely for improving brewing recommendations.*""",
        "discord_message": """Hey coffee friends! 👋

I'm building an AI brewing assistant and could use your V60 experiences to train it.

Quick ask: If you've made a V60 recently, could you share:
• Coffee/water amounts
• Grind size & brew time
• How it tasted
• What you'd change

Takes 2 mins: [Form Link]

Goal is to create better, personalized brewing advice based on real experiences rather than generic guides.

Thanks! ☕🤖""",
        "coffee_shop_email": """Subject: Research Collaboration - V60 Brewing Data Collection

Dear [Coffee Shop Name] Team,

I hope this email finds you well. I'm reaching out regarding a research project on V60 brewing optimization that I believe would interest your team.

## Project Overview
I'm developing an AI-powered brewing assistant that provides personalized grind adjustment recommendations based on brewing parameters and taste outcomes. The goal is to help coffee enthusiasts achieve better extractions more consistently.

## How You Can Help
I'm seeking experienced baristas and coffee professionals to contribute brewing data. This involves sharing details from V60 brews including:
- Brewing parameters (coffee/water ratio, grind size, time)
- Taste assessment
- Recommended adjustments

## What's In It For You
- Early access to the brewing assistant tool
- Recognition as a contributing partner
- Insights from the aggregated brewing data
- Potential collaboration opportunities

The data collection process is simple and takes just a few minutes per contribution. All data is anonymized and used solely for improving brewing recommendations.

Would you be interested in participating? I'd be happy to discuss this further and answer any questions.

Best regards,
[Your Name]
[Contact Information]

Contribution form: [Form Link]""",
        "slack_message": """🔬 **Coffee Research Project** ☕

Building an AI V60 brewing assistant and need real brewing data from experienced coffee people!

**Quick contribution needed:**
Share details from your recent V60 brews (ratios, grind, time, taste, what you'd adjust)

**Why:** Create personalized brewing recommendations vs generic guides

**Time:** 2-3 minutes per contribution

**Link:** [Form Link]

Thanks for helping improve coffee for everyone! 🙏""",
    }


def save_outreach_templates(output_dir: str = "data/community") -> None:
    """Save outreach templates to files."""
    templates = create_outreach_templates()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, content in templates.items():
        file_path = output_path / f"{name}_template.txt"
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Saved {name} template to {file_path}")


def save_web_form(output_dir: str = "data/community") -> None:
    """Save the web form HTML to a file."""
    html_content = generate_web_form_html()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / "contribution_form.html"
    with open(file_path, "w") as f:
        f.write(html_content)
    print(f"Saved web form to {file_path}")


if __name__ == "__main__":
    # Example usage
    processor = CommunityDataProcessor()

    # Save templates and form
    save_outreach_templates()
    save_web_form()

    print("Community contribution collection system ready!")
    print("Next steps:")
    print("1. Deploy the web form or create a Google Form with the same fields")
    print("2. Share outreach templates with coffee communities")
    print("3. Process responses using CommunityDataProcessor.process_csv_responses()")
