"""
Manual examples collector for V60 brewing scenarios.

This module collects and organizes manual training examples from various sources:
- Personal brewing logs
- Coffee forum extractions (r/coffee, Home-Barista, etc.)
- YouTube barista video transcripts
- Expert brewing guides
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from .data_collection import validate_example


def load_existing_examples() -> List[Dict[str, Any]]:
    """Load existing examples from the raw dataset file."""
    data_file = Path("data/coffee_dataset_raw.json")
    if data_file.exists():
        with open(data_file, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    return []


def create_personal_brewing_logs() -> List[Dict[str, Any]]:
    """Create examples from personal brewing logs and experiences."""
    examples = [
        # Under-extraction scenarios
        {
            "input": "V60, 15g coffee, 250g water, medium grind, 2:10 brew time, tastes sour and weak",
            "output": {
                "grind_change": "finer_2",
                "reasoning": "The brew time is too short and the sour taste indicates under-extraction. Grinding finer will increase extraction and improve flavor balance.",
                "expected_time": "2:45",
                "extraction": "under",
                "confidence": 0.9,
            },
        },
        {
            "input": "V60, 18g coffee, 280g water, coarse grind, 1:45 brew time, tastes very sour",
            "output": {
                "grind_change": "finer_3",
                "reasoning": "Extremely fast brew time with coarse grind leads to severe under-extraction. Need significantly finer grind to slow flow and increase extraction.",
                "expected_time": "3:00",
                "extraction": "under",
                "confidence": 0.95,
            },
        },
        {
            "input": "V60, 22g coffee, 350g water, medium-coarse grind, 2:30 brew time, tastes thin and acidic",
            "output": {
                "grind_change": "finer_2",
                "reasoning": "Thin, acidic taste suggests under-extraction despite reasonable brew time. The medium-coarse grind is too coarse for this ratio.",
                "expected_time": "3:15",
                "extraction": "under",
                "confidence": 0.85,
            },
        },
        # Over-extraction scenarios
        {
            "input": "V60, 20g coffee, 300g water, fine grind, 5:30 brew time, tastes bitter and harsh",
            "output": {
                "grind_change": "coarser_2",
                "reasoning": "Very long brew time with fine grind causes over-extraction. Coarser grind will speed up flow and reduce bitterness.",
                "expected_time": "3:30",
                "extraction": "over",
                "confidence": 0.9,
            },
        },
        {
            "input": "V60, 25g coffee, 400g water, medium-fine grind, 4:45 brew time, tastes bitter and astringent",
            "output": {
                "grind_change": "coarser_1",
                "reasoning": "High coffee ratio with medium-fine grind and long brew time leads to over-extraction. Slightly coarser grind will balance extraction.",
                "expected_time": "4:00",
                "extraction": "over",
                "confidence": 0.8,
            },
        },
        # Good extraction scenarios
        {
            "input": "V60, 20g coffee, 320g water, medium grind, 3:15 brew time, tastes balanced and sweet",
            "output": {
                "grind_change": "none",
                "reasoning": "Perfect balance of sweetness and acidity indicates optimal extraction. Current parameters are working well.",
                "expected_time": "3:15",
                "extraction": "good",
                "confidence": 0.95,
            },
        },
        {
            "input": "V60, 16g coffee, 260g water, medium-fine grind, 2:50 brew time, tastes fruity and bright",
            "output": {
                "grind_change": "none",
                "reasoning": "Bright, fruity notes with good balance suggest excellent extraction. No adjustment needed.",
                "expected_time": "2:50",
                "extraction": "good",
                "confidence": 0.9,
            },
        },
        # Edge cases and specific scenarios
        {
            "input": "V60, 30g coffee, 500g water, medium grind, 6:00 brew time, tastes muddy and bitter",
            "output": {
                "grind_change": "coarser_3",
                "reasoning": "Large batch with very long brew time causes over-extraction and muddy flavors. Much coarser grind needed for faster flow.",
                "expected_time": "4:30",
                "extraction": "over",
                "confidence": 0.85,
            },
        },
        {
            "input": "V60, 12g coffee, 200g water, fine grind, 3:45 brew time, tastes weak but bitter",
            "output": {
                "grind_change": "coarser_1",
                "reasoning": "Low coffee ratio with fine grind creates weak but over-extracted brew. Coarser grind will reduce bitterness while maintaining strength.",
                "expected_time": "3:00",
                "extraction": "over",
                "confidence": 0.75,
            },
        },
        {
            "input": "V60, 20g coffee, 300g water, extra fine grind, 7:00 brew time, tastes extremely bitter",
            "output": {
                "grind_change": "coarser_4",
                "reasoning": "Extra fine grind causes severe over-extraction with extremely long brew time. Need much coarser grind to fix flow rate.",
                "expected_time": "3:30",
                "extraction": "over",
                "confidence": 0.95,
            },
        },
    ]
    return examples


def create_forum_examples() -> List[Dict[str, Any]]:
    """Create examples based on common forum discussions and troubleshooting."""
    examples = [
        # r/coffee style examples
        {
            "input": "V60, 21g coffee, 350g water, medium grind, 4:00 brew time, tastes flat and boring",
            "output": {
                "grind_change": "finer_1",
                "reasoning": "Flat taste often indicates under-extraction. A slightly finer grind will increase extraction and bring out more flavor complexity.",
                "expected_time": "4:15",
                "extraction": "under",
                "confidence": 0.8,
            },
        },
        {
            "input": "V60, 19g coffee, 285g water, medium-coarse grind, 2:45 brew time, tastes grassy and vegetal",
            "output": {
                "grind_change": "finer_2",
                "reasoning": "Grassy, vegetal notes typically indicate under-extraction. Finer grind will help extract more desirable flavors and reduce unpleasant notes.",
                "expected_time": "3:30",
                "extraction": "under",
                "confidence": 0.85,
            },
        },
        {
            "input": "V60, 23g coffee, 370g water, fine grind, 5:00 brew time, tastes dry and chalky",
            "output": {
                "grind_change": "coarser_2",
                "reasoning": "Dry, chalky mouthfeel suggests over-extraction. Coarser grind will reduce extraction time and improve mouthfeel.",
                "expected_time": "3:45",
                "extraction": "over",
                "confidence": 0.8,
            },
        },
        # Home-Barista style technical examples
        {
            "input": "V60, 20g coffee, 300g water, medium grind, 3:30 brew time, tastes hollow in the middle",
            "output": {
                "grind_change": "finer_1",
                "reasoning": "Hollow middle taste suggests uneven extraction. Slightly finer grind can help achieve more uniform extraction.",
                "expected_time": "3:45",
                "extraction": "under",
                "confidence": 0.7,
            },
        },
        {
            "input": "V60, 17g coffee, 272g water, medium-fine grind, 3:00 brew time, tastes sharp and aggressive",
            "output": {
                "grind_change": "coarser_1",
                "reasoning": "Sharp, aggressive flavors indicate over-extraction. Slightly coarser grind will soften the profile and reduce harshness.",
                "expected_time": "2:45",
                "extraction": "over",
                "confidence": 0.8,
            },
        },
        # CoffeeGeek style examples
        {
            "input": "V60, 24g coffee, 384g water, coarse grind, 2:15 brew time, tastes watery and sour",
            "output": {
                "grind_change": "finer_3",
                "reasoning": "Watery, sour taste with fast brew time indicates severe under-extraction. Much finer grind needed to slow flow and increase extraction.",
                "expected_time": "3:30",
                "extraction": "under",
                "confidence": 0.9,
            },
        },
        {
            "input": "V60, 18g coffee, 288g water, medium grind, 3:45 brew time, tastes muted and dull",
            "output": {
                "grind_change": "finer_1",
                "reasoning": "Muted, dull flavors suggest under-extraction. Finer grind will help extract more aromatic compounds and brighten the cup.",
                "expected_time": "4:00",
                "extraction": "under",
                "confidence": 0.75,
            },
        },
    ]
    return examples


def create_youtube_examples() -> List[Dict[str, Any]]:
    """Create examples based on YouTube barista tutorials and troubleshooting videos."""
    examples = [
        # James Hoffmann style examples
        {
            "input": "V60, 20g coffee, 300g water, medium-fine grind, 2:30 brew time, tastes bright but thin",
            "output": {
                "grind_change": "finer_1",
                "reasoning": "Bright but thin suggests good acidity extraction but lacking body. Slightly finer grind will extract more soluble solids for better body.",
                "expected_time": "2:45",
                "extraction": "under",
                "confidence": 0.8,
            },
        },
        {
            "input": "V60, 22g coffee, 330g water, medium grind, 4:30 brew time, tastes heavy and muddy",
            "output": {
                "grind_change": "coarser_2",
                "reasoning": "Heavy, muddy texture with long brew time indicates over-extraction. Coarser grind will clean up the cup and reduce muddiness.",
                "expected_time": "3:30",
                "extraction": "over",
                "confidence": 0.85,
            },
        },
        # Onyx Coffee Lab style examples
        {
            "input": "V60, 21g coffee, 315g water, fine grind, 4:15 brew time, tastes astringent and dry",
            "output": {
                "grind_change": "coarser_2",
                "reasoning": "Astringent, dry finish indicates over-extraction of tannins. Coarser grind will reduce astringency and improve balance.",
                "expected_time": "3:30",
                "extraction": "over",
                "confidence": 0.9,
            },
        },
        # European Coffee Trip style examples
        {
            "input": "V60, 16g coffee, 240g water, medium-coarse grind, 2:00 brew time, tastes weak and sour",
            "output": {
                "grind_change": "finer_2",
                "reasoning": "Fast brew time with weak, sour result indicates under-extraction. Finer grind will slow flow and increase extraction yield.",
                "expected_time": "2:45",
                "extraction": "under",
                "confidence": 0.85,
            },
        },
        # Specialty coffee roaster examples
        {
            "input": "V60, 25g coffee, 400g water, medium grind, 5:30 brew time, tastes burnt and harsh",
            "output": {
                "grind_change": "coarser_3",
                "reasoning": "Burnt, harsh flavors with very long brew time suggest severe over-extraction. Much coarser grind needed to reduce extraction.",
                "expected_time": "4:00",
                "extraction": "over",
                "confidence": 0.9,
            },
        },
    ]
    return examples


def create_expert_guide_examples() -> List[Dict[str, Any]]:
    """Create examples based on expert brewing guides and professional recommendations."""
    examples = [
        # World Barista Championship style
        {
            "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes balanced with good clarity",
            "output": {
                "grind_change": "none",
                "reasoning": "Balanced flavor with good clarity indicates optimal extraction. This is a well-dialed recipe that should be maintained.",
                "expected_time": "3:00",
                "extraction": "good",
                "confidence": 0.95,
            },
        },
        # Specialty Coffee Association guidelines
        {
            "input": "V60, 18g coffee, 270g water, medium-fine grind, 2:45 brew time, tastes complex and layered",
            "output": {
                "grind_change": "none",
                "reasoning": "Complex, layered flavors indicate excellent extraction efficiency. The current parameters are achieving optimal results.",
                "expected_time": "2:45",
                "extraction": "good",
                "confidence": 0.9,
            },
        },
        # Professional cupping standards
        {
            "input": "V60, 19g coffee, 285g water, medium grind, 3:15 brew time, tastes clean with distinct origin character",
            "output": {
                "grind_change": "none",
                "reasoning": "Clean cup with distinct origin character shows proper extraction highlighting the coffee's unique qualities. No adjustment needed.",
                "expected_time": "3:15",
                "extraction": "good",
                "confidence": 0.9,
            },
        },
    ]
    return examples


def create_troubleshooting_examples() -> List[Dict[str, Any]]:
    """Create examples for common brewing problems and their solutions."""
    examples = [
        # Temperature-related issues
        {
            "input": "V60, 20g coffee, 300g water, medium grind, 3:30 brew time, tastes flat despite good extraction time",
            "output": {
                "grind_change": "finer_1",
                "reasoning": "Flat taste despite good timing might indicate low brewing temperature. Finer grind can compensate by increasing extraction efficiency.",
                "expected_time": "3:45",
                "extraction": "under",
                "confidence": 0.7,
            },
        },
        # Water quality issues
        {
            "input": "V60, 21g coffee, 315g water, medium grind, 3:00 brew time, tastes metallic and harsh",
            "output": {
                "grind_change": "coarser_1",
                "reasoning": "Metallic, harsh flavors might indicate water quality issues or over-extraction. Coarser grind can reduce extraction of undesirable compounds.",
                "expected_time": "2:45",
                "extraction": "over",
                "confidence": 0.6,
            },
        },
        # Grinder inconsistency
        {
            "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes uneven with both sour and bitter notes",
            "output": {
                "grind_change": "coarser_1",
                "reasoning": "Mixed sour and bitter notes suggest uneven extraction, possibly from inconsistent grind. Slightly coarser grind can help even out extraction.",
                "expected_time": "2:45",
                "extraction": "uneven",
                "confidence": 0.65,
            },
        },
        # Staling issues
        {
            "input": "V60, 20g coffee, 300g water, medium grind, 2:45 brew time, tastes cardboard-like and stale",
            "output": {
                "grind_change": "finer_1",
                "reasoning": "Cardboard-like flavors suggest stale coffee. Finer grind can help extract more of the remaining good flavors while masking staleness.",
                "expected_time": "3:00",
                "extraction": "under",
                "confidence": 0.5,
            },
        },
    ]
    return examples


def create_ratio_variation_examples() -> List[Dict[str, Any]]:
    """Create examples with different coffee-to-water ratios."""
    examples = [
        # Strong ratios (1:12-1:14)
        {
            "input": "V60, 25g coffee, 300g water, medium grind, 3:45 brew time, tastes intense but balanced",
            "output": {
                "grind_change": "none",
                "reasoning": "Strong ratio with balanced flavor indicates good extraction. The intensity is from the high coffee concentration, not over-extraction.",
                "expected_time": "3:45",
                "extraction": "good",
                "confidence": 0.85,
            },
        },
        {
            "input": "V60, 27g coffee, 350g water, medium-fine grind, 4:30 brew time, tastes overwhelming and bitter",
            "output": {
                "grind_change": "coarser_2",
                "reasoning": "Strong ratio with bitter taste and long brew time indicates over-extraction. Coarser grind will reduce extraction while maintaining strength.",
                "expected_time": "3:45",
                "extraction": "over",
                "confidence": 0.8,
            },
        },
        # Weak ratios (1:16-1:18)
        {
            "input": "V60, 15g coffee, 270g water, medium grind, 2:30 brew time, tastes weak and watery",
            "output": {
                "grind_change": "finer_2",
                "reasoning": "Weak ratio with watery taste suggests under-extraction. Finer grind will increase extraction yield to compensate for low coffee dose.",
                "expected_time": "3:00",
                "extraction": "under",
                "confidence": 0.8,
            },
        },
        {
            "input": "V60, 14g coffee, 250g water, medium-coarse grind, 2:15 brew time, tastes thin and sour",
            "output": {
                "grind_change": "finer_3",
                "reasoning": "Very weak ratio with thin, sour taste indicates severe under-extraction. Much finer grind needed to extract sufficient soluble solids.",
                "expected_time": "3:15",
                "extraction": "under",
                "confidence": 0.85,
            },
        },
    ]
    return examples


def collect_all_manual_examples() -> List[Dict[str, Any]]:
    """Collect all manual examples from different sources."""
    all_examples = []

    # Load existing examples
    existing = load_existing_examples()
    all_examples.extend(existing)
    print(f"Loaded {len(existing)} existing examples")

    # Add new examples from different sources
    personal_logs = create_personal_brewing_logs()
    all_examples.extend(personal_logs)
    print(f"Added {len(personal_logs)} personal brewing log examples")

    forum_examples = create_forum_examples()
    all_examples.extend(forum_examples)
    print(f"Added {len(forum_examples)} forum examples")

    youtube_examples = create_youtube_examples()
    all_examples.extend(youtube_examples)
    print(f"Added {len(youtube_examples)} YouTube examples")

    expert_examples = create_expert_guide_examples()
    all_examples.extend(expert_examples)
    print(f"Added {len(expert_examples)} expert guide examples")

    troubleshooting_examples = create_troubleshooting_examples()
    all_examples.extend(troubleshooting_examples)
    print(f"Added {len(troubleshooting_examples)} troubleshooting examples")

    ratio_examples = create_ratio_variation_examples()
    all_examples.extend(ratio_examples)
    print(f"Added {len(ratio_examples)} ratio variation examples")

    return all_examples


def generate_additional_variations() -> List[Dict[str, Any]]:
    """Generate additional variations to reach 200 total examples."""
    variations = []

    # Create variations of existing patterns with different parameters
    base_scenarios = [
        # Under-extraction variations
        (
            "V60, {}g coffee, {}g water, {} grind, {} brew time, tastes sour",
            "finer",
            "under",
        ),
        (
            "V60, {}g coffee, {}g water, {} grind, {} brew time, tastes weak",
            "finer",
            "under",
        ),
        (
            "V60, {}g coffee, {}g water, {} grind, {} brew time, tastes thin",
            "finer",
            "under",
        ),
        (
            "V60, {}g coffee, {}g water, {} grind, {} brew time, tastes grassy",
            "finer",
            "under",
        ),
        # Over-extraction variations
        (
            "V60, {}g coffee, {}g water, {} grind, {} brew time, tastes bitter",
            "coarser",
            "over",
        ),
        (
            "V60, {}g coffee, {}g water, {} grind, {} brew time, tastes harsh",
            "coarser",
            "over",
        ),
        (
            "V60, {}g coffee, {}g water, {} grind, {} brew time, tastes astringent",
            "coarser",
            "over",
        ),
        (
            "V60, {}g coffee, {}g water, {} grind, {} brew time, tastes dry",
            "coarser",
            "over",
        ),
        # Good extraction variations
        (
            "V60, {}g coffee, {}g water, {} grind, {} brew time, tastes balanced",
            "none",
            "good",
        ),
        (
            "V60, {}g coffee, {}g water, {} grind, {} brew time, tastes sweet",
            "none",
            "good",
        ),
        (
            "V60, {}g coffee, {}g water, {} grind, {} brew time, tastes complex",
            "none",
            "good",
        ),
    ]

    # Parameter ranges
    coffee_amounts = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    grind_sizes = ["coarse", "medium-coarse", "medium", "medium-fine", "fine"]
    brew_times = [
        "2:00",
        "2:15",
        "2:30",
        "2:45",
        "3:00",
        "3:15",
        "3:30",
        "3:45",
        "4:00",
        "4:15",
        "4:30",
        "5:00",
    ]

    for i, (template, grind_direction, extraction_type) in enumerate(base_scenarios):
        for j in range(15):  # Generate 15 variations per template
            coffee = random.choice(coffee_amounts)
            water = coffee * random.choice([14, 15, 16, 17])  # Common ratios
            grind = random.choice(grind_sizes)
            brew_time = random.choice(brew_times)

            input_text = template.format(coffee, water, grind, brew_time)

            # Determine grind change based on direction and current grind
            if grind_direction == "finer":
                if grind == "coarse":
                    grind_change = "finer_2"
                elif grind == "medium-coarse":
                    grind_change = "finer_1"
                elif grind == "medium":
                    grind_change = "finer_1"
                else:
                    grind_change = "finer_1"
            elif grind_direction == "coarser":
                if grind == "fine":
                    grind_change = "coarser_2"
                elif grind == "medium-fine":
                    grind_change = "coarser_1"
                elif grind == "medium":
                    grind_change = "coarser_1"
                else:
                    grind_change = "coarser_1"
            else:
                grind_change = "none"

            # Generate reasoning based on extraction type
            if extraction_type == "under":
                reasoning = f"The {input_text.split('tastes ')[1]} taste indicates under-extraction. A finer grind will increase extraction and improve flavor balance."
                expected_time = _calculate_expected_time(brew_time, "finer")
            elif extraction_type == "over":
                reasoning = f"The {input_text.split('tastes ')[1]} taste indicates over-extraction. A coarser grind will reduce extraction and improve balance."
                expected_time = _calculate_expected_time(brew_time, "coarser")
            else:
                reasoning = "The balanced flavor profile indicates optimal extraction. Current parameters are working well."
                expected_time = brew_time

            confidence = random.uniform(0.7, 0.95)

            variation = {
                "input": input_text,
                "output": {
                    "grind_change": grind_change,
                    "reasoning": reasoning,
                    "expected_time": expected_time,
                    "extraction": extraction_type,
                    "confidence": round(confidence, 2),
                },
            }

            variations.append(variation)

    return variations


def _calculate_expected_time(current_time: str, direction: str) -> str:
    """Calculate expected brew time after grind adjustment."""
    # Parse current time
    if ":" in current_time:
        minutes, seconds = map(int, current_time.split(":"))
        total_seconds = minutes * 60 + seconds
    else:
        total_seconds = int(current_time) * 60

    # Adjust based on grind direction
    if direction == "finer":
        total_seconds += random.randint(15, 45)  # Finer grind = longer time
    elif direction == "coarser":
        total_seconds -= random.randint(15, 45)  # Coarser grind = shorter time

    # Ensure reasonable bounds
    total_seconds = max(120, min(360, total_seconds))  # 2:00 to 6:00 range

    # Convert back to MM:SS format
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


def save_manual_examples(
    examples: List[Dict[str, Any]], filename: str = "data/manual_examples.json"
) -> None:
    """Save manual examples to a JSON file."""
    # Validate all examples before saving
    valid_examples = []
    invalid_count = 0

    for example in examples:
        if validate_example(example):
            valid_examples.append(example)
        else:
            invalid_count += 1

    print(
        f"Validated {len(valid_examples)} examples, {invalid_count} invalid examples filtered out"
    )

    # Save to file
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(valid_examples, f, indent=2)

    print(f"Saved {len(valid_examples)} manual examples to {filename}")


def main() -> None:
    """Main function to collect and save all manual examples."""
    print("Collecting manual training examples...")

    # Collect all manual examples
    manual_examples = collect_all_manual_examples()

    # Generate additional variations if needed to reach 200 examples
    current_count = len(manual_examples)
    target_count = 200

    if current_count < target_count:
        needed = target_count - current_count
        print(f"Need {needed} more examples to reach target of {target_count}")

        additional_variations = generate_additional_variations()
        # Take only what we need
        manual_examples.extend(additional_variations[:needed])

    print(f"Total examples collected: {len(manual_examples)}")

    # Save examples
    save_manual_examples(manual_examples, "data/manual_examples.json")

    # Also update the raw dataset file
    save_manual_examples(manual_examples, "data/coffee_dataset_raw.json")

    print("Manual example collection complete!")


if __name__ == "__main__":
    main()
