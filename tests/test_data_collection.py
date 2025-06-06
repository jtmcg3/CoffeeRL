"""Tests for data collection templates and validation functions."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_collection import (  # noqa: E402
    PROMPT_TEMPLATE,
    _determine_grind_change,
    _generate_reasoning,
    _parse_brew_time,
    create_manual_example,
    extract_brewing_parameters,
    get_gpt_generation_prompt,
    validate_batch_examples,
    validate_example,
)


class TestCreateManualExample:
    """Test the manual example creation function."""

    def test_create_basic_example(self) -> None:
        """Test creating a basic manual example."""
        example = create_manual_example(
            coffee_amount=15.0,
            water_amount=250.0,
            grind_size="medium",
            brew_time="2:30",
            taste_notes="balanced",
        )

        assert "input" in example
        assert "output" in example
        assert "V60, 15.0g coffee, 250.0g water" in example["input"]
        assert "medium grind, 2:30 brew time, tastes balanced" in example["input"]
        assert "grind_change" in example["output"]
        assert "reasoning" in example["output"]

    def test_bitter_coffee_example(self) -> None:
        """Test example with bitter taste notes."""
        example = create_manual_example(
            coffee_amount=20.0,
            water_amount=300.0,
            grind_size="fine",
            brew_time="4:00",
            taste_notes="bitter and over-extracted",
        )

        assert example["output"]["grind_change"] == "coarser"
        assert "over-extraction" in example["output"]["reasoning"]

    def test_sour_coffee_example(self) -> None:
        """Test example with sour taste notes."""
        example = create_manual_example(
            coffee_amount=12.0,
            water_amount=200.0,
            grind_size="coarse",
            brew_time="1:30",
            taste_notes="sour and under-extracted",
        )

        assert example["output"]["grind_change"] == "finer"
        assert "under-extraction" in example["output"]["reasoning"]


class TestBrewTimeHelpers:
    """Test brew time parsing and analysis functions."""

    def test_parse_brew_time_minutes_seconds(self) -> None:
        """Test parsing time in MM:SS format."""
        assert _parse_brew_time("2:30") == 150
        assert _parse_brew_time("4:15") == 255
        assert _parse_brew_time("0:45") == 45

    def test_parse_brew_time_seconds_only(self) -> None:
        """Test parsing time as seconds."""
        assert _parse_brew_time("180") == 180
        assert _parse_brew_time("120.5") == 120

    def test_parse_brew_time_invalid(self) -> None:
        """Test parsing invalid time formats."""
        assert _parse_brew_time("invalid") == 180  # Default fallback
        assert _parse_brew_time("") == 180

    def test_determine_grind_change_by_taste(self) -> None:
        """Test grind change determination based on taste."""
        assert _determine_grind_change("medium", "2:30", "bitter") == "coarser"
        assert _determine_grind_change("medium", "2:30", "sour") == "finer"
        assert _determine_grind_change("medium", "2:30", "balanced") == "none"

    def test_determine_grind_change_by_time(self) -> None:
        """Test grind change determination based on brew time."""
        assert _determine_grind_change("medium", "6:00", "balanced") == "coarser"
        assert _determine_grind_change("medium", "1:00", "balanced") == "finer"
        assert _determine_grind_change("medium", "2:30", "balanced") == "none"

    def test_generate_reasoning_taste_based(self) -> None:
        """Test reasoning generation for taste-based adjustments."""
        reasoning = _generate_reasoning("medium", "2:30", "bitter")
        assert "over-extraction" in reasoning.lower()
        assert "coarser" in reasoning.lower()

        reasoning = _generate_reasoning("medium", "2:30", "sour")
        assert "under-extraction" in reasoning.lower()
        assert "finer" in reasoning.lower()

    def test_generate_reasoning_time_based(self) -> None:
        """Test reasoning generation for time-based adjustments."""
        reasoning = _generate_reasoning("medium", "6:00", "balanced")
        assert "too long" in reasoning.lower()
        assert "coarser" in reasoning.lower()

        reasoning = _generate_reasoning("medium", "1:00", "balanced")
        assert "too fast" in reasoning.lower()
        assert "finer" in reasoning.lower()


class TestGPTPromptGeneration:
    """Test GPT prompt template functions."""

    def test_default_prompt(self) -> None:
        """Test default prompt generation."""
        prompt = get_gpt_generation_prompt()
        assert "coffee expert" in prompt
        assert "with an issue" in prompt
        assert "V60, 15g coffee" in prompt

    def test_balanced_scenario_prompt(self) -> None:
        """Test balanced scenario prompt."""
        prompt = get_gpt_generation_prompt("balanced")
        assert "that is well-balanced" in prompt
        assert "with an issue" not in prompt

    def test_random_scenario_prompt(self) -> None:
        """Test random scenario prompt."""
        prompt = get_gpt_generation_prompt("random")
        assert "with random parameters" in prompt
        assert "with an issue" not in prompt

    def test_prompt_template_constant(self) -> None:
        """Test that the prompt template constant is properly formatted."""
        assert "You are a coffee expert" in PROMPT_TEMPLATE
        assert "grind_change" in PROMPT_TEMPLATE
        assert "reasoning" in PROMPT_TEMPLATE


class TestValidateExample:
    """Test example validation functions."""

    def test_valid_example(self) -> None:
        """Test validation of a properly formatted example."""
        valid_example = {
            "input": "V60, 15g coffee, 250g water, medium grind, 2:30 time, tastes balanced",
            "output": {
                "grind_change": "none",
                "reasoning": "Current parameters produce balanced extraction.",
            },
        }
        assert validate_example(valid_example) is True

    def test_missing_input_field(self) -> None:
        """Test validation fails when input is missing."""
        invalid_example = {
            "output": {"grind_change": "none", "reasoning": "Test reasoning"}
        }
        assert validate_example(invalid_example) is False

    def test_missing_output_field(self) -> None:
        """Test validation fails when output is missing."""
        invalid_example = {
            "input": "V60, 15g coffee, 250g water, medium grind, 2:30 time, tastes balanced"
        }
        assert validate_example(invalid_example) is False

    def test_missing_required_input_parameters(self) -> None:
        """Test validation fails when required input parameters are missing."""
        # Missing 'coffee'
        invalid_example = {
            "input": "V60, 250g water, medium grind, 2:30 time, tastes balanced",
            "output": {"grind_change": "none", "reasoning": "Test"},
        }
        assert validate_example(invalid_example) is False

        # Missing 'water'
        invalid_example = {
            "input": "V60, 15g coffee, medium grind, 2:30 time, tastes balanced",
            "output": {"grind_change": "none", "reasoning": "Test"},
        }
        assert validate_example(invalid_example) is False

    def test_missing_required_output_fields(self) -> None:
        """Test validation fails when required output fields are missing."""
        # Missing 'grind_change'
        invalid_example = {
            "input": "V60, 15g coffee, 250g water, medium grind, 2:30 time, tastes balanced",
            "output": {"reasoning": "Test reasoning"},
        }
        assert validate_example(invalid_example) is False

        # Missing 'reasoning'
        invalid_example = {
            "input": "V60, 15g coffee, 250g water, medium grind, 2:30 time, tastes balanced",
            "output": {"grind_change": "none"},
        }
        assert validate_example(invalid_example) is False

    def test_invalid_grind_change_value(self) -> None:
        """Test validation fails with invalid grind_change values."""
        invalid_example = {
            "input": "V60, 15g coffee, 250g water, medium grind, 2:30 time, tastes balanced",
            "output": {"grind_change": "invalid_value", "reasoning": "Test reasoning"},
        }
        assert validate_example(invalid_example) is False

    def test_non_dict_input(self) -> None:
        """Test validation fails with non-dictionary input."""
        assert validate_example("not a dict") is False
        assert validate_example(None) is False
        assert validate_example([]) is False


class TestValidateBatchExamples:
    """Test batch validation functionality."""

    def test_all_valid_examples(self) -> None:
        """Test batch validation with all valid examples."""
        valid_examples = [
            {
                "input": "V60, 15g coffee, 250g water, medium grind, 2:30 time, tastes balanced",
                "output": {"grind_change": "none", "reasoning": "Balanced extraction"},
            },
            {
                "input": "V60, 20g coffee, 300g water, fine grind, 3:00 time, tastes bitter",
                "output": {"grind_change": "coarser", "reasoning": "Over-extracted"},
            },
        ]

        results = validate_batch_examples(valid_examples)
        assert results["total"] == 2
        assert results["valid"] == 2
        assert results["invalid"] == 0
        assert results["invalid_indices"] == []

    def test_mixed_valid_invalid_examples(self) -> None:
        """Test batch validation with mixed valid and invalid examples."""
        mixed_examples = [
            {
                "input": "V60, 15g coffee, 250g water, medium grind, 2:30 time, tastes balanced",
                "output": {"grind_change": "none", "reasoning": "Balanced extraction"},
            },
            {
                "input": "Invalid input without required fields",
                "output": {"grind_change": "none", "reasoning": "Test"},
            },
            {
                "input": "V60, 20g coffee, 300g water, fine grind, 3:00 time, tastes bitter",
                "output": {"invalid_field": "missing required fields"},
            },
        ]

        results = validate_batch_examples(mixed_examples)
        assert results["total"] == 3
        assert results["valid"] == 1
        assert results["invalid"] == 2
        assert results["invalid_indices"] == [1, 2]

    def test_empty_batch(self) -> None:
        """Test batch validation with empty list."""
        results = validate_batch_examples([])
        assert results["total"] == 0
        assert results["valid"] == 0
        assert results["invalid"] == 0
        assert results["invalid_indices"] == []


class TestExtractBrewingParameters:
    """Test parameter extraction from text."""

    def test_extract_all_parameters(self) -> None:
        """Test extracting all parameters from well-formatted text."""
        text = (
            "V60, 15g coffee, 250g water, medium grind, 2:30 brew time, tastes balanced"
        )
        params = extract_brewing_parameters(text)

        assert params["coffee_amount"] == "15"
        assert params["water_amount"] == "250"
        assert params["grind_size"] == "medium"
        assert params["brew_time"] == "2:30"
        assert params["taste_notes"] == "balanced"

    def test_extract_partial_parameters(self) -> None:
        """Test extracting parameters when some are missing."""
        text = "V60 with 20g coffee, fine grind, tastes bitter"
        params = extract_brewing_parameters(text)

        assert params["coffee_amount"] == "20"
        assert params["grind_size"] == "fine"
        assert params["taste_notes"] == "bitter"
        assert "water_amount" not in params
        assert "brew_time" not in params

    def test_extract_decimal_amounts(self) -> None:
        """Test extracting decimal amounts."""
        text = "V60, 15.5g coffee, 250.0g water, coarse grind"
        params = extract_brewing_parameters(text)

        assert params["coffee_amount"] == "15.5"
        assert params["water_amount"] == "250.0"
        assert params["grind_size"] == "coarse"

    def test_extract_different_grind_sizes(self) -> None:
        """Test extracting various grind size descriptions."""
        test_cases = [
            ("extra fine grind", "extra fine"),
            ("fine grind", "fine"),
            ("medium grind", "medium"),
            ("coarse grind", "coarse"),
            ("extra coarse grind", "extra coarse"),
        ]

        for text, expected in test_cases:
            params = extract_brewing_parameters(text)
            assert params["grind_size"] == expected

    def test_extract_time_formats(self) -> None:
        """Test extracting different time formats."""
        test_cases = [
            ("brew time 2:30", "2:30"),
            ("3 min brew", "3 min"),
            ("5 minutes total", "5 minutes"),
        ]

        for text, expected in test_cases:
            params = extract_brewing_parameters(text)
            assert params["brew_time"] == expected

    def test_extract_no_parameters(self) -> None:
        """Test extraction when no parameters are found."""
        text = "This is just random text with no brewing parameters"
        params = extract_brewing_parameters(text)

        assert len(params) == 0


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_create_and_validate_example(self) -> None:
        """Test creating an example and validating it."""
        example = create_manual_example(
            coffee_amount=15.0,
            water_amount=250.0,
            grind_size="medium",
            brew_time="2:30",
            taste_notes="balanced",
        )

        assert validate_example(example) is True

    def test_extract_and_recreate_example(self) -> None:
        """Test extracting parameters and recreating an example."""
        original_text = (
            "V60, 18g coffee, 280g water, fine grind, 3:15 brew time, tastes sour"
        )
        params = extract_brewing_parameters(original_text)

        # Recreate example using extracted parameters
        example = create_manual_example(
            coffee_amount=float(params["coffee_amount"]),
            water_amount=float(params["water_amount"]),
            grind_size=params["grind_size"],
            brew_time=params["brew_time"],
            taste_notes=params["taste_notes"],
        )

        assert validate_example(example) is True
        assert (
            example["output"]["grind_change"] == "finer"
        )  # Sour taste should suggest finer grind
