"""
Tests for manual examples collector module.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.manual_examples_collector import (
    _calculate_expected_time,
    collect_all_manual_examples,
    create_expert_guide_examples,
    create_forum_examples,
    create_personal_brewing_logs,
    create_ratio_variation_examples,
    create_troubleshooting_examples,
    create_youtube_examples,
    generate_additional_variations,
    load_existing_examples,
    save_manual_examples,
)


class TestManualExamplesCollector:
    """Test cases for manual examples collector functions."""

    def test_create_personal_brewing_logs(self) -> None:
        """Test creation of personal brewing log examples."""
        examples = create_personal_brewing_logs()

        assert len(examples) > 0
        assert all(isinstance(ex, dict) for ex in examples)
        assert all("input" in ex and "output" in ex for ex in examples)

        # Check for variety in extraction types
        extraction_types = [ex["output"]["extraction"] for ex in examples]
        assert "under" in extraction_types
        assert "over" in extraction_types
        assert "good" in extraction_types

    def test_create_forum_examples(self) -> None:
        """Test creation of forum-based examples."""
        examples = create_forum_examples()

        assert len(examples) > 0
        assert all(isinstance(ex, dict) for ex in examples)

        # Check for specific forum-style language
        inputs = [ex["input"] for ex in examples]
        assert any("flat and boring" in inp for inp in inputs)
        assert any("grassy and vegetal" in inp for inp in inputs)

    def test_create_youtube_examples(self) -> None:
        """Test creation of YouTube-based examples."""
        examples = create_youtube_examples()

        assert len(examples) > 0
        assert all(isinstance(ex, dict) for ex in examples)

        # Check for YouTube-style descriptions
        inputs = [ex["input"] for ex in examples]
        assert any("bright but thin" in inp for inp in inputs)
        assert any("heavy and muddy" in inp for inp in inputs)

    def test_create_expert_guide_examples(self) -> None:
        """Test creation of expert guide examples."""
        examples = create_expert_guide_examples()

        assert len(examples) > 0
        assert all(isinstance(ex, dict) for ex in examples)

        # Expert examples should mostly be "good" extraction
        extraction_types = [ex["output"]["extraction"] for ex in examples]
        assert all(ext == "good" for ext in extraction_types)

    def test_create_troubleshooting_examples(self) -> None:
        """Test creation of troubleshooting examples."""
        examples = create_troubleshooting_examples()

        assert len(examples) > 0
        assert all(isinstance(ex, dict) for ex in examples)

        # Check for troubleshooting-specific issues
        inputs = [ex["input"] for ex in examples]
        assert any("metallic" in inp for inp in inputs)
        assert any("cardboard" in inp for inp in inputs)

    def test_create_ratio_variation_examples(self) -> None:
        """Test creation of ratio variation examples."""
        examples = create_ratio_variation_examples()

        assert len(examples) > 0
        assert all(isinstance(ex, dict) for ex in examples)

        # Check for different ratios
        inputs = [ex["input"] for ex in examples]
        coffee_amounts = []
        for inp in inputs:
            # Extract coffee amount from input
            parts = inp.split("g coffee")
            if parts:
                coffee_str = parts[0].split(", ")[-1]
                coffee_amounts.append(int(coffee_str))

        # Should have variety in coffee amounts
        assert len(set(coffee_amounts)) > 1

    def test_calculate_expected_time(self) -> None:
        """Test brew time calculation function."""
        # Test finer grind (should increase time)
        result = _calculate_expected_time("3:00", "finer")
        assert ":" in result
        minutes, seconds = map(int, result.split(":"))
        original_seconds = 3 * 60
        new_seconds = minutes * 60 + seconds
        assert new_seconds > original_seconds

        # Test coarser grind (should decrease time)
        result = _calculate_expected_time("3:00", "coarser")
        minutes, seconds = map(int, result.split(":"))
        new_seconds = minutes * 60 + seconds
        assert new_seconds < original_seconds

        # Test bounds
        result = _calculate_expected_time("1:00", "coarser")
        minutes, seconds = map(int, result.split(":"))
        assert minutes * 60 + seconds >= 120  # Should not go below 2:00

    def test_generate_additional_variations(self) -> None:
        """Test generation of additional variations."""
        variations = generate_additional_variations()

        assert len(variations) > 0
        assert all(isinstance(var, dict) for var in variations)
        assert all("input" in var and "output" in var for var in variations)

        # Check for variety in parameters
        coffee_amounts = []
        for var in variations:
            inp = var["input"]
            parts = inp.split("g coffee")
            if parts:
                coffee_str = parts[0].split(", ")[-1]
                coffee_amounts.append(int(coffee_str))

        assert len(set(coffee_amounts)) > 5  # Should have variety

    def test_load_existing_examples_file_not_exists(self) -> None:
        """Test loading examples when file doesn't exist."""
        with patch("src.manual_examples_collector.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = load_existing_examples()
            assert result == []

    def test_load_existing_examples_file_exists(self) -> None:
        """Test loading examples when file exists."""
        test_data = [
            {
                "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
                "output": {
                    "grind_change": "coarser_1",
                    "reasoning": "Test reasoning",
                    "expected_time": "2:45",
                    "extraction": "over",
                    "confidence": 0.8,
                },
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            with patch("src.manual_examples_collector.Path") as mock_path:
                mock_path.return_value.exists.return_value = True
                with patch("builtins.open", create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = (
                        json.dumps(test_data)
                    )
                    with patch("json.load", return_value=test_data):
                        result = load_existing_examples()
                        assert len(result) == 1
                        assert result[0]["input"] == test_data[0]["input"]
        finally:
            Path(temp_path).unlink()

    def test_save_manual_examples(self) -> None:
        """Test saving manual examples to file."""
        test_examples = [
            {
                "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
                "output": {
                    "grind_change": "coarser_1",
                    "reasoning": "Test reasoning",
                    "expected_time": "2:45",
                    "extraction": "over",
                    "confidence": 0.8,
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "test_examples.json"

            # Capture print output
            with patch("builtins.print") as mock_print:
                save_manual_examples(test_examples, str(temp_file))

            # Check file was created and contains correct data
            assert temp_file.exists()
            with open(temp_file) as f:
                saved_data = json.load(f)

            assert len(saved_data) == 1
            assert saved_data[0]["input"] == test_examples[0]["input"]

            # Check print statements were called
            mock_print.assert_called()

    def test_save_manual_examples_with_invalid_data(self) -> None:
        """Test saving examples with some invalid data."""
        test_examples = [
            # Valid example
            {
                "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
                "output": {
                    "grind_change": "coarser_1",
                    "reasoning": "Test reasoning",
                    "expected_time": "2:45",
                    "extraction": "over",
                    "confidence": 0.8,
                },
            },
            # Invalid example (missing required fields)
            {"input": "Invalid input", "output": {"invalid_field": "invalid"}},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "test_examples.json"

            with patch("builtins.print") as mock_print:
                save_manual_examples(test_examples, str(temp_file))

            # Check only valid examples were saved
            with open(temp_file) as f:
                saved_data = json.load(f)

            assert len(saved_data) == 1  # Only the valid example
            assert saved_data[0]["input"] == test_examples[0]["input"]

            # Check that print was called (for validation messages)
            mock_print.assert_called()

    @patch("src.manual_examples_collector.load_existing_examples")
    @patch("builtins.print")
    def test_collect_all_manual_examples(
        self, mock_print: MagicMock, mock_load: MagicMock
    ) -> None:
        """Test collecting all manual examples."""
        # Mock existing examples
        mock_load.return_value = [
            {
                "input": "Existing example",
                "output": {"grind_change": "none", "reasoning": "Existing reasoning"},
            }
        ]

        examples = collect_all_manual_examples()

        # Should have examples from all sources plus existing
        assert len(examples) > 20  # Should have many examples

        # Check that print statements were called for each source
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Loaded" in call for call in print_calls)
        assert any("personal brewing log" in call for call in print_calls)
        assert any("forum examples" in call for call in print_calls)
        assert any("YouTube examples" in call for call in print_calls)

    def test_example_format_consistency(self) -> None:
        """Test that all generated examples follow consistent format."""
        all_sources = [
            create_personal_brewing_logs(),
            create_forum_examples(),
            create_youtube_examples(),
            create_expert_guide_examples(),
            create_troubleshooting_examples(),
            create_ratio_variation_examples(),
        ]

        for source_examples in all_sources:
            for example in source_examples:
                # Check basic structure
                assert "input" in example
                assert "output" in example

                # Check input format
                input_text = example["input"]
                assert "V60" in input_text
                assert "coffee" in input_text
                assert "water" in input_text
                assert "grind" in input_text
                assert "brew time" in input_text
                assert "tastes" in input_text

                # Check output structure
                output = example["output"]
                assert "grind_change" in output
                assert "reasoning" in output

                # Check grind_change format
                grind_change = output["grind_change"]
                valid_changes = [
                    "none",
                    "finer_1",
                    "finer_2",
                    "finer_3",
                    "finer_4",
                    "coarser_1",
                    "coarser_2",
                    "coarser_3",
                    "coarser_4",
                ]
                assert grind_change in valid_changes

                # Check optional fields if present
                if "confidence" in output:
                    assert 0.0 <= output["confidence"] <= 1.0

                if "extraction" in output:
                    valid_extractions = ["under", "good", "over", "uneven"]
                    assert output["extraction"] in valid_extractions
