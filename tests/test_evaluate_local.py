"""Tests for local evaluation script functionality."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluate_local import (
    calculate_metrics,
    extract_input_and_expected_output,
    parse_arguments,
    parse_model_output,
)


class TestArgumentParsing(unittest.TestCase):
    """Test command line argument parsing."""

    def test_required_arguments(self):
        """Test that model-path is required."""
        with patch("sys.argv", ["evaluate_local.py"]):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_default_arguments(self):
        """Test default argument values."""
        with patch("sys.argv", ["evaluate_local.py", "--model-path", "/test/model"]):
            args = parse_arguments()
            self.assertEqual(args.model_path, "/test/model")
            self.assertEqual(args.eval_data, "data/processed/coffee_validation_dataset")
            self.assertEqual(args.output_file, "evaluation_results.json")
            self.assertEqual(args.temperature, 0.1)
            self.assertFalse(args.quick_eval)
            self.assertFalse(args.save_predictions)
            self.assertFalse(args.force_cpu)

    def test_custom_arguments(self):
        """Test custom argument values."""
        with patch(
            "sys.argv",
            [
                "evaluate_local.py",
                "--model-path",
                "/custom/model",
                "--eval-data",
                "/custom/data",
                "--output-file",
                "custom_results.json",
                "--max-samples",
                "100",
                "--temperature",
                "0.5",
                "--quick-eval",
                "--save-predictions",
                "--force-cpu",
            ],
        ):
            args = parse_arguments()
            self.assertEqual(args.model_path, "/custom/model")
            self.assertEqual(args.eval_data, "/custom/data")
            self.assertEqual(args.output_file, "custom_results.json")
            self.assertEqual(args.max_samples, 100)
            self.assertEqual(args.temperature, 0.5)
            self.assertTrue(args.quick_eval)
            self.assertTrue(args.save_predictions)
            self.assertTrue(args.force_cpu)


class TestDataProcessing(unittest.TestCase):
    """Test data processing functions."""

    def test_extract_input_and_expected_output_valid(self):
        """Test extracting input and expected output from valid text."""
        text = """Analyze this V60 brew: V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes sour

Provide grind adjustment, expected brew time, extraction assessment, and reasoning:
{"grind_change": "finer_1", "reasoning": "Sour taste indicates under-extraction", "expected_time": "3:30", "extraction": "under", "confidence": 0.8}"""

        input_text, expected_output = extract_input_and_expected_output(text)

        self.assertIn("V60, 20g coffee", input_text)
        self.assertEqual(expected_output["grind_change"], "finer_1")
        self.assertEqual(expected_output["extraction"], "under")
        self.assertEqual(expected_output["expected_time"], "3:30")

    def test_extract_input_and_expected_output_invalid_format(self):
        """Test handling of invalid text format."""
        text = "Invalid text without proper separator"

        with pytest.raises(ValueError, match="Invalid text format"):
            extract_input_and_expected_output(text)

    def test_extract_input_and_expected_output_invalid_json(self):
        """Test handling of invalid JSON in expected output."""
        text = """Input text

Provide grind adjustment, expected brew time, extraction assessment, and reasoning:
{invalid json}"""

        with pytest.raises(ValueError, match="Invalid JSON"):
            extract_input_and_expected_output(text)


class TestModelOutputParsing(unittest.TestCase):
    """Test model output parsing functions."""

    def test_parse_model_output_valid_json(self):
        """Test parsing valid JSON output."""
        output = (
            '{"grind_change": "finer_1", "extraction": "under", "reasoning": "Test"}'
        )
        result = parse_model_output(output)

        self.assertEqual(result["grind_change"], "finer_1")
        self.assertEqual(result["extraction"], "under")
        self.assertEqual(result["reasoning"], "Test")

    def test_parse_model_output_embedded_json(self):
        """Test parsing JSON embedded in text."""
        output = 'Some text before {"grind_change": "none", "extraction": "good"} some text after'
        result = parse_model_output(output)

        self.assertEqual(result["grind_change"], "none")
        self.assertEqual(result["extraction"], "good")

    def test_parse_model_output_regex_fallback(self):
        """Test regex fallback when JSON parsing fails."""
        output = "grind_change: finer_2, extraction: under, some other text"
        result = parse_model_output(output)

        self.assertEqual(result["grind_change"], "finer_2")
        self.assertEqual(result["extraction"], "under")

    def test_parse_model_output_no_matches(self):
        """Test handling when no patterns match."""
        output = "Random text with no relevant information"
        result = parse_model_output(output)

        self.assertEqual(result, {})

    def test_parse_model_output_partial_matches(self):
        """Test handling partial pattern matches."""
        output = "The grind should be finer_1 but no extraction info"
        result = parse_model_output(output)

        self.assertEqual(result["grind_change"], "finer_1")
        self.assertNotIn("extraction", result)


class TestMetricsCalculation(unittest.TestCase):
    """Test metrics calculation functions."""

    def test_calculate_metrics_perfect_accuracy(self):
        """Test metrics calculation with perfect accuracy."""
        predictions = [
            {"grind_change": "finer_1", "extraction": "under"},
            {"grind_change": "none", "extraction": "good"},
            {"grind_change": "coarser_1", "extraction": "over"},
        ]
        expected = [
            {"grind_change": "finer_1", "extraction": "under"},
            {"grind_change": "none", "extraction": "good"},
            {"grind_change": "coarser_1", "extraction": "over"},
        ]

        metrics = calculate_metrics(predictions, expected)

        self.assertEqual(metrics["grind_accuracy"], 1.0)
        self.assertEqual(metrics["extraction_accuracy"], 1.0)
        self.assertEqual(metrics["average_accuracy"], 1.0)

    def test_calculate_metrics_zero_accuracy(self):
        """Test metrics calculation with zero accuracy."""
        predictions = [
            {"grind_change": "finer_1", "extraction": "under"},
            {"grind_change": "none", "extraction": "good"},
        ]
        expected = [
            {"grind_change": "coarser_1", "extraction": "over"},
            {"grind_change": "finer_2", "extraction": "under"},
        ]

        metrics = calculate_metrics(predictions, expected)

        self.assertEqual(metrics["grind_accuracy"], 0.0)
        self.assertEqual(metrics["extraction_accuracy"], 0.0)
        self.assertEqual(metrics["average_accuracy"], 0.0)

    def test_calculate_metrics_partial_accuracy(self):
        """Test metrics calculation with partial accuracy."""
        predictions = [
            {"grind_change": "finer_1", "extraction": "under"},  # Both correct
            {
                "grind_change": "none",
                "extraction": "under",
            },  # Grind correct, extraction wrong
            {
                "grind_change": "finer_1",
                "extraction": "good",
            },  # Grind wrong, extraction correct
            {},  # Empty prediction
        ]
        expected = [
            {"grind_change": "finer_1", "extraction": "under"},
            {"grind_change": "none", "extraction": "good"},
            {"grind_change": "coarser_1", "extraction": "good"},
            {"grind_change": "finer_2", "extraction": "over"},
        ]

        metrics = calculate_metrics(predictions, expected)

        # 2 out of 4 grind predictions correct (50%)
        self.assertEqual(metrics["grind_accuracy"], 0.5)
        # 2 out of 4 extraction predictions correct (50%)
        self.assertEqual(metrics["extraction_accuracy"], 0.5)
        # Average of 50% and 50%
        self.assertEqual(metrics["average_accuracy"], 0.5)

    def test_calculate_metrics_missing_fields(self):
        """Test metrics calculation with missing fields."""
        predictions = [
            {"grind_change": "finer_1"},  # Missing extraction
            {"extraction": "good"},  # Missing grind_change
            {},  # Missing both
        ]
        expected = [
            {"grind_change": "finer_1", "extraction": "under"},
            {"grind_change": "none", "extraction": "good"},
            {"grind_change": "coarser_1", "extraction": "over"},
        ]

        metrics = calculate_metrics(predictions, expected)

        # Only 1 out of 3 grind predictions available and correct
        self.assertEqual(metrics["grind_accuracy"], 1 / 3)
        # Only 1 out of 3 extraction predictions available and correct
        self.assertEqual(metrics["extraction_accuracy"], 1 / 3)
        # Average accuracy
        self.assertAlmostEqual(metrics["average_accuracy"], 1 / 3)

    def test_calculate_metrics_empty_lists(self):
        """Test metrics calculation with empty lists."""
        metrics = calculate_metrics([], [])

        self.assertEqual(metrics["grind_accuracy"], 0.0)
        self.assertEqual(metrics["extraction_accuracy"], 0.0)
        self.assertEqual(metrics["average_accuracy"], 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for evaluation workflow."""

    def test_evaluation_data_format_compatibility(self):
        """Test that the evaluation script can handle the expected data format."""
        # Create sample data in the expected format
        sample_data = [
            {
                "text": """Analyze this V60 brew: V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes sour

Provide grind adjustment, expected brew time, extraction assessment, and reasoning:
{"grind_change": "finer_1", "reasoning": "Sour taste indicates under-extraction", "expected_time": "3:30", "extraction": "under", "confidence": 0.8}"""
            }
        ]

        # Test that we can extract input and expected output
        input_text, expected_output = extract_input_and_expected_output(
            sample_data[0]["text"]
        )

        self.assertIn("V60, 20g coffee", input_text)
        self.assertEqual(expected_output["grind_change"], "finer_1")
        self.assertEqual(expected_output["extraction"], "under")

    def test_end_to_end_evaluation_workflow(self):
        """Test the complete evaluation workflow with mock data."""
        # Sample predictions and expected outputs
        predictions = [
            {"grind_change": "finer_1", "extraction": "under"},
            {"grind_change": "none", "extraction": "good"},
        ]
        expected = [
            {"grind_change": "finer_1", "extraction": "under"},
            {"grind_change": "none", "extraction": "good"},
        ]

        # Calculate metrics
        metrics = calculate_metrics(predictions, expected)

        # Verify results
        self.assertEqual(metrics["grind_accuracy"], 1.0)
        self.assertEqual(metrics["extraction_accuracy"], 1.0)
        self.assertEqual(metrics["average_accuracy"], 1.0)

        # Test that results can be serialized to JSON
        results = {
            "evaluation_info": {
                "model_path": "/test/model",
                "num_samples": 2,
                "evaluation_time_seconds": 10.5,
            },
            "metrics": metrics,
        }

        # Should not raise an exception
        json_str = json.dumps(results, indent=2)
        self.assertIn("grind_accuracy", json_str)


if __name__ == "__main__":
    unittest.main()
