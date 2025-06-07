#!/usr/bin/env python3
"""
Tests for format_training_data.py script.

Tests the formatting of training data for QLoRA fine-tuning of Qwen2 models.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# Add scripts directory to path before importing local modules
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

import pytest
from datasets import Dataset
from format_training_data import (
    create_qwen2_dataset,
    format_example_for_qwen2,
    format_training_data_for_qwen2,
    load_training_data,
    save_formatted_dataset,
    validate_token_lengths,
)


class TestLoadTrainingData:
    """Test loading training and validation data."""

    def test_load_training_data_success(self) -> None:
        """Test successful loading of training data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)

            # Create sample data files
            training_data = [
                {
                    "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
                    "output": {
                        "grind_change": "coarser_2",
                        "reasoning": "Bitter taste indicates over-extraction",
                        "expected_time": "3:30",
                        "extraction": "good",
                        "confidence": 0.9,
                    },
                }
            ]

            validation_data = [
                {
                    "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes sour",
                    "output": {
                        "grind_change": "finer_2",
                        "reasoning": "Sour taste indicates under-extraction",
                        "expected_time": "2:45",
                        "extraction": "good",
                        "confidence": 0.8,
                    },
                }
            ]

            # Write test data files
            with open(data_dir / "training_examples.json", "w") as f:
                json.dump(training_data, f)

            with open(data_dir / "validation_examples.json", "w") as f:
                json.dump(validation_data, f)

            # Test loading
            training, validation = load_training_data(data_dir)

            assert len(training) == 1
            assert len(validation) == 1
            assert training[0]["input"] == training_data[0]["input"]
            assert validation[0]["output"]["grind_change"] == "finer_2"

    def test_load_training_data_missing_files(self) -> None:
        """Test handling of missing data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)

            with pytest.raises(FileNotFoundError):
                load_training_data(data_dir)


class TestFormatExampleForQwen2:
    """Test formatting individual examples for Qwen2."""

    def test_format_example_basic(self) -> None:
        """Test basic example formatting."""
        example = {
            "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
            "output": {
                "grind_change": "coarser_2",
                "reasoning": "Bitter taste indicates over-extraction",
                "expected_time": "3:30",
                "extraction": "good",
                "confidence": 0.9,
            },
        }

        formatted = format_example_for_qwen2(example)

        # Check that prompt is included
        assert "Analyze this V60 brew:" in formatted
        assert example["input"] in formatted

        # Check that output is JSON formatted
        assert '"grind_change": "coarser_2"' in formatted
        assert '"confidence": 0.9' in formatted

        # Check structure
        lines = formatted.split("\n")
        assert len(lines) >= 3  # Prompt + blank line + output

    def test_format_example_special_characters(self) -> None:
        """Test formatting with special characters."""
        example = {
            "input": "V60, 20g coffee, 300g water, café grind, 3:00 brew time",
            "output": {
                "grind_change": "coarser_1",
                "reasoning": "Test with café and other unicode: ñ, é, ü",
                "expected_time": "3:15",
                "extraction": "good",
                "confidence": 0.7,
            },
        }

        formatted = format_example_for_qwen2(example)

        # Check unicode handling
        assert "café" in formatted
        assert "ñ, é, ü" in formatted

        # Verify JSON is valid
        output_start = formatted.find('{"grind_change"')
        output_json = formatted[output_start:]
        parsed_output = json.loads(output_json)
        assert parsed_output["reasoning"] == example["output"]["reasoning"]


class TestCreateQwen2Dataset:
    """Test dataset creation for Qwen2."""

    def test_create_dataset_basic(self) -> None:
        """Test basic dataset creation."""
        examples = [
            {
                "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
                "output": {
                    "grind_change": "coarser_2",
                    "reasoning": "Bitter taste indicates over-extraction",
                    "expected_time": "3:30",
                    "extraction": "good",
                    "confidence": 0.9,
                },
            },
            {
                "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes sour",
                "output": {
                    "grind_change": "finer_2",
                    "reasoning": "Sour taste indicates under-extraction",
                    "expected_time": "2:45",
                    "extraction": "good",
                    "confidence": 0.8,
                },
            },
        ]

        dataset = create_qwen2_dataset(examples)

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 2
        assert "text" in list(dataset.column_names)

        # Check first example
        first_text = dataset[0]["text"]
        assert "Analyze this V60 brew:" in first_text
        assert "tastes bitter" in first_text
        assert '"grind_change": "coarser_2"' in first_text

    def test_create_dataset_empty(self) -> None:
        """Test dataset creation with empty input."""
        examples: List[Dict[str, Any]] = []
        dataset = create_qwen2_dataset(examples)

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 0


class TestValidateTokenLengths:
    """Test token length validation."""

    def test_validate_token_lengths_mock_tokenizer(self) -> None:
        """Test token validation with mocked tokenizer."""
        # Create test dataset
        examples = [
            {"text": "Short example"},
            {
                "text": "This is a longer example with more tokens to test the validation"
            },
            {"text": "Medium length example"},
        ]
        dataset = Dataset.from_list(examples)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = [
            [1, 2, 3],  # 3 tokens
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # 15 tokens
            [1, 2, 3, 4, 5, 6, 7, 8],  # 8 tokens
        ]

        with patch(
            "format_training_data.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ):
            stats = validate_token_lengths(dataset, "0.5B")

        assert stats["status"] == "validated"
        assert stats["total_examples"] == 3
        assert stats["max_tokens"] == 15
        assert stats["min_tokens"] == 3
        assert stats["avg_tokens"] == (3 + 15 + 8) / 3
        assert stats["over_limit"] == 0
        assert stats["model_max_length"] == 8192

    def test_validate_token_lengths_fallback(self) -> None:
        """Test token validation fallback when tokenizer fails."""
        examples = [{"text": "Test example"}]
        dataset = Dataset.from_list(examples)

        with patch(
            "format_training_data.AutoTokenizer.from_pretrained",
            side_effect=Exception("Network error"),
        ):
            stats = validate_token_lengths(dataset, "0.5B")

        assert stats["status"] == "estimated"
        assert stats["max_length"] == 8192

    def test_validate_token_lengths_over_limit(self) -> None:
        """Test detection of examples over token limit."""
        examples = [{"text": "Test"}]
        dataset = Dataset.from_list(examples)

        # Mock tokenizer that returns too many tokens
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(
            range(10000)
        )  # 10000 tokens > 8192 limit

        with patch(
            "format_training_data.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ):
            stats = validate_token_lengths(dataset, "0.5B")

        assert stats["over_limit"] == 1
        assert stats["max_tokens"] == 10000


class TestSaveFormattedDataset:
    """Test saving formatted datasets."""

    def test_save_formatted_dataset(self) -> None:
        """Test saving dataset to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Create test dataset
            examples = [{"text": "Example 1"}, {"text": "Example 2"}]
            dataset = Dataset.from_list(examples)

            # Save dataset
            save_formatted_dataset(dataset, output_dir, "training", "0.5B")

            # Check that directory was created
            expected_dir = output_dir / "coffee_training_qwen2_0.5B"
            assert expected_dir.exists()

            # Check that JSON file was created
            json_file = output_dir / "coffee_training_qwen2_0.5B.json"
            assert json_file.exists()

            # Verify JSON content
            with open(json_file, "r") as f:
                saved_data = json.load(f)

            assert len(saved_data) == 2
            assert saved_data[0]["text"] == "Example 1"


class TestFormatTrainingDataForQwen2:
    """Test the main formatting function."""

    def test_format_training_data_integration(self) -> None:
        """Test the complete formatting pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            output_dir = Path(temp_dir) / "output"

            # Create sample data
            training_data = [
                {
                    "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
                    "output": {
                        "grind_change": "coarser_2",
                        "reasoning": "Bitter taste indicates over-extraction",
                        "expected_time": "3:30",
                        "extraction": "good",
                        "confidence": 0.9,
                    },
                }
            ]

            validation_data = [
                {
                    "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes sour",
                    "output": {
                        "grind_change": "finer_2",
                        "reasoning": "Sour taste indicates under-extraction",
                        "expected_time": "2:45",
                        "extraction": "good",
                        "confidence": 0.8,
                    },
                }
            ]

            # Write test data
            with open(data_dir / "training_examples.json", "w") as f:
                json.dump(training_data, f)

            with open(data_dir / "validation_examples.json", "w") as f:
                json.dump(validation_data, f)

            # Mock tokenizer for validation
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

            with patch(
                "format_training_data.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ):
                results = format_training_data_for_qwen2(
                    model_size="0.5B", data_dir=data_dir, output_dir=output_dir
                )

            # Check results
            assert results["model_size"] == "0.5B"
            assert results["training_examples"] == 1
            assert results["validation_examples"] == 1
            assert results["training_stats"]["status"] == "validated"
            assert results["validation_stats"]["status"] == "validated"

            # Check output files exist
            training_dir = output_dir / "coffee_training_qwen2_0.5B"
            validation_dir = output_dir / "coffee_validation_qwen2_0.5B"
            assert training_dir.exists()
            assert validation_dir.exists()

    def test_format_training_data_different_model_sizes(self) -> None:
        """Test formatting for different model sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)

            # Create minimal test data
            training_data = [{"input": "test", "output": {"grind_change": "none"}}]
            validation_data = [{"input": "test", "output": {"grind_change": "none"}}]

            with open(data_dir / "training_examples.json", "w") as f:
                json.dump(training_data, f)

            with open(data_dir / "validation_examples.json", "w") as f:
                json.dump(validation_data, f)

            # Test both model sizes
            for model_size in ["0.5B", "1.5B"]:
                with patch(
                    "format_training_data.AutoTokenizer.from_pretrained"
                ) as mock_tokenizer_class:
                    mock_tokenizer = MagicMock()
                    mock_tokenizer.encode.return_value = [1, 2, 3]
                    mock_tokenizer_class.return_value = mock_tokenizer

                    results = format_training_data_for_qwen2(
                        model_size=model_size, data_dir=data_dir, output_dir=data_dir
                    )

                    assert results["model_size"] == model_size

                    # Check that correct model name was used for tokenizer
                    expected_model_name = f"Qwen/Qwen2-{model_size}"
                    mock_tokenizer_class.assert_called_with(expected_model_name)


if __name__ == "__main__":
    pytest.main([__file__])
