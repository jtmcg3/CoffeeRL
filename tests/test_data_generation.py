"""
Tests for data generation functionality.

This module tests the GPT-3.5 data generation script including
API calls, response parsing, validation, and dataset creation.
"""

import json
import os
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

if TYPE_CHECKING:
    pass

import pytest
from datasets import Dataset

from src.data_generation import DataGenerator, create_training_dataset


class TestDataGenerator:
    """Test the DataGenerator class."""

    def test_init_with_api_key(self) -> None:
        """Test DataGenerator initialization with explicit API key."""
        generator = DataGenerator(api_key="test-key")
        assert generator.client.api_key == "test-key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    def test_init_with_env_key(self) -> None:
        """Test DataGenerator initialization with environment variable."""
        generator = DataGenerator()
        assert generator.client.api_key == "env-key"

    def test_generation_template_format(self) -> None:
        """Test that generation template contains required elements."""
        generator = DataGenerator(api_key="test")
        template = generator.generation_template

        assert "V60" in template
        assert "JSON" in template
        assert "grind_change" in template
        assert "reasoning" in template
        assert "confidence" in template

    def test_parse_json_response(self) -> None:
        """Test parsing valid JSON response."""
        generator = DataGenerator(api_key="test")

        response = """```json
        {
          "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
          "output": {
            "grind_change": "coarser_2",
            "reasoning": "Bitter taste indicates over-extraction",
            "expected_time": "2:45",
            "extraction": "over",
            "confidence": 0.8
          }
        }
        ```"""

        result = generator._parse_response(response)

        assert result is not None
        assert result["input"].startswith("V60")
        assert result["output"]["grind_change"] == "coarser_2"
        assert result["output"]["confidence"] == 0.8

    def test_parse_plain_json_response(self) -> None:
        """Test parsing JSON response without code blocks."""
        generator = DataGenerator(api_key="test")

        response = """{
          "input": "V60, 18g coffee, 280g water, fine grind, 4:00 brew time, tastes sour",
          "output": {
            "grind_change": "finer_1",
            "reasoning": "Sour taste indicates under-extraction",
            "expected_time": "3:30",
            "extraction": "under",
            "confidence": 0.9
          }
        }"""

        result = generator._parse_response(response)

        assert result is not None
        assert result["output"]["grind_change"] == "finer_1"
        assert result["output"]["extraction"] == "under"

    def test_parse_text_response_fallback(self) -> None:
        """Test fallback parsing for non-JSON responses."""
        generator = DataGenerator(api_key="test")

        response = """Input: "V60, 22g coffee, 350g water, coarse grind, 2:00 brew time, tastes weak"
        Output: {"grind_change": "finer_3", "reasoning": "Weak taste suggests under-extraction", "confidence": 0.7}"""

        result = generator._parse_text_response(response)

        assert result is not None
        assert "V60" in result["input"]
        assert result["output"]["grind_change"] == "finer_3"

    def test_parse_invalid_response(self) -> None:
        """Test parsing invalid response returns None."""
        generator = DataGenerator(api_key="test")

        invalid_responses = [
            "This is not JSON at all",
            "```json\n{invalid json}\n```",
            "Input: incomplete\nOutput: also incomplete",
        ]

        for response in invalid_responses:
            result = generator._parse_response(response)
            # Should either be None or fail validation
            if result is not None:
                from src.data_collection import validate_example

                assert not validate_example(result)

    @patch("src.data_generation.DataGenerator.generate_single_example")
    def test_generate_batch_success(self, mock_generate: Mock) -> None:
        """Test successful batch generation."""
        generator = DataGenerator(api_key="test")

        # Mock successful example generation
        mock_example = {
            "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
            "output": {
                "grind_change": "coarser_2",
                "reasoning": "Bitter taste indicates over-extraction",
                "confidence": 0.8,
            },
        }
        mock_generate.return_value = mock_example

        examples = generator.generate_batch(5)

        assert len(examples) == 5
        assert all(ex == mock_example for ex in examples)
        assert mock_generate.call_count == 5

    @patch("src.data_generation.DataGenerator.generate_single_example")
    def test_generate_batch_with_failures(self, mock_generate: Mock) -> None:
        """Test batch generation with some failures."""
        generator = DataGenerator(api_key="test")

        # Mock alternating success/failure
        mock_example = {
            "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
            "output": {
                "grind_change": "coarser_2",
                "reasoning": "Bitter taste indicates over-extraction",
                "confidence": 0.8,
            },
        }
        mock_generate.side_effect = [
            mock_example,
            None,
            mock_example,
            None,
            mock_example,
        ]

        examples = generator.generate_batch(5)

        assert len(examples) == 3  # Only successful ones
        assert all(ex == mock_example for ex in examples)

    @patch("src.data_generation.DataGenerator.generate_single_example")
    def test_generate_batch_too_many_failures(self, mock_generate: Mock) -> None:
        """Test batch generation stops with too many failures."""
        generator = DataGenerator(api_key="test")

        # Mock all failures
        mock_generate.return_value = None

        examples = generator.generate_batch(10)

        assert len(examples) == 0
        # Should stop early due to too many failures
        assert mock_generate.call_count <= 10

    def test_save_dataset(self) -> None:
        """Test saving dataset to JSON and HuggingFace format."""
        generator = DataGenerator(api_key="test")

        examples = [
            {
                "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
                "output": {
                    "grind_change": "coarser_2",
                    "reasoning": "Bitter taste indicates over-extraction",
                    "confidence": 0.8,
                },
            },
            {
                "input": "V60, 18g coffee, 280g water, fine grind, 4:00 brew time, tastes sour",
                "output": {
                    "grind_change": "finer_1",
                    "reasoning": "Sour taste indicates under-extraction",
                    "confidence": 0.9,
                },
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = generator.save_dataset(examples, temp_dir)

            # Check JSON file was created
            json_path = os.path.join(temp_dir, "coffee_dataset_raw.json")
            assert os.path.exists(json_path)

            with open(json_path) as f:
                saved_data = json.load(f)
            assert saved_data == examples

            # Check HuggingFace dataset was created
            dataset_path = os.path.join(temp_dir, "coffee_dataset")
            assert os.path.exists(dataset_path)
            assert isinstance(dataset, Dataset)
            assert len(dataset) == 2


class TestAPIIntegration:
    """Test API integration with mocked OpenAI calls."""

    @patch("src.data_generation.OpenAI")
    def test_generate_single_example_success(self, mock_openai_class: Mock) -> None:
        """Test successful API call and parsing."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = """{
          "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
          "output": {
            "grind_change": "coarser_2",
            "reasoning": "Bitter taste indicates over-extraction",
            "confidence": 0.8
          }
        }"""

        mock_client.chat.completions.create.return_value = mock_response

        generator = DataGenerator(api_key="test")
        result = generator.generate_single_example()

        assert result is not None
        assert "V60" in result["input"]
        assert result["output"]["grind_change"] == "coarser_2"

        # Verify API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-3.5-turbo"
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["max_tokens"] == 200

    @patch("src.data_generation.OpenAI")
    def test_generate_single_example_api_failure(self, mock_openai_class: Mock) -> None:
        """Test handling of API failures with retries."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API failure
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        generator = DataGenerator(api_key="test")
        result = generator.generate_single_example(max_retries=2, delay=0.1)

        assert result is None
        assert mock_client.chat.completions.create.call_count == 2

    @patch("src.data_generation.OpenAI")
    def test_generate_single_example_empty_response(
        self, mock_openai_class: Mock
    ) -> None:
        """Test handling of empty API responses."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None

        mock_client.chat.completions.create.return_value = mock_response

        generator = DataGenerator(api_key="test")
        result = generator.generate_single_example()

        assert result is None


class TestEndToEnd:
    """Test end-to-end functionality."""

    @patch("src.data_generation.DataGenerator")
    def test_create_training_dataset(self, mock_generator_class: Mock) -> None:
        """Test complete dataset creation workflow."""
        mock_examples = [
            {
                "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
                "output": {
                    "grind_change": "coarser_2",
                    "reasoning": "Bitter taste indicates over-extraction",
                    "confidence": 0.8,
                },
            }
        ]

        # Mock the generator instance and its methods
        mock_generator = Mock()
        mock_generator.generate_batch.return_value = mock_examples
        mock_generator.save_dataset.return_value = Dataset.from_dict(
            {
                "input": [ex["input"] for ex in mock_examples],
                "output": [ex["output"] for ex in mock_examples],
            }
        )
        mock_generator_class.return_value = mock_generator

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = create_training_dataset(n_examples=1, output_dir=temp_dir)

            assert isinstance(dataset, Dataset)
            assert len(dataset) == 1

            # Verify the generator was called correctly
            mock_generator.generate_batch.assert_called_once_with(1)
            mock_generator.save_dataset.assert_called_once()

    @patch("src.data_generation.DataGenerator")
    def test_create_training_dataset_no_examples(
        self, mock_generator_class: Mock
    ) -> None:
        """Test error handling when no examples are generated."""
        mock_generator = Mock()
        mock_generator.generate_batch.return_value = []
        mock_generator_class.return_value = mock_generator

        with pytest.raises(RuntimeError, match="Failed to generate any valid examples"):
            create_training_dataset(n_examples=1)


class TestValidationIntegration:
    """Test integration with validation functions."""

    def test_generated_example_validation(self) -> None:
        """Test that properly formatted examples pass validation."""
        from src.data_collection import validate_example

        valid_example = {
            "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
            "output": {
                "grind_change": "coarser_2",
                "reasoning": "Bitter taste indicates over-extraction",
                "expected_time": "2:45",
                "extraction": "over",
                "confidence": 0.8,
            },
        }

        assert validate_example(valid_example)

    def test_invalid_example_validation(self) -> None:
        """Test that improperly formatted examples fail validation."""
        from src.data_collection import validate_example

        invalid_examples = [
            # Missing required fields
            {
                "input": "V60, 20g coffee, 300g water",
                "output": {"grind_change": "coarser"},
            },
            # Invalid grind_change format
            {
                "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
                "output": {
                    "grind_change": "invalid_direction",
                    "reasoning": "Some reasoning",
                },
            },
            # Invalid confidence value
            {
                "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
                "output": {
                    "grind_change": "coarser_2",
                    "reasoning": "Some reasoning",
                    "confidence": 1.5,  # Invalid: > 1.0
                },
            },
        ]

        for example in invalid_examples:
            assert not validate_example(example)
