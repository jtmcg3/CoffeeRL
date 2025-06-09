"""Tests for the ModelComparator module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset

from src.model_comparator import ModelComparator, load_evaluation_dataset


class TestModelComparatorStructure:
    """Test the basic structure and initialization of ModelComparator."""

    def test_init_default_components(self):
        """Test initialization with default components."""
        comparator = ModelComparator()
        assert comparator.reward_calculator is not None
        assert comparator.hf_manager is not None

    def test_init_custom_components(self):
        """Test initialization with custom components."""
        mock_reward_calc = MagicMock()
        mock_hf_manager = MagicMock()

        comparator = ModelComparator(
            reward_calculator=mock_reward_calc, hf_manager=mock_hf_manager
        )

        assert comparator.reward_calculator is mock_reward_calc
        assert comparator.hf_manager is mock_hf_manager

    def test_methods_exist(self):
        """Test that all required methods exist."""
        comparator = ModelComparator()

        # Check that all expected methods exist
        assert hasattr(comparator, "load_model_for_comparison")
        assert hasattr(comparator, "evaluate_model_on_dataset")
        assert hasattr(comparator, "compare_models")
        assert hasattr(comparator, "print_comparison_report")
        assert hasattr(comparator, "_extract_coffee_data_from_sample")
        assert hasattr(comparator, "_parse_coffee_data_from_text")
        assert hasattr(comparator, "_generate_model_response")


class TestModelLoading:
    """Test model loading functionality."""

    @patch("src.model_comparator.AutoTokenizer")
    @patch("src.model_comparator.AutoModelForCausalLM")
    def test_load_local_model(self, mock_model_class, mock_tokenizer_class):
        """Test loading a model from local path."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        comparator = ModelComparator()
        model, tokenizer = comparator.load_model_for_comparison(
            "test/model/path", is_hf_version=False
        )

        mock_tokenizer_class.from_pretrained.assert_called_once_with("test/model/path")
        assert mock_tokenizer.pad_token == "<eos>"
        assert model is mock_model
        assert tokenizer is mock_tokenizer

    def test_load_hf_model_success(self):
        """Test loading a model from HF Hub successfully."""
        mock_hf_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_hf_manager.load_model_and_tokenizer.return_value = (
            mock_model,
            mock_tokenizer,
        )

        comparator = ModelComparator(hf_manager=mock_hf_manager)
        model, tokenizer = comparator.load_model_for_comparison(
            "batch-1", is_hf_version=True
        )

        mock_hf_manager.load_model_and_tokenizer.assert_called_once_with(
            version_tag="batch-1"
        )
        assert model is mock_model
        assert tokenizer is mock_tokenizer

    def test_load_hf_model_failure(self):
        """Test handling of HF Hub model loading failure."""
        mock_hf_manager = MagicMock()
        mock_hf_manager.load_model_and_tokenizer.return_value = (None, None)

        comparator = ModelComparator(hf_manager=mock_hf_manager)

        with pytest.raises(ValueError, match="Failed to load model version"):
            comparator.load_model_for_comparison("invalid-version", is_hf_version=True)


class TestDataExtraction:
    """Test coffee data extraction from samples."""

    def test_extract_from_text_sample(self):
        """Test extracting coffee data from text-based sample."""
        comparator = ModelComparator()

        sample = {
            "text": "Bean Type: Ethiopian Yirgacheffe\nGrind Size: Fine\nWater Temperature: 95°C"
        }

        coffee_data = comparator._extract_coffee_data_from_sample(sample)
        assert coffee_data["bean_type"] == "Ethiopian Yirgacheffe"
        assert coffee_data["grind_size"] == "Fine"
        assert coffee_data["water_temp"] == 95

    def test_extract_from_dict_sample(self):
        """Test extracting coffee data from dictionary sample."""
        comparator = ModelComparator()

        sample = {
            "bean_type": "Colombian",
            "grind_size": "Medium",
            "water_temp": 93,
            "actual_time": 240,
            "user_rating": 4,
        }

        coffee_data = comparator._extract_coffee_data_from_sample(sample)
        assert coffee_data["bean_type"] == "Colombian"
        assert coffee_data["grind_size"] == "Medium"
        assert coffee_data["water_temp"] == 93
        assert coffee_data["actual_time"] == 240
        assert coffee_data["user_rating"] == 4

    def test_parse_coffee_data_from_text(self):
        """Test parsing coffee data from text format."""
        comparator = ModelComparator()

        text = """
        Bean: Ethiopian Yirgacheffe
        Grind: Fine grind size
        Temperature: 95°C water
        Method: V60 pour over
        """

        coffee_data = comparator._parse_coffee_data_from_text(text)

        assert coffee_data["bean_type"] == "Ethiopian Yirgacheffe"
        assert coffee_data["grind_size"] == "Fine"
        assert coffee_data["water_temp"] == 95

    def test_parse_coffee_data_defaults(self):
        """Test that default values are used when data is not found."""
        comparator = ModelComparator()

        text = "Some random text without coffee parameters"

        coffee_data = comparator._parse_coffee_data_from_text(text)

        # Check defaults
        assert coffee_data["bean_type"] == "Unknown"
        assert coffee_data["grind_size"] == "Medium"
        assert coffee_data["water_temp"] == 93
        assert coffee_data["brew_method"] == "Pour Over"


class TestModelEvaluation:
    """Test model evaluation functionality."""

    @patch("src.model_comparator.RLTrainingLoop")
    def test_evaluate_model_basic(self, mock_rl_loop_class):
        """Test basic model evaluation functionality."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_rl_loop = MagicMock()
        mock_rl_loop_class.return_value = mock_rl_loop

        mock_rl_loop.format_coffee_prompt.return_value = "test prompt"
        mock_rl_loop.parse_model_response.return_value = {
            "yield_percentage": 20.0,
            "predicted_time": 120.0,
        }
        mock_rl_loop.calculate_reward_for_episode.return_value = 0.8

        dataset = Dataset.from_list(
            [
                {"text": "Bean: Ethiopian", "actual_time": 120, "user_rating": 4},
                {"text": "Bean: Colombian", "actual_time": 150, "user_rating": 3},
            ]
        )

        comparator = ModelComparator()

        with (
            patch.object(
                comparator, "_generate_model_response", return_value="test response"
            ),
            patch.object(
                comparator, "_extract_coffee_data_from_sample"
            ) as mock_extract,
        ):
            # Mock coffee data extraction to include ground truth data
            mock_extract.side_effect = [
                {"bean_type": "Ethiopian", "actual_time": 120, "user_rating": 4},
                {"bean_type": "Colombian", "actual_time": 150, "user_rating": 3},
            ]

            metrics = comparator.evaluate_model_on_dataset(
                mock_model, mock_tokenizer, dataset, max_samples=10
            )

        assert "average_reward" in metrics
        assert "task_completion_rate" in metrics
        assert metrics["total_samples"] == 2
        assert metrics["successful_predictions"] == 2
        assert metrics["task_completion_rate"] == 1.0
        assert metrics["average_reward"] == 0.8

    def test_evaluate_model_sample_limit(self):
        """Test that sample limiting works correctly."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        dataset = Dataset.from_list([{"text": f"sample {i}"} for i in range(100)])

        comparator = ModelComparator()

        with (
            patch.object(
                comparator, "_generate_model_response", return_value="test response"
            ),
            patch("src.model_comparator.RLTrainingLoop") as mock_rl_loop_class,
        ):
            mock_rl_loop = MagicMock()
            mock_rl_loop_class.return_value = mock_rl_loop
            mock_rl_loop.parse_model_response.return_value = {"yield_percentage": 20.0}

            metrics = comparator.evaluate_model_on_dataset(
                mock_model, mock_tokenizer, dataset, max_samples=10
            )

        assert metrics["total_samples"] == 10

    @patch("torch.no_grad")
    def test_generate_model_response(self, mock_no_grad):
        """Test model response generation."""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock device
        mock_model.parameters.return_value = [MagicMock()]
        next(mock_model.parameters()).device = "cpu"

        # Mock tokenizer
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.return_value = mock_inputs

        # Mock generation
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_outputs

        # Mock decoding
        mock_tokenizer.decode.return_value = "test prompt generated response"

        comparator = ModelComparator()

        response = comparator._generate_model_response(
            mock_model, mock_tokenizer, "test prompt"
        )

        # Verify the response is the generated part only
        assert response == "generated response"

        # Verify model.generate was called with correct parameters
        mock_model.generate.assert_called_once()
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 150
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["do_sample"] is True


class TestModelComparison:
    """Test model comparison functionality."""

    def test_compare_models_basic(self):
        """Test basic model comparison functionality."""
        comparator = ModelComparator()

        mock_metrics1 = {
            "average_reward": 0.7,
            "task_completion_rate": 0.8,
            "total_samples": 50,
            "successful_predictions": 40,
            "reward_samples": 50,
            "reward_std": 0.2,
            "min_reward": 0.0,
            "max_reward": 1.0,
        }

        mock_metrics2 = {
            "average_reward": 0.9,
            "task_completion_rate": 0.9,
            "total_samples": 50,
            "successful_predictions": 45,
            "reward_samples": 50,
            "reward_std": 0.15,
            "min_reward": 0.1,
            "max_reward": 1.0,
        }

        dataset = Dataset.from_list([{"text": "test sample"}])

        with (
            patch.object(comparator, "load_model_for_comparison") as mock_load,
            patch.object(comparator, "evaluate_model_on_dataset") as mock_evaluate,
        ):
            mock_load.side_effect = [
                (MagicMock(), MagicMock()),
                (MagicMock(), MagicMock()),
            ]
            mock_evaluate.side_effect = [mock_metrics1, mock_metrics2]

            comparison = comparator.compare_models(
                "model1_path", "model2_path", dataset
            )

        assert "model1" in comparison
        assert "model2" in comparison
        assert comparison["model1"]["path"] == "model1_path"
        assert comparison["model2"]["path"] == "model2_path"
        assert abs(comparison["comparison"]["reward_improvement"] - 0.2) < 1e-10
        assert comparison["comparison"]["better_model"] == "model2"

    def test_compare_models_hf_flags(self):
        """Test model comparison with HF Hub flags."""
        comparator = ModelComparator()

        dataset = Dataset.from_list([{"text": "test"}])

        with (
            patch.object(comparator, "load_model_for_comparison") as mock_load,
            patch.object(comparator, "evaluate_model_on_dataset", return_value={}),
        ):
            # Mock load to return proper tuples
            mock_load.return_value = (MagicMock(), MagicMock())

            comparator.compare_models(
                "batch-1",
                "batch-2",
                dataset,
                model1_is_hf=True,
                model2_is_hf=True,
            )

            # Verify load calls with correct HF flags
            assert mock_load.call_count == 2
            mock_load.assert_any_call("batch-1", True)
            mock_load.assert_any_call("batch-2", True)


class TestComparisonReporting:
    """Test comparison report functionality."""

    def test_print_comparison_report(self, capsys):
        """Test that comparison report prints correctly."""
        comparator = ModelComparator()

        comparison = {
            "model1": {
                "path": "model1_path",
                "metrics": {
                    "average_reward": 0.7,
                    "task_completion_rate": 0.8,
                    "reward_std": 0.2,
                },
            },
            "model2": {
                "path": "model2_path",
                "metrics": {
                    "average_reward": 0.9,
                    "task_completion_rate": 0.9,
                    "reward_std": 0.15,
                },
            },
            "comparison": {
                "reward_improvement": 0.2,
                "completion_rate_improvement": 0.1,
                "better_model": "model2",
            },
        }

        comparator.print_comparison_report(comparison)

        captured = capsys.readouterr()
        output = captured.out

        assert "MODEL PERFORMANCE COMPARISON REPORT" in output
        assert "model1_path" in output
        assert "model2_path" in output
        assert "WINNER: MODEL2" in output


class TestDatasetLoading:
    """Test dataset loading functionality."""

    def test_load_dataset_from_directory(self):
        """Test loading dataset from directory."""
        with (
            patch("src.model_comparator.load_from_disk") as mock_load,
            patch("src.model_comparator.Path") as mock_path,
        ):
            mock_dataset = Dataset.from_list([{"text": "test"}])
            mock_load.return_value = mock_dataset

            # Mock Path.is_dir to return True
            mock_path.return_value.is_dir.return_value = True

            dataset = load_evaluation_dataset("test/dataset/path")

            mock_load.assert_called_once_with("test/dataset/path")
            assert len(dataset) == 1

    def test_load_dataset_from_json(self):
        """Test loading dataset from JSON file."""
        test_data = [{"text": "sample 1"}, {"text": "sample 2"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            dataset = load_evaluation_dataset(temp_path)
            assert len(dataset) == 2
            assert dataset[0]["text"] == "sample 1"
        finally:
            Path(temp_path).unlink()

    def test_load_dataset_error_handling(self):
        """Test error handling for invalid dataset paths."""
        with pytest.raises(Exception):
            load_evaluation_dataset("nonexistent/path")


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_quick_comparison_workflow(self):
        """Test the typical quick comparison workflow."""
        comparator = ModelComparator()

        # Mock all dependencies
        with (
            patch.object(comparator, "load_model_for_comparison") as mock_load,
            patch.object(comparator, "evaluate_model_on_dataset") as mock_evaluate,
        ):
            # Setup mocks
            mock_load.return_value = (MagicMock(), MagicMock())
            mock_evaluate.return_value = {
                "average_reward": 0.8,
                "task_completion_rate": 0.9,
                "total_samples": 10,
                "successful_predictions": 9,
                "reward_samples": 10,
                "reward_std": 0.1,
                "min_reward": 0.5,
                "max_reward": 1.0,
            }

            dataset = Dataset.from_list([{"text": f"sample {i}"} for i in range(10)])

            comparison = comparator.compare_models(
                "model1", "model2", dataset, max_samples=10, temperature=0.7
            )

            # Verify the workflow completed successfully
            assert comparison is not None
            assert "model1" in comparison
            assert "model2" in comparison
            assert comparison["evaluation_config"]["max_samples"] == 10
            assert comparison["evaluation_config"]["temperature"] == 0.7

    def test_error_handling_during_evaluation(self):
        """Test error handling during model evaluation."""
        comparator = ModelComparator()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Create dataset with problematic sample
        dataset = Dataset.from_list([{"text": "normal sample"}, {"invalid": "data"}])

        with (
            patch.object(
                comparator,
                "_generate_model_response",
                side_effect=Exception("Model error"),
            ),
            patch("src.model_comparator.RLTrainingLoop"),
        ):
            # Should not raise exception, but handle errors gracefully
            metrics = comparator.evaluate_model_on_dataset(
                mock_model, mock_tokenizer, dataset
            )

            # Should still return valid metrics structure
            assert "average_reward" in metrics
            assert "task_completion_rate" in metrics
            assert metrics["total_samples"] == 2

        dataset = Dataset.from_list([{"text": "normal sample"}, {"invalid": "data"}])

        with (
            patch.object(
                comparator,
                "_generate_model_response",
                side_effect=Exception("Model error"),
            ),
            patch("src.model_comparator.RLTrainingLoop"),
        ):
            # Should not raise exception, but handle errors gracefully
            metrics = comparator.evaluate_model_on_dataset(
                mock_model, mock_tokenizer, dataset
            )

            # Should still return valid metrics structure
            assert "average_reward" in metrics
            assert "task_completion_rate" in metrics
            assert metrics["total_samples"] == 2
