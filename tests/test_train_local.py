"""Tests for local training script functionality."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from datasets import Dataset

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from train_local import (
    create_local_training_args,
    format_coffee_example,
    load_datasets_with_args,
    parse_arguments,
    setup_environment_variables,
)


class TestArgumentParsing(unittest.TestCase):
    """Test command line argument parsing."""

    def test_default_arguments(self):
        """Test default argument values."""
        with patch("sys.argv", ["train_local.py"]):
            args = parse_arguments()

        self.assertEqual(args.train_data, "data/processed/coffee_training_dataset")
        self.assertEqual(args.eval_data, "data/processed/coffee_validation_dataset")
        self.assertEqual(args.model_name, "Qwen/Qwen2-0.5B")
        self.assertEqual(args.output_dir, "./models/coffee-qwen2-local")
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.learning_rate, 2e-4)
        self.assertEqual(args.max_seq_length, 512)
        self.assertFalse(args.dev_mode)
        self.assertFalse(args.no_eval)
        self.assertFalse(args.force_cpu)

    def test_custom_arguments(self):
        """Test custom argument values."""
        test_args = [
            "train_local.py",
            "--train-data",
            "custom/train",
            "--eval-data",
            "custom/eval",
            "--model-name",
            "custom/model",
            "--output-dir",
            "custom/output",
            "--epochs",
            "5",
            "--batch-size",
            "4",
            "--learning-rate",
            "1e-4",
            "--dev-mode",
            "--no-eval",
            "--force-cpu",
        ]

        with patch("sys.argv", test_args):
            args = parse_arguments()

        self.assertEqual(args.train_data, "custom/train")
        self.assertEqual(args.eval_data, "custom/eval")
        self.assertEqual(args.model_name, "custom/model")
        self.assertEqual(args.output_dir, "custom/output")
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertTrue(args.dev_mode)
        self.assertTrue(args.no_eval)
        self.assertTrue(args.force_cpu)


class TestEnvironmentSetup(unittest.TestCase):
    """Test environment variable setup."""

    def setUp(self):
        """Save original environment variables."""
        self.original_env = dict(os.environ)

    def tearDown(self):
        """Restore original environment variables."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_setup_environment_variables_basic(self):
        """Test basic environment variable setup."""
        # Create mock args
        args = MagicMock()
        args.dev_mode = False
        args.output_dir = "/test/output"

        setup_environment_variables(args)

        self.assertEqual(os.environ["QWEN_MODEL_SIZE"], "0.5B")
        self.assertEqual(os.environ["COFFEE_OUTPUT_DIR"], "/test/output")

    def test_setup_environment_variables_dev_mode(self):
        """Test environment setup with development mode."""
        args = MagicMock()
        args.dev_mode = True
        args.max_train_samples = 50
        args.max_eval_samples = 10
        args.output_dir = "/test/output"

        setup_environment_variables(args)

        self.assertEqual(os.environ["COFFEE_DEV_MODE"], "true")
        self.assertEqual(os.environ["COFFEE_MAX_TRAIN_SAMPLES"], "50")
        self.assertEqual(os.environ["COFFEE_MAX_EVAL_SAMPLES"], "10")


class TestDataFormatting(unittest.TestCase):
    """Test data formatting functions."""

    def test_format_coffee_example(self):
        """Test coffee example formatting."""
        example = {
            "input": "V60, 20g coffee, 300g water, medium grind, 3:30 brew time, tastes bitter",
            "output": {
                "grind_change": "coarser_2",
                "reasoning": "Coffee tastes bitter, indicating over-extraction. Recommend coarser grind.",
                "expected_time": "3:00",
                "extraction": "over",
            },
        }

        formatted = format_coffee_example(example)

        self.assertIn("### Coffee Brewing Input:", formatted)
        self.assertIn("### Recommended Adjustment:", formatted)
        self.assertIn(example["input"], formatted)
        self.assertIn("coarser_2", formatted)
        self.assertIn("over-extraction", formatted)

    def test_format_coffee_example_with_special_characters(self):
        """Test formatting with special characters."""
        example = {
            "input": 'V60, 20g coffee, 300g water, "fine" grind, 4:00 brew time',
            "output": {
                "grind_change": "none",
                "reasoning": "Coffee is well-balanced with good extraction.",
                "expected_time": "4:00",
                "extraction": "good",
            },
        }

        formatted = format_coffee_example(example)

        # Should handle quotes properly in JSON
        self.assertIn('"fine"', formatted)
        self.assertIn("well-balanced", formatted)


class TestTrainingArguments(unittest.TestCase):
    """Test training argument creation."""

    @patch("config.platform_config.get_training_batch_size")
    @patch("config.platform_config.get_gradient_accumulation_steps")
    def test_create_local_training_args_defaults(
        self, mock_grad_accum, mock_batch_size
    ):
        """Test training arguments with default values."""
        mock_batch_size.return_value = 2
        mock_grad_accum.return_value = 8

        args = MagicMock()
        args.output_dir = "/test/output"
        args.epochs = 3
        args.batch_size = None
        args.gradient_accumulation = None
        args.learning_rate = 2e-4
        args.no_eval = False
        args.save_steps = None
        args.logging_steps = 10
        args.force_cpu = False

        with patch("torch.cuda.is_available", return_value=True):
            training_args = create_local_training_args(args)

        self.assertEqual(training_args.output_dir, "/test/output")
        self.assertEqual(training_args.num_train_epochs, 3)
        self.assertEqual(training_args.per_device_train_batch_size, 2)
        self.assertEqual(training_args.gradient_accumulation_steps, 8)
        self.assertEqual(training_args.logging_steps, 10)
        self.assertEqual(training_args.eval_strategy, "epoch")
        self.assertTrue(training_args.fp16)

    @patch("config.platform_config.get_training_batch_size")
    @patch("config.platform_config.get_gradient_accumulation_steps")
    def test_create_local_training_args_custom(self, mock_grad_accum, mock_batch_size):
        """Test training arguments with custom values."""
        mock_batch_size.return_value = 2  # Won't be used due to custom value
        mock_grad_accum.return_value = 8  # Won't be used due to custom value

        args = MagicMock()
        args.output_dir = "/custom/output"
        args.epochs = 5
        args.batch_size = 4
        args.gradient_accumulation = 4
        args.learning_rate = 1e-4
        args.no_eval = True
        args.save_steps = 100
        args.logging_steps = 5
        args.force_cpu = True

        with patch("torch.cuda.is_available", return_value=True):
            training_args = create_local_training_args(args)

        self.assertEqual(training_args.num_train_epochs, 5)
        self.assertEqual(training_args.per_device_train_batch_size, 4)
        self.assertEqual(training_args.gradient_accumulation_steps, 4)
        self.assertEqual(training_args.logging_steps, 5)
        self.assertEqual(training_args.eval_strategy, "no")
        self.assertEqual(training_args.save_strategy, "steps")
        self.assertEqual(training_args.save_steps, 100)
        self.assertFalse(training_args.fp16)  # Force CPU disables fp16


class TestDatasetLoading(unittest.TestCase):
    """Test dataset loading functionality."""

    def setUp(self):
        """Set up test datasets."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock training dataset
        self.train_data = [
            {
                "input": f"V60, 20g coffee, 300g water, medium grind, 3:30 brew time, sample {i}",
                "output": {
                    "grind_change": "none",
                    "reasoning": f"Sample reasoning {i}",
                    "expected_time": "3:30",
                    "extraction": "good",
                },
            }
            for i in range(100)
        ]

        # Create mock evaluation dataset
        self.eval_data = [
            {
                "input": f"V60, 20g coffee, 300g water, fine grind, 4:00 brew time, eval {i}",
                "output": {
                    "grind_change": "coarser_1",
                    "reasoning": f"Eval reasoning {i}",
                    "expected_time": "3:45",
                    "extraction": "over",
                },
            }
            for i in range(30)
        ]

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("train_local.load_from_disk")
    def test_load_datasets_normal_mode(self, mock_load_from_disk):
        """Test loading datasets in normal mode."""
        # Mock the datasets
        mock_train_dataset = Dataset.from_list(self.train_data)
        mock_eval_dataset = Dataset.from_list(self.eval_data)
        mock_load_from_disk.side_effect = [mock_train_dataset, mock_eval_dataset]

        args = MagicMock()
        args.train_data = "train/path"
        args.eval_data = "eval/path"
        args.dev_mode = False

        train_dataset, eval_dataset = load_datasets_with_args(args)

        self.assertEqual(len(train_dataset), 100)
        self.assertEqual(len(eval_dataset), 30)

    @patch("train_local.load_from_disk")
    def test_load_datasets_dev_mode(self, mock_load_from_disk):
        """Test loading datasets in development mode."""
        # Mock the datasets
        mock_train_dataset = Dataset.from_list(self.train_data)
        mock_eval_dataset = Dataset.from_list(self.eval_data)
        mock_load_from_disk.side_effect = [mock_train_dataset, mock_eval_dataset]

        args = MagicMock()
        args.train_data = "train/path"
        args.eval_data = "eval/path"
        args.dev_mode = True
        args.max_train_samples = 50
        args.max_eval_samples = 10

        train_dataset, eval_dataset = load_datasets_with_args(args)

        self.assertEqual(len(train_dataset), 50)
        self.assertEqual(len(eval_dataset), 10)

    @patch("train_local.load_from_disk")
    def test_load_datasets_error_handling(self, mock_load_from_disk):
        """Test error handling in dataset loading."""
        mock_load_from_disk.side_effect = FileNotFoundError("Dataset not found")

        args = MagicMock()
        args.train_data = "nonexistent/path"
        args.eval_data = "nonexistent/path"
        args.dev_mode = False

        with self.assertRaises(FileNotFoundError):
            load_datasets_with_args(args)


class TestIntegration(unittest.TestCase):
    """Integration tests for training components."""

    def test_argument_parsing_integration(self):
        """Test that parsed arguments work with other functions."""
        test_args = [
            "train_local.py",
            "--dev-mode",
            "--epochs",
            "2",
            "--output-dir",
            "/tmp/test",
        ]

        with patch("sys.argv", test_args):
            args = parse_arguments()

        # Test that args work with environment setup
        setup_environment_variables(args)
        self.assertEqual(os.environ["COFFEE_DEV_MODE"], "true")

        # Test that args work with training args creation
        with (
            patch("config.platform_config.get_training_batch_size", return_value=2),
            patch(
                "config.platform_config.get_gradient_accumulation_steps", return_value=4
            ),
            patch("torch.cuda.is_available", return_value=False),
        ):
            training_args = create_local_training_args(args)

        self.assertEqual(training_args.num_train_epochs, 2)
        self.assertEqual(training_args.output_dir, "/tmp/test")


if __name__ == "__main__":
    unittest.main()
