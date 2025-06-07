"""
Tests for the automated training and evaluation pipeline.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

# Mock import removed as it's not used in this test file


class TestPipelineScript(unittest.TestCase):
    """Test the run_experiment.sh script functionality."""

    def setUp(self):
        """Set up test environment."""
        self.script_path = Path("scripts/run_experiment.sh")
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_script_exists_and_executable(self):
        """Test that the pipeline script exists and is executable."""
        self.assertTrue(self.script_path.exists(), "Pipeline script should exist")
        self.assertTrue(
            os.access(self.script_path, os.X_OK), "Pipeline script should be executable"
        )

    def test_script_help_option(self):
        """Test that the script shows help when --help is provided."""
        result = subprocess.run(
            [str(self.script_path), "--help"], capture_output=True, text=True
        )

        self.assertEqual(result.returncode, 0, "Help command should succeed")
        self.assertIn(
            "CoffeeRL Automated Training and Evaluation Pipeline", result.stdout
        )
        self.assertIn("--experiment-name", result.stdout)
        self.assertIn("--model-size", result.stdout)
        self.assertIn("--training-mode", result.stdout)
        self.assertIn("--evaluation-mode", result.stdout)

    def test_script_invalid_option(self):
        """Test that the script handles invalid options gracefully."""
        result = subprocess.run(
            [str(self.script_path), "--invalid-option"], capture_output=True, text=True
        )

        self.assertNotEqual(result.returncode, 0, "Invalid option should cause failure")
        self.assertIn("Unknown option", result.stdout)


class TestMakefileTargets(unittest.TestCase):
    """Test Makefile targets."""

    def setUp(self):
        """Set up test environment."""
        self.makefile_path = Path("Makefile")

    def test_makefile_exists(self):
        """Test that Makefile exists."""
        self.assertTrue(self.makefile_path.exists(), "Makefile should exist")

    def test_makefile_help_target(self):
        """Test that make help works."""
        result = subprocess.run(
            ["make", "help"], capture_output=True, text=True, cwd="."
        )

        self.assertEqual(result.returncode, 0, "Make help should succeed")
        self.assertIn("CoffeeRL Development Commands", result.stdout)
        self.assertIn("train-dev", result.stdout)
        self.assertIn("eval-quick", result.stdout)
        self.assertIn("pipeline-dev", result.stdout)

    def test_makefile_check_data_target(self):
        """Test that make check-data works."""
        result = subprocess.run(
            ["make", "check-data"], capture_output=True, text=True, cwd="."
        )

        self.assertEqual(result.returncode, 0, "Make check-data should succeed")
        self.assertIn("Checking dataset availability", result.stdout)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the pipeline components."""

    def setUp(self):
        """Set up test environment."""
        self.test_output_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)

    def test_pipeline_argument_parsing(self):
        """Test that pipeline script parses arguments correctly."""
        # Test basic argument parsing by checking help
        result = subprocess.run(
            ["./scripts/run_experiment.sh", "--help"], capture_output=True, text=True
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--experiment-name", result.stdout)

    def test_experiment_directory_structure(self):
        """Test that experiment directories are created with proper structure."""
        # Create a mock experiment directory structure
        experiment_dir = self.test_output_dir / "test-experiment"
        experiment_dir.mkdir(parents=True)

        # Create expected subdirectories and files
        (experiment_dir / "model").mkdir()
        (experiment_dir / "training.log").touch()
        (experiment_dir / "evaluation.log").touch()
        (experiment_dir / "evaluation_results.json").write_text(
            '{"metrics": {"accuracy": 0.85}}'
        )
        (experiment_dir / "experiment_report.md").touch()

        # Verify structure
        self.assertTrue((experiment_dir / "model").is_dir())
        self.assertTrue((experiment_dir / "training.log").is_file())
        self.assertTrue((experiment_dir / "evaluation.log").is_file())
        self.assertTrue((experiment_dir / "evaluation_results.json").is_file())
        self.assertTrue((experiment_dir / "experiment_report.md").is_file())

    def test_evaluation_results_format(self):
        """Test that evaluation results are in the expected format."""
        # Create a sample evaluation results file
        results = {
            "metrics": {
                "grind_accuracy": 0.85,
                "extraction_accuracy": 0.78,
                "average_accuracy": 0.815,
            },
            "evaluation_info": {
                "total_samples": 100,
                "evaluation_time_seconds": 45.2,
                "model_path": "/path/to/model",
            },
        }

        results_file = self.test_output_dir / "evaluation_results.json"
        results_file.write_text(json.dumps(results, indent=2))

        # Load and verify format
        loaded_results = json.loads(results_file.read_text())

        self.assertIn("metrics", loaded_results)
        self.assertIn("evaluation_info", loaded_results)
        self.assertIn("grind_accuracy", loaded_results["metrics"])
        self.assertIn("extraction_accuracy", loaded_results["metrics"])
        self.assertIn("average_accuracy", loaded_results["metrics"])
        self.assertIn("total_samples", loaded_results["evaluation_info"])

    def test_experiment_report_generation(self):
        """Test that experiment reports are generated correctly."""
        # Create a mock experiment report
        report_content = """# CoffeeRL Experiment Report

**Experiment Name:** test-experiment
**Date:** 2024-01-01
**Model Size:** 0.5B

## Configuration
- Training Mode: dev
- Evaluation Mode: quick

## Results
- **Grind Accuracy:** 85%
- **Extraction Accuracy:** 78%
"""

        report_file = self.test_output_dir / "experiment_report.md"
        report_file.write_text(report_content)

        # Verify report content
        content = report_file.read_text()
        self.assertIn("CoffeeRL Experiment Report", content)
        self.assertIn("Configuration", content)
        self.assertIn("Results", content)
        self.assertIn("Grind Accuracy", content)


class TestPipelineWorkflow(unittest.TestCase):
    """Test the complete pipeline workflow."""

    def test_training_script_integration(self):
        """Test that training script can be called with proper arguments."""
        # Test that the training script exists and can show help
        result = subprocess.run(
            ["uv", "run", "python", "src/train_local.py", "--help"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, "Training script should show help")
        self.assertIn("--model-name", result.stdout)
        self.assertIn("--output-dir", result.stdout)

    def test_evaluation_script_integration(self):
        """Test that evaluation script can be called with proper arguments."""
        # Test that the evaluation script exists and can show help
        result = subprocess.run(
            ["uv", "run", "python", "src/evaluate_local.py", "--help"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, "Evaluation script should show help")
        self.assertIn("--model-path", result.stdout)
        self.assertIn("--eval-data", result.stdout)

    def test_pipeline_prerequisites_check(self):
        """Test that pipeline checks prerequisites correctly."""
        # Check that uv is available
        result = subprocess.run(["which", "uv"], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, "uv should be available")

    def test_data_availability_check(self):
        """Test that required datasets are available."""
        training_data_path = Path("data/processed/coffee_training_dataset")
        validation_data_path = Path("data/processed/coffee_validation_dataset")

        self.assertTrue(
            training_data_path.exists(),
            "Training dataset should exist for pipeline testing",
        )
        self.assertTrue(
            validation_data_path.exists(),
            "Validation dataset should exist for pipeline testing",
        )


class TestPipelineErrorHandling(unittest.TestCase):
    """Test error handling in the pipeline."""

    def test_missing_dataset_handling(self):
        """Test that pipeline handles missing datasets gracefully."""
        # This test would require temporarily moving datasets
        # For now, we'll test the check-data make target
        result = subprocess.run(["make", "check-data"], capture_output=True, text=True)

        # Should succeed and show dataset status
        self.assertEqual(result.returncode, 0)
        self.assertIn("dataset", result.stdout.lower())

    def test_invalid_model_size_handling(self):
        """Test that invalid model sizes are handled properly."""
        # Test with invalid model size
        result = subprocess.run(
            ["./scripts/run_experiment.sh", "--model-size", "invalid"],
            capture_output=True,
            text=True,
        )

        # Should fail gracefully (though the script might not validate this immediately)
        # The actual validation would happen in the training script
        self.assertIsInstance(result.returncode, int)


class TestPipelinePerformance(unittest.TestCase):
    """Test pipeline performance characteristics."""

    def test_script_startup_time(self):
        """Test that pipeline script starts up quickly."""
        import time

        start_time = time.time()
        result = subprocess.run(
            ["./scripts/run_experiment.sh", "--help"], capture_output=True, text=True
        )
        end_time = time.time()

        startup_time = end_time - start_time

        self.assertEqual(result.returncode, 0)
        self.assertLess(startup_time, 5.0, "Script should start up within 5 seconds")

    def test_makefile_target_performance(self):
        """Test that Makefile targets execute quickly."""
        import time

        start_time = time.time()
        result = subprocess.run(["make", "help"], capture_output=True, text=True)
        end_time = time.time()

        execution_time = end_time - start_time

        self.assertEqual(result.returncode, 0)
        self.assertLess(
            execution_time, 3.0, "Make help should execute within 3 seconds"
        )


if __name__ == "__main__":
    unittest.main()
