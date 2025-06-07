"""
Tests for the model performance analysis script.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

# Add src to path for imports
sys.path.insert(0, "src")

from analyze_performance import ModelPerformanceAnalyzer

matplotlib.use("Agg")  # Use non-interactive backend for testing


class TestModelPerformanceAnalyzer(unittest.TestCase):
    """Test the ModelPerformanceAnalyzer class."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.analyzer = ModelPerformanceAnalyzer(str(self.test_dir))

        # Sample evaluation results
        self.sample_results_0_5b = {
            "metrics": {
                "grind_accuracy": 0.75,
                "extraction_accuracy": 0.68,
                "average_accuracy": 0.715,
            },
            "evaluation_info": {
                "model_path": "models/qwen2-0.5b-coffee",
                "total_samples": 100,
                "evaluation_time_seconds": 45.2,
            },
        }

        self.sample_results_1_5b = {
            "metrics": {
                "grind_accuracy": 0.85,
                "extraction_accuracy": 0.78,
                "average_accuracy": 0.815,
            },
            "evaluation_info": {
                "model_path": "models/qwen2-1.5b-coffee",
                "total_samples": 100,
                "evaluation_time_seconds": 67.8,
            },
        }

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_analyzer_initialization(self):
        """Test that the analyzer initializes correctly."""
        self.assertTrue(self.test_dir.exists())
        self.assertIsInstance(self.analyzer.success_criteria, dict)
        self.assertIn("minimum", self.analyzer.success_criteria)
        self.assertIn("stretch", self.analyzer.success_criteria)
        self.assertIsInstance(self.analyzer.model_sizes, dict)

    def test_load_evaluation_results_valid_file(self):
        """Test loading valid evaluation results."""
        # Create a test results file
        results_file = self.test_dir / "test_results.json"
        with open(results_file, "w") as f:
            json.dump(self.sample_results_0_5b, f)

        results = self.analyzer.load_evaluation_results(str(results_file))
        self.assertEqual(results, self.sample_results_0_5b)

    def test_load_evaluation_results_missing_file(self):
        """Test loading non-existent file."""
        results = self.analyzer.load_evaluation_results("nonexistent.json")
        self.assertEqual(results, {})

    def test_load_evaluation_results_invalid_json(self):
        """Test loading invalid JSON file."""
        # Create an invalid JSON file
        invalid_file = self.test_dir / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("invalid json content")

        results = self.analyzer.load_evaluation_results(str(invalid_file))
        self.assertEqual(results, {})

    def test_extract_model_name_from_path(self):
        """Test extracting model name from evaluation results."""
        # Test 0.5B model
        name = self.analyzer.extract_model_name(self.sample_results_0_5b)
        self.assertEqual(name, "Qwen2-0.5B")

        # Test 1.5B model
        name = self.analyzer.extract_model_name(self.sample_results_1_5b)
        self.assertEqual(name, "Qwen2-1.5B")

        # Test unknown model
        unknown_results = {"evaluation_info": {"model_path": "models/unknown-model"}}
        name = self.analyzer.extract_model_name(unknown_results)
        self.assertEqual(name, "Unknown Model")

    def test_analyze_single_model(self):
        """Test analyzing a single model."""
        analysis = self.analyzer.analyze_single_model(self.sample_results_0_5b)

        # Check basic metrics
        self.assertEqual(analysis["model_name"], "Qwen2-0.5B")
        self.assertEqual(analysis["model_size_b"], 0.5)
        self.assertEqual(analysis["grind_accuracy"], 0.75)
        self.assertEqual(analysis["extraction_accuracy"], 0.68)
        self.assertEqual(analysis["average_accuracy"], 0.715)

        # Check success criteria evaluation
        self.assertTrue(analysis["meets_minimum_grind"])  # 0.75 >= 0.70
        self.assertFalse(analysis["meets_stretch_grind"])  # 0.75 < 0.90
        self.assertTrue(analysis["meets_minimum_extraction"])  # 0.68 >= 0.60
        self.assertFalse(analysis["meets_stretch_extraction"])  # 0.68 < 0.85

        # Check efficiency metrics
        self.assertAlmostEqual(analysis["accuracy_per_param"], 0.715 / 0.5, places=3)

    def test_analyze_single_model_with_custom_name(self):
        """Test analyzing a single model with custom name."""
        analysis = self.analyzer.analyze_single_model(
            self.sample_results_0_5b, model_name="Custom Model"
        )
        self.assertEqual(analysis["model_name"], "Custom Model")

    def test_analyze_single_model_empty_results(self):
        """Test analyzing empty results."""
        analysis = self.analyzer.analyze_single_model({})
        self.assertEqual(analysis, {})

    def test_compare_models(self):
        """Test comparing multiple models."""
        # Analyze both models
        analysis_0_5b = self.analyzer.analyze_single_model(self.sample_results_0_5b)
        analysis_1_5b = self.analyzer.analyze_single_model(self.sample_results_1_5b)

        model_analyses = [analysis_0_5b, analysis_1_5b]
        comparison = self.analyzer.compare_models(model_analyses)

        # Check comparison results
        self.assertEqual(comparison["total_models"], 2)
        self.assertEqual(comparison["models_meeting_minimum"], 2)  # Both meet minimum
        self.assertEqual(
            comparison["models_meeting_stretch"], 0
        )  # Neither meets stretch
        self.assertEqual(comparison["best_accuracy_model"], "Qwen2-1.5B")
        self.assertEqual(comparison["best_accuracy_score"], 0.85)

    def test_compare_models_empty_list(self):
        """Test comparing empty model list."""
        comparison = self.analyzer.compare_models([])
        self.assertEqual(comparison, {})

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_performance_comparison(self, mock_close, mock_savefig):
        """Test creating performance comparison plots."""
        # Analyze both models
        analysis_0_5b = self.analyzer.analyze_single_model(self.sample_results_0_5b)
        analysis_1_5b = self.analyzer.analyze_single_model(self.sample_results_1_5b)

        model_analyses = [analysis_0_5b, analysis_1_5b]

        # This should not raise an exception
        self.analyzer.plot_performance_comparison(model_analyses)

        # Check that savefig was called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    def test_plot_performance_comparison_empty_list(self):
        """Test plotting with empty model list."""
        # This should not raise an exception
        self.analyzer.plot_performance_comparison([])

    def test_generate_report(self):
        """Test generating a comprehensive report."""
        # Analyze both models
        analysis_0_5b = self.analyzer.analyze_single_model(self.sample_results_0_5b)
        analysis_1_5b = self.analyzer.analyze_single_model(self.sample_results_1_5b)

        model_analyses = [analysis_0_5b, analysis_1_5b]
        comparison = self.analyzer.compare_models(model_analyses)

        # Generate report
        self.analyzer.generate_report(model_analyses, comparison)

        # Check that report file was created
        report_path = self.test_dir / "performance_analysis_report.md"
        self.assertTrue(report_path.exists())

        # Check report content
        with open(report_path, "r") as f:
            content = f.read()

        self.assertIn("CoffeeRL Model Performance Analysis Report", content)
        self.assertIn("Executive Summary", content)
        self.assertIn("Success Criteria Analysis", content)
        self.assertIn("Detailed Model Performance", content)
        self.assertIn("Qwen2-0.5B", content)
        self.assertIn("Qwen2-1.5B", content)

    def test_success_criteria_values(self):
        """Test that success criteria are set correctly."""
        criteria = self.analyzer.success_criteria

        # Check minimum criteria
        self.assertEqual(criteria["minimum"]["grind_direction_accuracy"], 0.70)
        self.assertEqual(criteria["minimum"]["extraction_category_accuracy"], 0.60)

        # Check stretch criteria
        self.assertEqual(criteria["stretch"]["grind_direction_accuracy"], 0.90)
        self.assertEqual(criteria["stretch"]["extraction_category_accuracy"], 0.85)

    def test_model_sizes_mapping(self):
        """Test that model sizes are mapped correctly."""
        sizes = self.analyzer.model_sizes

        self.assertEqual(sizes["Qwen2-0.5B"], 0.5)
        self.assertEqual(sizes["Qwen2-1.5B"], 1.5)
        self.assertEqual(sizes["qwen2-0.5b"], 0.5)
        self.assertEqual(sizes["qwen2-1.5b"], 1.5)

    def test_efficiency_calculation(self):
        """Test efficiency metric calculation."""
        analysis = self.analyzer.analyze_single_model(self.sample_results_0_5b)

        expected_efficiency = 0.715 / 0.5  # average_accuracy / model_size_b
        self.assertAlmostEqual(
            analysis["accuracy_per_param"], expected_efficiency, places=3
        )

        expected_grind_efficiency = 0.75 / 0.5  # grind_accuracy / model_size_b
        self.assertAlmostEqual(
            analysis["grind_accuracy_per_param"], expected_grind_efficiency, places=3
        )

    def test_run_analysis_integration(self):
        """Test the complete analysis workflow."""
        # Create test results files
        results_file_1 = self.test_dir / "model1_results.json"
        results_file_2 = self.test_dir / "model2_results.json"

        with open(results_file_1, "w") as f:
            json.dump(self.sample_results_0_5b, f)

        with open(results_file_2, "w") as f:
            json.dump(self.sample_results_1_5b, f)

        # Run analysis
        results_files = [str(results_file_1), str(results_file_2)]
        model_names = ["Qwen2-0.5B", "Qwen2-1.5B"]

        # Mock the plotting to avoid display issues in tests
        with patch.object(self.analyzer, "plot_performance_comparison"):
            self.analyzer.run_analysis(results_files, model_names)

        # Check that output files were created
        self.assertTrue((self.test_dir / "performance_analysis_report.md").exists())

    def test_edge_case_zero_model_size(self):
        """Test handling of unknown model with zero size."""
        # Create results with unknown model
        unknown_results = {
            "metrics": {
                "grind_accuracy": 0.80,
                "extraction_accuracy": 0.70,
                "average_accuracy": 0.75,
            },
            "evaluation_info": {
                "model_path": "models/unknown-model",
                "total_samples": 50,
                "evaluation_time_seconds": 30.0,
            },
        }

        analysis = self.analyzer.analyze_single_model(unknown_results)

        # Should handle zero model size gracefully
        self.assertEqual(analysis["model_size_b"], 0)
        self.assertEqual(analysis["accuracy_per_param"], 0)
        self.assertEqual(analysis["grind_accuracy_per_param"], 0)

    def test_missing_metrics_handling(self):
        """Test handling of missing metrics in results."""
        incomplete_results = {
            "metrics": {
                "grind_accuracy": 0.80
                # Missing extraction_accuracy and average_accuracy
            },
            "evaluation_info": {
                "model_path": "models/qwen2-0.5b-test",
                "total_samples": 50,
            },
        }

        analysis = self.analyzer.analyze_single_model(incomplete_results)

        # Should handle missing metrics gracefully
        self.assertEqual(analysis["grind_accuracy"], 0.80)
        self.assertEqual(analysis["extraction_accuracy"], 0)  # Default value
        self.assertEqual(analysis["average_accuracy"], 0)  # Default value


class TestAnalysisScriptIntegration(unittest.TestCase):
    """Integration tests for the analysis script."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_script_help_output(self):
        """Test that the script shows help correctly."""
        import subprocess

        result = subprocess.run(
            ["python", "src/analyze_performance.py", "--help"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Analyze CoffeeRL model performance", result.stdout)
        self.assertIn("--results", result.stdout)
        self.assertIn("--model-names", result.stdout)
        self.assertIn("--output-dir", result.stdout)

    def test_script_missing_required_args(self):
        """Test that the script fails with missing required arguments."""
        import subprocess

        result = subprocess.run(
            ["python", "src/analyze_performance.py"], capture_output=True, text=True
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
