#!/usr/bin/env python3
"""Tests for parameter_space_explorer module."""

import tempfile
from pathlib import Path

import pytest

from src.parameter_space import (
    BrewingParameters,
    BrewMethod,
    GrindSize,
    ParameterRanges,
)
from src.parameter_space_explorer import (
    ParameterSpaceExplorer,
    create_explorer_from_data,
    suggest_next_experiments_simple,
)


class TestParameterSpaceExplorer:
    """Test ParameterSpaceExplorer class."""

    def test_initialization(self):
        """Test basic initialization."""
        explorer = ParameterSpaceExplorer()

        assert explorer.parameter_ranges is not None
        assert explorer.kdtree_explorer is not None
        assert len(explorer.exploration_history) == 0

    def test_initialization_with_custom_ranges(self):
        """Test initialization with custom parameter ranges."""
        custom_ranges = ParameterRanges(
            water_temp_min=85.0,
            water_temp_max=95.0,
        )
        explorer = ParameterSpaceExplorer(parameter_ranges=custom_ranges)

        assert explorer.parameter_ranges.water_temp_min == 85.0
        assert explorer.parameter_ranges.water_temp_max == 95.0

    def test_load_existing_experiments(self):
        """Test loading existing experiments."""
        explorer = ParameterSpaceExplorer()

        experiments = []
        for i in range(5):
            experiment = {
                "water_temp": 90.0 + i,
                "coffee_dose": 18.0,
                "water_amount": 300.0,
                "grind_size": "medium",
                "brew_time": 240.0,
                "brew_method": "pour_over",
            }
            experiments.append(experiment)

        count = explorer.load_existing_experiments(experiments)
        assert count == 5
        assert len(explorer.exploration_history) == 1
        assert explorer.exploration_history[0]["action"] == "load_experiments"

    def test_load_empty_experiments(self):
        """Test loading empty experiment list."""
        explorer = ParameterSpaceExplorer()
        count = explorer.load_existing_experiments([])
        assert count == 0

    def test_suggest_experiments_basic(self):
        """Test basic experiment suggestion."""
        explorer = ParameterSpaceExplorer()

        # Load some initial experiments
        experiments = []
        for i in range(3):
            experiment = {
                "water_temp": 90.0 + i,
                "coffee_dose": 18.0,
                "water_amount": 300.0,
                "grind_size": "medium",
                "brew_time": 240.0,
                "brew_method": "pour_over",
            }
            experiments.append(experiment)

        explorer.load_existing_experiments(experiments)

        # Get suggestions
        suggestions = explorer.suggest_experiments(
            num_suggestions=3, brew_method=BrewMethod.POUR_OVER
        )

        assert len(suggestions) == 3
        for suggestion in suggestions:
            assert isinstance(suggestion, BrewingParameters)
            assert suggestion.brew_method == BrewMethod.POUR_OVER

    def test_suggest_experiments_detailed(self):
        """Test detailed experiment suggestion."""
        explorer = ParameterSpaceExplorer()

        # Load some initial experiments
        experiments = []
        for i in range(3):
            experiment = {
                "water_temp": 90.0 + i,
                "coffee_dose": 18.0,
                "water_amount": 300.0,
                "grind_size": "medium",
                "brew_time": 240.0,
                "brew_method": "pour_over",
            }
            experiments.append(experiment)

        explorer.load_existing_experiments(experiments)

        # Get detailed suggestions
        candidates = explorer.suggest_experiments(
            num_suggestions=3, brew_method=BrewMethod.POUR_OVER, return_detailed=True
        )

        assert len(candidates) == 3
        for candidate in candidates:
            assert hasattr(candidate, "parameters")
            assert hasattr(candidate, "priority_score")
            assert hasattr(candidate, "reasoning")

    def test_add_experiment_result(self):
        """Test adding experiment results."""
        explorer = ParameterSpaceExplorer()

        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )

        results = {"yield": 20.5, "rating": 4.2}
        index = explorer.add_experiment_result(
            parameters=params, results=results, experiment_id="test_001"
        )

        assert index == 0
        assert len(explorer.exploration_history) == 1
        assert explorer.exploration_history[0]["action"] == "add_experiment_result"

    def test_add_experiment_result_from_dict(self):
        """Test adding experiment results from dictionary."""
        explorer = ParameterSpaceExplorer()

        params_dict = {
            "water_temp": 93.0,
            "coffee_dose": 18.0,
            "water_amount": 300.0,
            "grind_size": "medium",
            "brew_time": 240.0,
            "brew_method": "pour_over",
        }

        index = explorer.add_experiment_result(parameters=params_dict)
        assert index == 0

    def test_analyze_coverage(self):
        """Test coverage analysis."""
        explorer = ParameterSpaceExplorer()

        # Initially empty
        analysis = explorer.analyze_coverage()
        assert analysis["total_experiments"] == 0
        assert analysis["exploration_efficiency"] == 0.0
        assert analysis["coverage_quality"] == "insufficient_data"

        # Add some experiments
        experiments = []
        for i in range(10):
            experiment = {
                "water_temp": 85.0 + i,
                "coffee_dose": 15.0 + i,
                "water_amount": 250.0 + i * 10,
                "grind_size": "medium",
                "brew_time": 200.0 + i * 10,
                "brew_method": "pour_over",
            }
            experiments.append(experiment)

        explorer.load_existing_experiments(experiments)

        analysis = explorer.analyze_coverage()
        assert analysis["total_experiments"] == 10
        assert analysis["exploration_efficiency"] > 0.0
        assert "coverage_quality" in analysis
        assert "recommended_next_experiments" in analysis

    def test_get_exploration_report(self):
        """Test exploration report generation."""
        explorer = ParameterSpaceExplorer()

        # Add some experiments
        experiments = []
        for i in range(5):
            experiment = {
                "water_temp": 90.0 + i,
                "coffee_dose": 18.0,
                "water_amount": 300.0,
                "grind_size": "medium",
                "brew_time": 240.0,
                "brew_method": "pour_over",
            }
            experiments.append(experiment)

        explorer.load_existing_experiments(experiments)

        report = explorer.get_exploration_report()
        assert "timestamp" in report
        assert "exploration_stats" in report
        assert "coverage_analysis" in report
        assert "recommendations" in report
        assert isinstance(report["recommendations"], list)

    def test_save_and_load_state(self):
        """Test saving and loading exploration state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_state.json"
            explorer = ParameterSpaceExplorer(save_path=str(save_path))

            # Add some experiments
            experiments = []
            for i in range(3):
                experiment = {
                    "water_temp": 90.0 + i,
                    "coffee_dose": 18.0,
                    "water_amount": 300.0,
                    "grind_size": "medium",
                    "brew_time": 240.0,
                    "brew_method": "pour_over",
                }
                experiments.append(experiment)

            explorer.load_existing_experiments(experiments)

            # Save state
            explorer.save_state()
            assert save_path.exists()

            # Create new explorer and load state
            new_explorer = ParameterSpaceExplorer(save_path=str(save_path))
            success = new_explorer.load_state()
            assert success

            # Verify loaded state
            stats = new_explorer.kdtree_explorer.get_exploration_stats()
            assert stats["total_experiments"] == 3

    def test_load_nonexistent_state(self):
        """Test loading from nonexistent file."""
        explorer = ParameterSpaceExplorer(save_path="nonexistent.json")
        success = explorer.load_state()
        assert not success

    def test_exploration_efficiency_calculation(self):
        """Test exploration efficiency calculation."""
        explorer = ParameterSpaceExplorer()

        # Test with no experiments
        efficiency = explorer._calculate_exploration_efficiency()
        assert efficiency == 0.0

        # Add experiments and test efficiency
        experiments = []
        for i in range(5):
            experiment = {
                "water_temp": 90.0 + i,
                "coffee_dose": 18.0,
                "water_amount": 300.0,
                "grind_size": "medium",
                "brew_time": 240.0,
                "brew_method": "pour_over",
            }
            experiments.append(experiment)

        explorer.load_existing_experiments(experiments)
        efficiency = explorer._calculate_exploration_efficiency()
        assert 0.0 <= efficiency <= 1.0

    def test_coverage_quality_assessment(self):
        """Test coverage quality assessment."""
        explorer = ParameterSpaceExplorer()

        # Test different coverage scenarios
        test_stats = {
            "total_experiments": 5,
            "coverage_estimate": 0.1,
            "avg_nearest_distance": 0.3,
        }

        quality = explorer._assess_coverage_quality(test_stats)
        assert quality in [
            "insufficient_data",
            "poor_coverage",
            "moderate_coverage",
            "good_coverage",
            "excellent_coverage",
        ]

    def test_recommendations_generation(self):
        """Test recommendation generation."""
        explorer = ParameterSpaceExplorer()

        analysis = {
            "total_experiments": 5,
            "coverage_quality": "poor_coverage",
            "exploration_efficiency": 0.3,
            "avg_nearest_distance": 0.4,
        }

        unexplored_regions = []  # Mock empty unexplored regions
        recommendations = explorer._generate_recommendations(
            analysis, unexplored_regions
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_explorer_from_data(self):
        """Test creating explorer from data."""
        experiments = []
        for i in range(3):
            experiment = {
                "water_temp": 90.0 + i,
                "coffee_dose": 18.0,
                "water_amount": 300.0,
                "grind_size": "medium",
                "brew_time": 240.0,
                "brew_method": "pour_over",
            }
            experiments.append(experiment)

        explorer = create_explorer_from_data(experiments)
        assert isinstance(explorer, ParameterSpaceExplorer)

        stats = explorer.kdtree_explorer.get_exploration_stats()
        assert stats["total_experiments"] == 3

    def test_create_explorer_from_data_with_custom_ranges(self):
        """Test creating explorer with custom ranges."""
        experiments = []
        custom_ranges = ParameterRanges(water_temp_min=85.0, water_temp_max=95.0)

        explorer = create_explorer_from_data(experiments, custom_ranges)
        assert explorer.parameter_ranges.water_temp_min == 85.0

    def test_suggest_next_experiments_simple(self):
        """Test simple experiment suggestion function."""
        experiments = []
        for i in range(5):
            experiment = {
                "water_temp": 90.0 + i,
                "coffee_dose": 18.0,
                "water_amount": 300.0,
                "grind_size": "medium",
                "brew_time": 240.0,
                "brew_method": "pour_over",
            }
            experiments.append(experiment)

        suggestions = suggest_next_experiments_simple(
            experiments=experiments, num_suggestions=3, brew_method="pour_over"
        )

        assert len(suggestions) == 3
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)
            assert "water_temp" in suggestion
            assert "brew_method" in suggestion
            assert suggestion["brew_method"] == "pour_over"

    def test_suggest_next_experiments_simple_different_methods(self):
        """Test simple suggestion with different brew methods."""
        experiments = []

        # Test different brew methods
        for method in ["espresso", "french_press", "aeropress", "cold_brew"]:
            suggestions = suggest_next_experiments_simple(
                experiments=experiments, num_suggestions=2, brew_method=method
            )

            assert len(suggestions) == 2
            for suggestion in suggestions:
                assert suggestion["brew_method"] == method

    def test_suggest_next_experiments_simple_with_optional_params(self):
        """Test simple suggestion includes optional parameters."""
        experiments = []

        # Test espresso (should include pressure)
        suggestions = suggest_next_experiments_simple(
            experiments=experiments, num_suggestions=1, brew_method="espresso"
        )

        assert len(suggestions) == 1
        suggestion = suggestions[0]
        assert "pressure" in suggestion
        assert suggestion["brew_method"] == "espresso"

    def test_suggest_next_experiments_simple_invalid_method(self):
        """Test simple suggestion with invalid brew method."""
        experiments = []

        suggestions = suggest_next_experiments_simple(
            experiments=experiments, num_suggestions=2, brew_method="invalid_method"
        )

        assert len(suggestions) == 2
        # Should default to pour_over
        for suggestion in suggestions:
            assert suggestion["brew_method"] == "pour_over"


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_complete_exploration_workflow(self):
        """Test a complete exploration workflow."""
        explorer = ParameterSpaceExplorer()

        # Step 1: Load initial experiments
        initial_experiments = []
        for i in range(5):
            experiment = {
                "water_temp": 90.0 + i,
                "coffee_dose": 18.0,
                "water_amount": 300.0,
                "grind_size": "medium",
                "brew_time": 240.0,
                "brew_method": "pour_over",
            }
            initial_experiments.append(experiment)

        explorer.load_existing_experiments(initial_experiments)

        # Step 2: Analyze coverage
        analysis = explorer.analyze_coverage()
        assert analysis["total_experiments"] == 5

        # Step 3: Get suggestions
        suggestions = explorer.suggest_experiments(num_suggestions=3)
        assert len(suggestions) == 3

        # Step 4: Add a new experiment result
        new_params = suggestions[0]  # Use first suggestion
        explorer.add_experiment_result(
            parameters=new_params,
            results={"yield": 21.0, "rating": 4.5},
            experiment_id="new_exp_001",
        )

        # Step 5: Verify the experiment was added
        updated_analysis = explorer.analyze_coverage()
        assert updated_analysis["total_experiments"] == 6

        # Step 6: Get exploration report
        report = explorer.get_exploration_report()
        assert "recommendations" in report
        assert len(report["recommendations"]) > 0

    def test_workflow_with_different_brew_methods(self):
        """Test workflow with multiple brew methods."""
        explorer = ParameterSpaceExplorer()

        # Add experiments for different brew methods
        methods = ["pour_over", "espresso", "french_press"]
        for method in methods:
            for i in range(2):
                experiment = {
                    "water_temp": 90.0 + i,
                    "coffee_dose": 18.0,
                    "water_amount": 300.0 if method != "espresso" else 36.0,
                    "grind_size": "medium" if method != "espresso" else "fine",
                    "brew_time": 240.0 if method != "espresso" else 30.0,
                    "brew_method": method,
                }
                if method == "espresso":
                    experiment["pressure"] = 9.0

                explorer.add_experiment_result(parameters=experiment)

        # Get suggestions for each method
        for method_enum in [
            BrewMethod.POUR_OVER,
            BrewMethod.ESPRESSO,
            BrewMethod.FRENCH_PRESS,
        ]:
            suggestions = explorer.suggest_experiments(
                num_suggestions=2, brew_method=method_enum
            )
            assert len(suggestions) == 2
            for suggestion in suggestions:
                assert suggestion.brew_method == method_enum


if __name__ == "__main__":
    pytest.main([__file__])
