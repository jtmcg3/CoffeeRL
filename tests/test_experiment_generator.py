#!/usr/bin/env python3
"""Tests for experiment_generator module."""

from unittest.mock import MagicMock

import pytest
import torch

from src.experiment_generator import ExperimentGenerator
from src.kdtree_explorer import SimpleKDTreeExplorer
from src.parameter_space import BrewingParameters, BrewMethod, GrindSize
from src.parameter_space_explorer import ParameterSpaceExplorer
from src.uncertainty_estimator import UncertaintyEstimator


class TestExperimentGenerator:
    """Test ExperimentGenerator class."""

    def test_initialization_empty(self):
        """Test basic initialization without components."""
        generator = ExperimentGenerator()

        assert generator.uncertainty_estimator is None
        assert generator.parameter_explorer is None
        assert generator.kdtree_explorer is None
        assert not generator.is_configured()

    def test_initialization_with_components(self):
        """Test initialization with all components."""
        # Create mock components
        uncertainty_estimator = MagicMock(spec=UncertaintyEstimator)
        parameter_explorer = MagicMock(spec=ParameterSpaceExplorer)
        kdtree_explorer = MagicMock(spec=SimpleKDTreeExplorer)

        generator = ExperimentGenerator(
            uncertainty_estimator=uncertainty_estimator,
            parameter_explorer=parameter_explorer,
            kdtree_explorer=kdtree_explorer,
        )

        assert generator.uncertainty_estimator is uncertainty_estimator
        assert generator.parameter_explorer is parameter_explorer
        assert generator.kdtree_explorer is kdtree_explorer
        assert generator.is_configured()

    def test_set_uncertainty_estimator(self):
        """Test setting uncertainty estimator."""
        generator = ExperimentGenerator()
        estimator = MagicMock(spec=UncertaintyEstimator)

        generator.set_uncertainty_estimator(estimator)

        assert generator.uncertainty_estimator is estimator

    def test_set_parameter_explorer(self):
        """Test setting parameter explorer."""
        generator = ExperimentGenerator()
        explorer = MagicMock(spec=ParameterSpaceExplorer)

        generator.set_parameter_explorer(explorer)

        assert generator.parameter_explorer is explorer

    def test_set_kdtree_explorer(self):
        """Test setting k-d tree explorer."""
        generator = ExperimentGenerator()
        explorer = MagicMock(spec=SimpleKDTreeExplorer)

        generator.set_kdtree_explorer(explorer)

        assert generator.kdtree_explorer is explorer

    def test_is_configured_with_single_component(self):
        """Test configuration check with single component."""
        generator = ExperimentGenerator()
        assert not generator.is_configured()

        # Add uncertainty estimator
        generator.set_uncertainty_estimator(MagicMock(spec=UncertaintyEstimator))
        assert generator.is_configured()

    def test_get_configuration_status(self):
        """Test configuration status reporting."""
        generator = ExperimentGenerator()

        # Initially all False
        status = generator.get_configuration_status()
        assert status["uncertainty_estimator"] is False
        assert status["parameter_explorer"] is False
        assert status["kdtree_explorer"] is False

        # Add components one by one
        generator.set_uncertainty_estimator(MagicMock(spec=UncertaintyEstimator))
        status = generator.get_configuration_status()
        assert status["uncertainty_estimator"] is True
        assert status["parameter_explorer"] is False
        assert status["kdtree_explorer"] is False

        generator.set_parameter_explorer(MagicMock(spec=ParameterSpaceExplorer))
        status = generator.get_configuration_status()
        assert status["uncertainty_estimator"] is True
        assert status["parameter_explorer"] is True
        assert status["kdtree_explorer"] is False

        generator.set_kdtree_explorer(MagicMock(spec=SimpleKDTreeExplorer))
        status = generator.get_configuration_status()
        assert status["uncertainty_estimator"] is True
        assert status["parameter_explorer"] is True
        assert status["kdtree_explorer"] is True

    def test_generate_uncertainty_based_experiments_signature(self):
        """Test uncertainty-based experiment generation method signature."""
        generator = ExperimentGenerator()

        # Test with default parameters (no uncertainty estimator)
        result = generator.generate_uncertainty_based_experiments()
        assert isinstance(result, list)
        assert len(result) == 0  # Returns empty list when no uncertainty estimator

        # Test with custom parameters (no uncertainty estimator)
        result = generator.generate_uncertainty_based_experiments(
            num_experiments=3, uncertainty_threshold=0.8
        )
        assert isinstance(result, list)
        assert len(result) == 0  # Returns empty list when no uncertainty estimator

    def test_generate_uncertainty_based_experiments_with_estimator(self):
        """Test uncertainty-based experiment generation with mock estimator."""
        # Create mock uncertainty estimator
        mock_estimator = MagicMock(spec=UncertaintyEstimator)
        mock_estimator.estimate_uncertainty.return_value = {
            "total_uncertainty": torch.tensor([1.5, 2.0])
        }

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        generator = ExperimentGenerator(uncertainty_estimator=mock_estimator)

        # Test with tokenizer
        result = generator.generate_uncertainty_based_experiments(
            num_experiments=2,
            tokenizer=mock_tokenizer,
            brew_method=BrewMethod.POUR_OVER,
        )

        assert isinstance(result, list)
        # Should return some results with mock estimator
        assert len(result) <= 2

    def test_generate_exploration_based_experiments_signature(self):
        """Test exploration-based experiment generation method signature."""
        generator = ExperimentGenerator()

        # Test with default parameters (no k-d tree explorer)
        result = generator.generate_exploration_based_experiments()
        assert isinstance(result, list)
        assert len(result) == 0  # Returns empty list when no k-d tree explorer

        # Test with custom parameters (no k-d tree explorer)
        result = generator.generate_exploration_based_experiments(
            num_experiments=4, brew_method=BrewMethod.ESPRESSO
        )
        assert isinstance(result, list)
        assert len(result) == 0  # Returns empty list when no k-d tree explorer

    def test_generate_exploration_based_experiments_with_kdtree(self):
        """Test exploration-based experiment generation with k-d tree explorer."""
        # Create mock k-d tree explorer
        mock_kdtree = MagicMock(spec=SimpleKDTreeExplorer)
        mock_kdtree.suggest_experiments.return_value = [
            BrewingParameters(
                water_temp=90.0,
                coffee_dose=18.0,
                water_amount=300.0,
                grind_size=GrindSize.MEDIUM,
                brew_time=240.0,
                brew_method=BrewMethod.POUR_OVER,
            )
        ]
        mock_kdtree.calculate_sparsity.return_value = 0.8

        generator = ExperimentGenerator(kdtree_explorer=mock_kdtree)

        # Test sparsity-guided strategy
        result = generator.generate_exploration_based_experiments(
            num_experiments=2,
            brew_method=BrewMethod.POUR_OVER,
            exploration_strategy="sparsity_guided",
        )

        assert isinstance(result, list)
        assert len(result) <= 2

        # Test basic k-d tree strategy
        result = generator.generate_exploration_based_experiments(
            num_experiments=1,
            brew_method=BrewMethod.POUR_OVER,
            exploration_strategy="random",
        )

        assert isinstance(result, list)
        # Should use k-d tree suggest_experiments method
        assert mock_kdtree.suggest_experiments.called

    def test_generate_experiments_signature(self):
        """Test main experiment generation method signature."""
        generator = ExperimentGenerator()

        # Test with default parameters
        result = generator.generate_experiments()
        assert isinstance(result, list)
        assert len(result) == 0  # Placeholder returns empty list

        # Test with custom parameters
        result = generator.generate_experiments(
            num_experiments=8,
            uncertainty_weight=0.6,
            exploration_weight=0.4,
            brew_method=BrewMethod.FRENCH_PRESS,
        )
        assert isinstance(result, list)
        assert len(result) == 0  # Placeholder returns empty list

    def test_generate_experiments_weight_distribution(self):
        """Test that experiment generation respects weight distribution."""
        generator = ExperimentGenerator()

        # Mock the individual generation methods to return identifiable results
        def mock_uncertainty_gen(num_experiments, **kwargs):
            return [f"uncertainty_{i}" for i in range(num_experiments)]

        def mock_exploration_gen(num_experiments, brew_method):
            return [f"exploration_{i}" for i in range(num_experiments)]

        generator.generate_uncertainty_based_experiments = mock_uncertainty_gen
        generator.generate_exploration_based_experiments = mock_exploration_gen

        # Test 50/50 split
        result = generator.generate_experiments(
            num_experiments=10, uncertainty_weight=0.5, exploration_weight=0.5
        )
        assert len(result) == 10
        uncertainty_count = sum(
            1 for item in result if str(item).startswith("uncertainty")
        )
        exploration_count = sum(
            1 for item in result if str(item).startswith("exploration")
        )
        assert uncertainty_count == 5
        assert exploration_count == 5

        # Test 70/30 split
        result = generator.generate_experiments(
            num_experiments=10, uncertainty_weight=0.7, exploration_weight=0.3
        )
        assert len(result) == 10
        uncertainty_count = sum(
            1 for item in result if str(item).startswith("uncertainty")
        )
        exploration_count = sum(
            1 for item in result if str(item).startswith("exploration")
        )
        assert uncertainty_count == 7
        assert exploration_count == 3

    def test_score_experiment_candidates_signature(self):
        """Test experiment candidate scoring method signature."""
        generator = ExperimentGenerator()

        # Create sample brewing parameters
        candidates = [
            BrewingParameters(
                water_temp=90.0,
                coffee_dose=18.0,
                water_amount=300.0,
                grind_size=GrindSize.MEDIUM,
                brew_time=240.0,
                brew_method=BrewMethod.POUR_OVER,
            ),
            BrewingParameters(
                water_temp=95.0,
                coffee_dose=20.0,
                water_amount=350.0,
                grind_size=GrindSize.FINE,
                brew_time=180.0,
                brew_method=BrewMethod.ESPRESSO,
            ),
        ]

        result = generator._score_experiment_candidates(candidates)

        assert isinstance(result, list)
        assert len(result) == 2
        for params, score in result:
            assert isinstance(params, BrewingParameters)
            assert isinstance(score, float)
            assert score == 0.0  # Placeholder returns 0.0 scores

    def test_score_experiment_candidates_sorting(self):
        """Test that candidate scoring returns sorted results."""
        generator = ExperimentGenerator()

        # Create sample candidates
        candidates = [
            BrewingParameters(
                water_temp=90.0,
                coffee_dose=18.0,
                water_amount=300.0,
                grind_size=GrindSize.MEDIUM,
                brew_time=240.0,
                brew_method=BrewMethod.POUR_OVER,
            )
        ]

        result = generator._score_experiment_candidates(candidates)

        # Verify sorting (descending by score)
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)


class TestExperimentGeneratorIntegration:
    """Integration tests for ExperimentGenerator with real components."""

    def test_integration_with_real_components(self):
        """Test ExperimentGenerator with actual component instances."""
        # Create real component instances
        parameter_explorer = ParameterSpaceExplorer()
        kdtree_explorer = SimpleKDTreeExplorer()

        generator = ExperimentGenerator(
            parameter_explorer=parameter_explorer,
            kdtree_explorer=kdtree_explorer,
        )

        assert generator.is_configured()
        assert generator.parameter_explorer is parameter_explorer
        assert generator.kdtree_explorer is kdtree_explorer

        # Test that methods can be called without errors
        result = generator.generate_experiments(num_experiments=5)
        assert isinstance(result, list)

    def test_partial_configuration(self):
        """Test generator with partial component configuration."""
        generator = ExperimentGenerator()

        # Add only parameter explorer
        parameter_explorer = ParameterSpaceExplorer()
        generator.set_parameter_explorer(parameter_explorer)

        assert generator.is_configured()
        status = generator.get_configuration_status()
        assert status["parameter_explorer"] is True
        assert status["uncertainty_estimator"] is False
        assert status["kdtree_explorer"] is False

    def test_parameters_to_text_conversion(self):
        """Test conversion of brewing parameters to text descriptions."""
        generator = ExperimentGenerator()

        # Test basic pour over parameters
        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,  # 4:00
            brew_method=BrewMethod.POUR_OVER,
            bloom_time=30.0,
        )

        text = generator._parameters_to_text(params)
        expected = "V60, 18.0g coffee, 300g water, medium grind, 4:00 brew time, 30s bloom time"
        assert text == expected

        # Test espresso parameters
        espresso_params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=36.0,
            grind_size=GrindSize.FINE,
            brew_time=30.0,  # 0:30
            brew_method=BrewMethod.ESPRESSO,
            pressure=9.0,
        )

        espresso_text = generator._parameters_to_text(espresso_params)
        expected_espresso = "Espresso, 18.0g coffee, 36g water, fine grind, 0:30 brew time, 9.0 bar pressure"
        assert espresso_text == expected_espresso

    def test_generate_parameter_candidates(self):
        """Test generation of parameter candidates."""
        generator = ExperimentGenerator()

        # Test candidate generation
        candidates = generator._generate_parameter_candidates(
            num_candidates=10, brew_method=BrewMethod.POUR_OVER
        )

        assert isinstance(candidates, list)
        assert len(candidates) <= 10  # May be fewer due to invalid combinations

        for candidate in candidates:
            assert isinstance(candidate, BrewingParameters)
            assert candidate.brew_method == BrewMethod.POUR_OVER

    def test_generate_parameter_candidates_espresso(self):
        """Test candidate generation for espresso method."""
        generator = ExperimentGenerator()

        candidates = generator._generate_parameter_candidates(
            num_candidates=5, brew_method=BrewMethod.ESPRESSO
        )

        assert isinstance(candidates, list)

        for candidate in candidates:
            assert isinstance(candidate, BrewingParameters)
            assert candidate.brew_method == BrewMethod.ESPRESSO
            # Espresso should have pressure parameter
            assert candidate.pressure is not None

    def test_score_candidates_by_uncertainty_no_estimator(self):
        """Test uncertainty scoring when no estimator is available."""
        generator = ExperimentGenerator()

        candidates = [
            BrewingParameters(
                water_temp=90.0,
                coffee_dose=18.0,
                water_amount=300.0,
                grind_size=GrindSize.MEDIUM,
                brew_time=240.0,
                brew_method=BrewMethod.POUR_OVER,
            )
        ]

        mock_tokenizer = MagicMock()

        # Should handle gracefully when no uncertainty estimator
        result = generator._score_candidates_by_uncertainty(candidates, mock_tokenizer)
        assert isinstance(result, list)

    def test_load_experiments_into_kdtree(self):
        """Test loading experiments into k-d tree explorer."""
        mock_kdtree = MagicMock(spec=SimpleKDTreeExplorer)
        generator = ExperimentGenerator(kdtree_explorer=mock_kdtree)

        experiments = [
            {
                "water_temp": 90.0,
                "coffee_dose": 18.0,
                "water_amount": 300.0,
                "grind_size": "medium",
                "brew_time": 240.0,
                "brew_method": "pour_over",
            },
            {
                "water_temp": 95.0,
                "coffee_dose": 20.0,
                "water_amount": 350.0,
                "grind_size": "fine",
                "brew_time": 180.0,
                "brew_method": "espresso",
            },
        ]

        generator._load_experiments_into_kdtree(experiments)

        # Verify that add_experiment was called for each valid experiment
        assert mock_kdtree.add_experiment.call_count == 2

    def test_get_exploration_stats(self):
        """Test getting exploration statistics."""
        generator = ExperimentGenerator()

        # Test without k-d tree explorer
        stats = generator.get_exploration_stats()
        assert stats["kdtree_configured"] is False
        assert stats["total_experiments"] == 0

        # Test with k-d tree explorer
        mock_kdtree = MagicMock(spec=SimpleKDTreeExplorer)
        mock_kdtree.get_stats.return_value = {
            "total_experiments": 5,
            "avg_distance": 0.3,
            "coverage_estimate": 0.7,
        }

        generator.set_kdtree_explorer(mock_kdtree)
        stats = generator.get_exploration_stats()
        assert stats["kdtree_configured"] is True
        assert stats["total_experiments"] == 5
        assert stats["avg_distance"] == 0.3
        assert stats["coverage_estimate"] == 0.7


class TestExperimentGeneratorUncertaintyIntegration:
    """Test uncertainty estimation integration."""

    def test_uncertainty_integration_workflow(self):
        """Test complete uncertainty-based experiment generation workflow."""
        # Create mock uncertainty estimator
        mock_estimator = MagicMock(spec=UncertaintyEstimator)
        mock_estimator.estimate_uncertainty.return_value = {
            "total_uncertainty": torch.tensor([1.5])
        }

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        generator = ExperimentGenerator(uncertainty_estimator=mock_estimator)

        # Test the complete workflow
        result = generator.generate_uncertainty_based_experiments(
            num_experiments=1,
            tokenizer=mock_tokenizer,
            brew_method=BrewMethod.POUR_OVER,
            uncertainty_threshold=1.0,
        )

        assert isinstance(result, list)
        # Verify tokenizer was called
        assert mock_tokenizer.called
        # Verify uncertainty estimator was called
        assert mock_estimator.estimate_uncertainty.called


class TestExperimentGeneratorKDTreeIntegration:
    """Test k-d tree exploration integration."""

    def test_kdtree_exploration_workflow(self):
        """Test complete k-d tree exploration workflow."""
        # Create mock k-d tree explorer
        mock_kdtree = MagicMock(spec=SimpleKDTreeExplorer)
        mock_kdtree.suggest_experiments.return_value = [
            BrewingParameters(
                water_temp=92.0,
                coffee_dose=17.0,
                water_amount=280.0,
                grind_size=GrindSize.MEDIUM_FINE,
                brew_time=220.0,
                brew_method=BrewMethod.POUR_OVER,
            )
        ]
        mock_kdtree.calculate_sparsity.return_value = 0.9

        generator = ExperimentGenerator(kdtree_explorer=mock_kdtree)

        # Test with existing experiments
        existing_experiments = [
            {
                "water_temp": 90.0,
                "coffee_dose": 18.0,
                "water_amount": 300.0,
                "grind_size": "medium",
                "brew_time": 240.0,
                "brew_method": "pour_over",
            }
        ]

        result = generator.generate_exploration_based_experiments(
            num_experiments=2,
            brew_method=BrewMethod.POUR_OVER,
            exploration_strategy="sparsity_guided",
            existing_experiments=existing_experiments,
        )

        assert isinstance(result, list)
        # Verify experiments were loaded into k-d tree
        assert mock_kdtree.add_experiment.called

    def test_sparsity_guided_vs_random_strategies(self):
        """Test different exploration strategies."""
        mock_kdtree = MagicMock(spec=SimpleKDTreeExplorer)
        mock_kdtree.suggest_experiments.return_value = [
            BrewingParameters(
                water_temp=88.0,
                coffee_dose=16.0,
                water_amount=260.0,
                grind_size=GrindSize.COARSE,
                brew_time=300.0,
                brew_method=BrewMethod.POUR_OVER,
            )
        ]
        mock_kdtree.calculate_sparsity.return_value = 0.7

        generator = ExperimentGenerator(kdtree_explorer=mock_kdtree)

        # Test sparsity-guided strategy
        sparsity_result = generator.generate_exploration_based_experiments(
            num_experiments=1, exploration_strategy="sparsity_guided"
        )

        # Test random strategy (falls back to basic k-d tree)
        random_result = generator.generate_exploration_based_experiments(
            num_experiments=1, exploration_strategy="random"
        )

        assert isinstance(sparsity_result, list)
        assert isinstance(random_result, list)
        # Random strategy should call suggest_experiments
        assert mock_kdtree.suggest_experiments.called

    def test_brewing_method_filtering(self):
        """Test that exploration respects brewing method constraints."""
        mock_kdtree = MagicMock(spec=SimpleKDTreeExplorer)
        mock_kdtree.suggest_experiments.return_value = [
            BrewingParameters(
                water_temp=93.0,
                coffee_dose=18.0,
                water_amount=36.0,
                grind_size=GrindSize.FINE,
                brew_time=30.0,
                brew_method=BrewMethod.POUR_OVER,  # Will be updated to target method
                pressure=9.0,
            )
        ]

        generator = ExperimentGenerator(kdtree_explorer=mock_kdtree)

        # Test espresso method
        result = generator.generate_exploration_based_experiments(
            num_experiments=1,
            brew_method=BrewMethod.ESPRESSO,
            exploration_strategy="random",
        )

        assert len(result) == 1
        assert result[0].brew_method == BrewMethod.ESPRESSO
        assert result[0].pressure is not None  # Should have pressure for espresso


if __name__ == "__main__":
    pytest.main([__file__])
