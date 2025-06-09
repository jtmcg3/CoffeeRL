#!/usr/bin/env python3
"""Tests for simple k-d tree parameter space explorer."""

import pytest

from src.kdtree_explorer import ExperimentPoint, SimpleKDTreeExplorer
from src.parameter_space import BrewingParameters, BrewMethod, GrindSize


class TestExperimentPoint:
    """Test ExperimentPoint dataclass."""

    def test_creation(self):
        """Test creating an experiment point."""
        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )

        point = ExperimentPoint(parameters=params)
        assert point.parameters == params
        assert point.results is None
        assert point.experiment_id is None

    def test_creation_with_results(self):
        """Test creating an experiment point with results."""
        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )

        results = {"yield": 20.5, "rating": 4.2}
        point = ExperimentPoint(parameters=params, results=results)

        assert point.parameters == params
        assert point.results == results


class TestSimpleKDTreeExplorer:
    """Test SimpleKDTreeExplorer class."""

    def test_initialization(self):
        """Test explorer initialization."""
        explorer = SimpleKDTreeExplorer()

        assert len(explorer.points) == 0
        assert explorer.kdtree is None
        assert explorer.parameter_ranges is not None

    def test_add_single_experiment(self):
        """Test adding a single experiment."""
        explorer = SimpleKDTreeExplorer()

        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )

        index = explorer.add_experiment(params)

        assert index == 0
        assert len(explorer.points) == 1
        assert explorer.kdtree is not None
        assert explorer.points[0].parameters == params

    def test_add_experiment_with_results(self):
        """Test adding experiment with results."""
        explorer = SimpleKDTreeExplorer()

        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )

        results = {"yield": 20.5, "rating": 4.2}
        index = explorer.add_experiment(params, results)

        assert index == 0
        assert explorer.points[0].results == results

    def test_add_experiments_from_dicts(self):
        """Test adding multiple experiments from dictionaries."""
        explorer = SimpleKDTreeExplorer()

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
                "water_amount": 320.0,
                "grind_size": "fine",
                "brew_time": 200.0,
                "brew_method": "espresso",
            },
        ]

        indices = explorer.add_experiments(experiments)

        assert indices == [0, 1]
        assert len(explorer.points) == 2
        assert explorer.points[0].parameters.water_temp == 90.0
        assert explorer.points[1].parameters.water_temp == 95.0

    def test_find_nearest_empty_explorer(self):
        """Test finding nearest neighbors on empty explorer."""
        explorer = SimpleKDTreeExplorer()

        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )

        nearest = explorer.find_nearest(params, k=3)
        assert nearest == []

    def test_find_nearest_single_point(self):
        """Test finding nearest neighbor with single point."""
        explorer = SimpleKDTreeExplorer()

        # Add one experiment
        params1 = BrewingParameters(
            water_temp=90.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )
        explorer.add_experiment(params1)

        # Query with similar parameters
        query_params = BrewingParameters(
            water_temp=91.0,  # Close to 90.0
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )

        nearest = explorer.find_nearest(query_params, k=1)

        assert len(nearest) == 1
        index, distance = nearest[0]
        assert index == 0
        assert distance >= 0.0

    def test_find_nearest_multiple_points(self):
        """Test finding nearest neighbors with multiple points."""
        explorer = SimpleKDTreeExplorer()

        # Add several experiments with different temperatures
        for temp in [85.0, 90.0, 95.0, 100.0]:
            params = BrewingParameters(
                water_temp=temp,
                coffee_dose=18.0,
                water_amount=300.0,
                grind_size=GrindSize.MEDIUM,
                brew_time=240.0,
                brew_method=BrewMethod.POUR_OVER,
            )
            explorer.add_experiment(params)

        # Query with temperature close to 90.0
        query_params = BrewingParameters(
            water_temp=91.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )

        nearest = explorer.find_nearest(query_params, k=2)

        assert len(nearest) == 2
        # First result should be closest (90.0 temp experiment)
        closest_index, closest_distance = nearest[0]
        assert explorer.points[closest_index].parameters.water_temp == 90.0

    def test_calculate_sparsity_empty_explorer(self):
        """Test sparsity calculation on empty explorer."""
        explorer = SimpleKDTreeExplorer()

        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )

        sparsity = explorer.calculate_sparsity(params)
        assert sparsity == 1.0  # Maximum sparsity for empty space

    def test_calculate_sparsity_with_points(self):
        """Test sparsity calculation with existing points."""
        explorer = SimpleKDTreeExplorer()

        # Add a point
        params1 = BrewingParameters(
            water_temp=90.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )
        explorer.add_experiment(params1)

        # Test sparsity of a nearby point (should be low sparsity)
        nearby_params = BrewingParameters(
            water_temp=90.5,  # Very close
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )

        nearby_sparsity = explorer.calculate_sparsity(nearby_params)

        # Test sparsity of a distant point (should be high sparsity)
        distant_params = BrewingParameters(
            water_temp=70.0,  # Very different
            coffee_dose=25.0,
            water_amount=500.0,
            grind_size=GrindSize.COARSE,
            brew_time=600.0,
            brew_method=BrewMethod.FRENCH_PRESS,
        )

        distant_sparsity = explorer.calculate_sparsity(distant_params)

        # Distant point should have higher sparsity than nearby point
        assert distant_sparsity > nearby_sparsity
        assert 0.0 <= nearby_sparsity <= 1.0
        assert 0.0 <= distant_sparsity <= 1.0

    def test_suggest_experiments_empty_explorer(self):
        """Test experiment suggestions on empty explorer."""
        explorer = SimpleKDTreeExplorer()

        suggestions = explorer.suggest_experiments(num_suggestions=3)

        assert len(suggestions) == 3
        for suggestion in suggestions:
            assert isinstance(suggestion, BrewingParameters)

    def test_suggest_experiments_with_existing_points(self):
        """Test experiment suggestions with existing points."""
        explorer = SimpleKDTreeExplorer()

        # Add some experiments
        for temp in [85.0, 90.0, 95.0]:
            params = BrewingParameters(
                water_temp=temp,
                coffee_dose=18.0,
                water_amount=300.0,
                grind_size=GrindSize.MEDIUM,
                brew_time=240.0,
                brew_method=BrewMethod.POUR_OVER,
            )
            explorer.add_experiment(params)

        suggestions = explorer.suggest_experiments(num_suggestions=2)

        assert len(suggestions) == 2
        for suggestion in suggestions:
            assert isinstance(suggestion, BrewingParameters)

    def test_get_stats_empty_explorer(self):
        """Test getting stats from empty explorer."""
        explorer = SimpleKDTreeExplorer()

        stats = explorer.get_stats()

        assert stats["total_experiments"] == 0
        assert stats["avg_distance"] == 0.0
        assert stats["coverage_estimate"] == 0.0

    def test_get_stats_with_experiments(self):
        """Test getting stats with experiments."""
        explorer = SimpleKDTreeExplorer()

        # Add several experiments
        for i, temp in enumerate([85.0, 90.0, 95.0, 100.0]):
            params = BrewingParameters(
                water_temp=temp,
                coffee_dose=18.0 + i,
                water_amount=300.0,
                grind_size=GrindSize.MEDIUM,
                brew_time=240.0,
                brew_method=BrewMethod.POUR_OVER,
            )
            explorer.add_experiment(params)

        stats = explorer.get_stats()

        assert stats["total_experiments"] == 4
        assert stats["avg_distance"] >= 0.0
        assert 0.0 <= stats["coverage_estimate"] <= 1.0

    def test_rebuild_tree_internal(self):
        """Test that k-d tree is properly rebuilt."""
        explorer = SimpleKDTreeExplorer()

        # Initially no tree
        assert explorer.kdtree is None

        # Add first experiment - tree should be built
        params1 = BrewingParameters(
            water_temp=90.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )
        explorer.add_experiment(params1)
        assert explorer.kdtree is not None

        # Add second experiment - tree should be rebuilt
        params2 = BrewingParameters(
            water_temp=95.0,
            coffee_dose=20.0,
            water_amount=320.0,
            grind_size=GrindSize.FINE,
            brew_time=200.0,
            brew_method=BrewMethod.ESPRESSO,
        )
        explorer.add_experiment(params2)
        assert explorer.kdtree is not None

        # Should be able to query the tree
        nearest = explorer.find_nearest(params1, k=1)
        assert len(nearest) == 1


if __name__ == "__main__":
    pytest.main([__file__])
