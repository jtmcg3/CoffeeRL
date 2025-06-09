#!/usr/bin/env python3
"""Main parameter space exploration interface for CoffeeRL.

This module provides a high-level interface for parameter space exploration,
integrating k-d trees, sparsity analysis, and priority queuing for intelligent
experiment suggestion.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from .kdtree_explorer import SimpleKDTreeExplorer
from .parameter_space import BrewingParameters, BrewMethod, ParameterRanges


class ParameterSpaceExplorer:
    """High-level interface for parameter space exploration."""

    def __init__(
        self,
        parameter_ranges: Optional[ParameterRanges] = None,
        save_path: Optional[str] = None,
    ):
        """Initialize the parameter space explorer.

        Args:
            parameter_ranges: Custom parameter ranges (uses defaults if None)
            save_path: Path to save/load exploration state
        """
        self.parameter_ranges = parameter_ranges or ParameterRanges()
        self.kdtree_explorer = SimpleKDTreeExplorer()
        self.save_path = save_path or "./data/exploration_state.json"
        self.exploration_history: List[Dict] = []

    def load_existing_experiments(self, experiments: List[Dict]) -> int:
        """Load existing experiments into the parameter space.

        Args:
            experiments: List of experiment dictionaries

        Returns:
            Number of experiments loaded
        """
        if not experiments:
            return 0

        # Add experiments directly using the kdtree explorer's API
        indices = self.kdtree_explorer.add_experiments(experiments)

        # Track in history
        self.exploration_history.append(
            {
                "timestamp": time.time(),
                "action": "load_experiments",
                "count": len(experiments),
                "total_experiments": len(self.kdtree_explorer.points),
            }
        )

        return len(indices)

    def suggest_experiments(
        self,
        num_suggestions: int = 5,
        brew_method: BrewMethod = BrewMethod.POUR_OVER,
        constraints: Optional[Dict] = None,
        return_detailed: bool = False,
    ) -> List[BrewingParameters]:
        """Suggest next experiments based on parameter space exploration.

        Args:
            num_suggestions: Number of experiments to suggest
            brew_method: Target brewing method (currently not used in simple implementation)
            constraints: Optional constraints to apply (currently not used)
            return_detailed: If True, return detailed candidates with priority info (currently not used)

        Returns:
            List of suggested experiments
        """
        suggestions = self.kdtree_explorer.suggest_experiments(
            num_suggestions=num_suggestions
        )

        # Track in history
        self.exploration_history.append(
            {
                "timestamp": time.time(),
                "action": "suggest_experiments",
                "num_suggestions": num_suggestions,
                "brew_method": brew_method.value,
                "suggestions_generated": len(suggestions),
            }
        )

        return suggestions

    def add_experiment_result(
        self,
        parameters: Union[BrewingParameters, Dict],
        results: Optional[Dict] = None,
        experiment_id: Optional[str] = None,
    ) -> int:
        """Add a completed experiment to the parameter space.

        Args:
            parameters: Brewing parameters used
            results: Experimental results (yield, time, rating, etc.)
            experiment_id: Optional experiment identifier

        Returns:
            Index of the added experiment
        """
        if isinstance(parameters, dict):
            params = BrewingParameters.from_dict(parameters)
        else:
            params = parameters

        index = self.kdtree_explorer.add_experiment(parameters=params, results=results)

        # Track in history
        self.exploration_history.append(
            {
                "timestamp": time.time(),
                "action": "add_experiment_result",
                "experiment_id": experiment_id,
                "index": index,
                "total_experiments": len(self.kdtree_explorer.points),
            }
        )

        return index

    def analyze_coverage(self) -> Dict:
        """Analyze the current parameter space coverage.

        Returns:
            Dictionary with coverage analysis
        """
        total_experiments = len(self.kdtree_explorer.points)

        if total_experiments == 0:
            return {
                "total_experiments": 0,
                "exploration_efficiency": 0.0,
                "coverage_quality": "insufficient_data",
                "recommended_next_experiments": 10,
            }

        # Get basic stats from kdtree explorer
        stats = self.kdtree_explorer.get_stats()

        analysis = {
            "total_experiments": total_experiments,
            "exploration_efficiency": self._calculate_exploration_efficiency(),
            "recommended_next_experiments": min(
                10, max(1, int(20 - total_experiments / 5))
            ),
            "coverage_quality": self._assess_coverage_quality(total_experiments),
            "avg_distance": stats.get("avg_distance", 0.0),
            "coverage_estimate": stats.get("coverage_estimate", 0.0),
        }

        # Track in history
        self.exploration_history.append(
            {
                "timestamp": time.time(),
                "action": "analyze_coverage",
                "analysis": analysis,
            }
        )

        return analysis

    def _calculate_exploration_efficiency(self) -> float:
        """Calculate exploration efficiency based on coverage and experiment count.

        Returns:
            Efficiency score between 0 and 1
        """
        total_experiments = len(self.kdtree_explorer.points)

        if total_experiments == 0:
            return 0.0

        # Simple efficiency calculation based on experiment count
        # More experiments generally means better coverage, but with diminishing returns
        efficiency = min(1.0, total_experiments / 50.0)

        # Penalty for too many experiments in small space
        if total_experiments > 100:
            efficiency *= 0.8

        return efficiency

    def _assess_coverage_quality(self, total_experiments: int) -> str:
        """Assess the quality of parameter space coverage.

        Args:
            total_experiments: Number of experiments

        Returns:
            Quality assessment string
        """
        if total_experiments < 5:
            return "insufficient_data"
        elif total_experiments < 20:
            return "sparse"
        elif total_experiments < 50:
            return "moderate"
        elif total_experiments < 100:
            return "good"
        else:
            return "comprehensive"

    def get_exploration_report(self) -> Dict:
        """Generate a comprehensive exploration report.

        Returns:
            Dictionary with exploration report
        """
        analysis = self.analyze_coverage()

        # For now, we'll use a simple approach for unexplored regions
        # In a more complex implementation, this would use the actual sparsity analysis
        unexplored_regions_count = max(0, 10 - len(self.kdtree_explorer.points) // 5)

        report = {
            "timestamp": time.time(),
            "coverage_analysis": analysis,
            "unexplored_regions": unexplored_regions_count,
            "exploration_history": self.exploration_history[-10:],  # Last 10 actions
            "recommendations": self._generate_recommendations(analysis, []),
        }

        return report

    def _generate_recommendations(
        self, analysis: Dict, unexplored_regions: List
    ) -> List[str]:
        """Generate exploration recommendations based on analysis.

        Args:
            analysis: Coverage analysis results
            unexplored_regions: List of unexplored regions

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if analysis["total_experiments"] < 10:
            recommendations.append(
                "Increase experiment count to at least 10 for meaningful analysis"
            )

        if analysis["coverage_quality"] == "sparse":
            recommendations.append(
                "Focus on exploring different parameter combinations"
            )

        if len(unexplored_regions) > 3:
            recommendations.append(
                f"Consider exploring {len(unexplored_regions)} identified sparse regions"
            )

        if analysis["exploration_efficiency"] < 0.5:
            recommendations.append(
                "Improve exploration efficiency by targeting diverse parameter ranges"
            )

        if not recommendations:
            recommendations.append("Current exploration coverage appears adequate")

        return recommendations

    def save_state(self, filepath: Optional[str] = None) -> None:
        """Save the current exploration state to file.

        Args:
            filepath: Optional custom filepath (uses default if None)
        """
        save_path = Path(filepath or self.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare state data
        state_data = {
            "timestamp": time.time(),
            "parameter_ranges": {
                "water_temp_min": self.parameter_ranges.water_temp_min,
                "water_temp_max": self.parameter_ranges.water_temp_max,
                "coffee_dose_min": self.parameter_ranges.coffee_dose_min,
                "coffee_dose_max": self.parameter_ranges.coffee_dose_max,
                "water_amount_min": self.parameter_ranges.water_amount_min,
                "water_amount_max": self.parameter_ranges.water_amount_max,
                "brew_time_min": self.parameter_ranges.brew_time_min,
                "brew_time_max": self.parameter_ranges.brew_time_max,
            },
            "experiments": [
                {
                    "parameters": {
                        "water_temp": exp.parameters.water_temp,
                        "coffee_dose": exp.parameters.coffee_dose,
                        "water_amount": exp.parameters.water_amount,
                        "grind_size": exp.parameters.grind_size.value,
                        "brew_time": exp.parameters.brew_time,
                        "brew_method": exp.parameters.brew_method.value,
                        "pressure": exp.parameters.pressure,
                        "bloom_time": exp.parameters.bloom_time,
                    },
                    "results": exp.results,
                    "experiment_id": exp.experiment_id,
                }
                for exp in self.kdtree_explorer.points
            ],
            "exploration_history": self.exploration_history,
        }

        with open(save_path, "w") as f:
            json.dump(state_data, f, indent=2)

        print(f"Exploration state saved to: {save_path}")

    def load_state(self, filepath: Optional[str] = None) -> bool:
        """Load exploration state from file.

        Args:
            filepath: Optional custom filepath (uses default if None)

        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = Path(filepath or self.save_path)

        if not load_path.exists():
            print(f"State file not found: {load_path}")
            return False

        try:
            with open(load_path, "r") as f:
                state_data = json.load(f)

            # Restore parameter ranges
            ranges_data = state_data.get("parameter_ranges", {})
            self.parameter_ranges = ParameterRanges(**ranges_data)

            # Restore experiments
            experiments_data = state_data.get("experiments", [])
            if experiments_data:
                self.load_existing_experiments(
                    [exp["parameters"] for exp in experiments_data]
                )

            # Restore history
            self.exploration_history = state_data.get("exploration_history", [])

            print(f"Exploration state loaded from: {load_path}")
            print(f"Loaded {len(experiments_data)} experiments")
            return True

        except Exception as e:
            print(f"Failed to load state: {e}")
            return False


def create_explorer_from_data(
    experiments: List[Dict], parameter_ranges: Optional[ParameterRanges] = None
) -> ParameterSpaceExplorer:
    """Create a parameter space explorer from existing experiment data.

    Args:
        experiments: List of experiment dictionaries
        parameter_ranges: Optional custom parameter ranges

    Returns:
        Configured ParameterSpaceExplorer instance
    """
    explorer = ParameterSpaceExplorer(parameter_ranges=parameter_ranges)
    explorer.load_existing_experiments(experiments)
    return explorer


def suggest_next_experiments_simple(
    experiments: List[Dict], num_suggestions: int = 5, brew_method: str = "pour_over"
) -> List[Dict]:
    """Simple utility function to suggest next experiments.

    Args:
        experiments: List of existing experiment dictionaries
        num_suggestions: Number of suggestions to generate
        brew_method: Target brewing method

    Returns:
        List of suggested experiment dictionaries
    """
    explorer = create_explorer_from_data(experiments)

    # Convert string to enum with error handling
    try:
        method_enum = BrewMethod(brew_method.lower())
    except ValueError:
        method_enum = BrewMethod.POUR_OVER  # Default fallback

    suggestions = explorer.suggest_experiments(
        num_suggestions=num_suggestions, brew_method=method_enum
    )

    # Convert back to dictionaries
    suggestion_dicts = []
    for suggestion in suggestions:
        suggestion_dict = {
            "water_temp": suggestion.water_temp,
            "coffee_dose": suggestion.coffee_dose,
            "water_amount": suggestion.water_amount,
            "grind_size": suggestion.grind_size.value,
            "brew_time": suggestion.brew_time,
            "brew_method": suggestion.brew_method.value,
        }

        if suggestion.pressure is not None:
            suggestion_dict["pressure"] = suggestion.pressure
        if suggestion.bloom_time is not None:
            suggestion_dict["bloom_time"] = suggestion.bloom_time

        suggestion_dicts.append(suggestion_dict)

    return suggestion_dicts
