#!/usr/bin/env python3
"""Simple k-d tree parameter space exploration for coffee brewing.

Minimal implementation focused on core functionality:
- Store experiment points in k-d tree
- Find nearest neighbors
- Calculate sparsity scores
- Suggest unexplored regions
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from .parameter_space import (
    BrewingParameters,
    BrewMethod,
    ParameterRanges,
    denormalize_parameters,
    normalize_parameters,
)


@dataclass
class ExperimentPoint:
    """Single experiment point with parameters and results."""

    parameters: BrewingParameters
    results: Optional[Dict] = None
    experiment_id: Optional[str] = None


class SimpleKDTreeExplorer:
    """Simple k-d tree explorer for parameter space."""

    def __init__(self):
        """Initialize empty explorer."""
        self.points: List[ExperimentPoint] = []
        self.kdtree: Optional[KDTree] = None
        self.parameter_ranges = ParameterRanges()

    def add_experiment(
        self, parameters: BrewingParameters, results: Optional[Dict] = None
    ) -> int:
        """Add single experiment.

        Args:
            parameters: Brewing parameters
            results: Optional experiment results

        Returns:
            Index of added experiment
        """
        point = ExperimentPoint(parameters=parameters, results=results)
        self.points.append(point)
        self._rebuild_tree()
        return len(self.points) - 1

    def add_experiments(self, experiments: List[Dict]) -> List[int]:
        """Add multiple experiments from dictionaries.

        Args:
            experiments: List of experiment dictionaries

        Returns:
            List of indices for added experiments
        """
        indices = []
        for exp_dict in experiments:
            params = BrewingParameters.from_dict(exp_dict)
            index = self.add_experiment(params)
            indices.append(index)
        return indices

    def find_nearest(
        self, parameters: BrewingParameters, k: int = 3
    ) -> List[Tuple[int, float]]:
        """Find k nearest experiments to given parameters.

        Args:
            parameters: Query parameters
            k: Number of neighbors to find

        Returns:
            List of (index, distance) tuples
        """
        if not self.points or self.kdtree is None:
            return []

        query_vector = normalize_parameters(parameters, self.parameter_ranges)
        k = min(k, len(self.points))

        distances, indices = self.kdtree.query(query_vector, k=k)

        # Handle single result case
        if k == 1:
            return [(int(indices), float(distances))]

        return [(int(idx), float(dist)) for idx, dist in zip(indices, distances)]

    def calculate_sparsity(self, parameters: BrewingParameters) -> float:
        """Calculate sparsity score for parameters (0=dense, 1=sparse).

        Args:
            parameters: Parameters to evaluate

        Returns:
            Sparsity score between 0 and 1
        """
        if not self.points:
            return 1.0  # Maximum sparsity for empty space

        nearest = self.find_nearest(parameters, k=1)
        if not nearest:
            return 1.0

        distance = nearest[0][1]
        # Convert distance to sparsity score (higher distance = higher sparsity)
        return min(1.0, distance * 5.0)  # Scale factor to get reasonable scores

    def suggest_experiments(self, num_suggestions: int = 5) -> List[BrewingParameters]:
        """Suggest experiments in sparse regions.

        Args:
            num_suggestions: Number of experiments to suggest

        Returns:
            List of suggested brewing parameters
        """
        suggestions = []

        # Determine the expected vector dimension from existing points
        expected_dim = 6  # Default to core dimensions
        if self.points:
            # Get dimension from first point
            first_vector = normalize_parameters(
                self.points[0].parameters, self.parameter_ranges
            )
            expected_dim = len(first_vector)

        # Generate random candidates and pick the sparsest ones
        candidates = []
        for _ in range(num_suggestions * 10):  # Generate more candidates than needed
            # Random parameters in normalized space with correct dimensions
            random_vector = np.random.random(expected_dim)
            params = denormalize_parameters(
                random_vector, self.parameter_ranges, BrewMethod.POUR_OVER
            )
            sparsity = self.calculate_sparsity(params)
            candidates.append((params, sparsity))

        # Sort by sparsity (highest first) and take top suggestions
        candidates.sort(key=lambda x: x[1], reverse=True)
        suggestions = [params for params, _ in candidates[:num_suggestions]]

        return suggestions

    def get_stats(self) -> Dict:
        """Get basic exploration statistics.

        Returns:
            Dictionary with exploration stats
        """
        if not self.points:
            return {
                "total_experiments": 0,
                "avg_distance": 0.0,
                "coverage_estimate": 0.0,
            }

        # Calculate average distance between points
        total_distance = 0.0
        count = 0

        for i, point in enumerate(self.points):
            nearest = self.find_nearest(point.parameters, k=2)  # k=2 to skip self
            if len(nearest) > 1:
                total_distance += nearest[1][
                    1
                ]  # Distance to second nearest (first is self)
                count += 1

        avg_distance = total_distance / count if count > 0 else 0.0
        coverage_estimate = max(
            0.0, 1.0 - avg_distance * 2.0
        )  # Rough coverage estimate

        return {
            "total_experiments": len(self.points),
            "avg_distance": avg_distance,
            "coverage_estimate": coverage_estimate,
        }

    def _rebuild_tree(self) -> None:
        """Rebuild k-d tree from current points."""
        if not self.points:
            self.kdtree = None
            return

        vectors = []
        for point in self.points:
            vector = normalize_parameters(point.parameters, self.parameter_ranges)
            vectors.append(vector)

        self.kdtree = KDTree(np.array(vectors))
