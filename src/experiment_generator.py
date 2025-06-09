#!/usr/bin/env python3
"""Experiment generation engine for CoffeeRL.

This module provides the ExperimentGenerator class that combines uncertainty-based
and exploration-based strategies to generate intelligent brewing experiments.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .kdtree_explorer import SimpleKDTreeExplorer
from .parameter_space import (
    BrewingParameters,
    BrewMethod,
    GrindSize,
    ParameterRanges,
    denormalize_parameters,
)
from .parameter_space_explorer import ParameterSpaceExplorer
from .uncertainty_estimator import UncertaintyEstimator


class ExperimentGenerator:
    """Generates brewing experiments using uncertainty and exploration strategies."""

    def __init__(
        self,
        uncertainty_estimator: Optional[UncertaintyEstimator] = None,
        parameter_explorer: Optional[ParameterSpaceExplorer] = None,
        kdtree_explorer: Optional[SimpleKDTreeExplorer] = None,
    ):
        """Initialize the experiment generator.

        Args:
            uncertainty_estimator: Module for uncertainty-based experiment generation
            parameter_explorer: High-level parameter space explorer
            kdtree_explorer: K-d tree based explorer for sparse regions
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.parameter_explorer = parameter_explorer
        self.kdtree_explorer = kdtree_explorer

    def generate_uncertainty_based_experiments(
        self,
        num_experiments: int = 5,
        uncertainty_threshold: Optional[float] = None,
        brew_method: BrewMethod = BrewMethod.POUR_OVER,
        tokenizer=None,
        num_candidates: int = 50,
    ) -> List[BrewingParameters]:
        """Generate experiments targeting high-uncertainty regions.

        Args:
            num_experiments: Number of experiments to generate
            uncertainty_threshold: Minimum uncertainty score to target
            brew_method: Target brewing method
            tokenizer: Tokenizer for text conversion (required for uncertainty estimation)
            num_candidates: Number of candidate parameters to evaluate

        Returns:
            List of brewing parameters for high-uncertainty experiments
        """
        if self.uncertainty_estimator is None:
            return []

        if tokenizer is None:
            # Cannot perform uncertainty estimation without tokenizer
            return []

        # Generate candidate brewing parameters
        candidates = self._generate_parameter_candidates(num_candidates, brew_method)

        # Score candidates by uncertainty
        scored_candidates = self._score_candidates_by_uncertainty(
            candidates, tokenizer, uncertainty_threshold
        )

        # Return top candidates
        top_candidates = scored_candidates[:num_experiments]
        return [params for params, _ in top_candidates]

    def generate_exploration_based_experiments(
        self,
        num_experiments: int = 5,
        brew_method: BrewMethod = BrewMethod.POUR_OVER,
        exploration_strategy: str = "sparsity_guided",
        existing_experiments: Optional[List[Dict]] = None,
    ) -> List[BrewingParameters]:
        """Generate experiments targeting unexplored parameter regions.

        Args:
            num_experiments: Number of experiments to generate
            brew_method: Target brewing method
            exploration_strategy: Strategy to use ("sparsity_guided" or "random")
            existing_experiments: Optional list of existing experiments to load into k-d tree

        Returns:
            List of brewing parameters for exploration experiments
        """
        if self.kdtree_explorer is None:
            return []

        # Load existing experiments if provided
        if existing_experiments:
            self._load_experiments_into_kdtree(existing_experiments)

        if exploration_strategy == "sparsity_guided":
            return self._generate_sparsity_guided_experiments(
                num_experiments, brew_method
            )
        else:
            # Fall back to basic k-d tree suggestion
            return self._generate_basic_kdtree_experiments(num_experiments, brew_method)

    def generate_experiments(
        self,
        num_experiments: int = 10,
        uncertainty_weight: float = 0.5,
        exploration_weight: float = 0.5,
        brew_method: BrewMethod = BrewMethod.POUR_OVER,
        tokenizer=None,
        uncertainty_threshold: Optional[float] = None,
    ) -> List[BrewingParameters]:
        """Generate experiments combining uncertainty and exploration strategies.

        Args:
            num_experiments: Total number of experiments to generate
            uncertainty_weight: Weight for uncertainty-based experiments (0-1)
            exploration_weight: Weight for exploration-based experiments (0-1)
            brew_method: Target brewing method
            tokenizer: Tokenizer for uncertainty-based generation
            uncertainty_threshold: Minimum uncertainty threshold

        Returns:
            List of brewing parameters combining both strategies
        """
        uncertainty_count = int(num_experiments * uncertainty_weight)
        exploration_count = num_experiments - uncertainty_count

        uncertainty_experiments = self.generate_uncertainty_based_experiments(
            uncertainty_count,
            uncertainty_threshold=uncertainty_threshold,
            brew_method=brew_method,
            tokenizer=tokenizer,
        )
        exploration_experiments = self.generate_exploration_based_experiments(
            exploration_count, brew_method
        )

        return uncertainty_experiments + exploration_experiments

    def _score_experiment_candidates(
        self, candidates: List[BrewingParameters]
    ) -> List[Tuple[BrewingParameters, float]]:
        """Score experiment candidates based on information gain potential.

        Args:
            candidates: List of candidate brewing parameters

        Returns:
            List of (parameters, score) tuples sorted by score descending
        """
        # Placeholder implementation
        # TODO: Implement scoring system balancing information gain and feasibility
        scored_candidates = [(params, 0.0) for params in candidates]
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    def set_uncertainty_estimator(self, estimator: UncertaintyEstimator) -> None:
        """Set the uncertainty estimation module.

        Args:
            estimator: Uncertainty estimator instance
        """
        self.uncertainty_estimator = estimator

    def set_parameter_explorer(self, explorer: ParameterSpaceExplorer) -> None:
        """Set the parameter space explorer.

        Args:
            explorer: Parameter space explorer instance
        """
        self.parameter_explorer = explorer

    def set_kdtree_explorer(self, explorer: SimpleKDTreeExplorer) -> None:
        """Set the k-d tree explorer.

        Args:
            explorer: K-d tree explorer instance
        """
        self.kdtree_explorer = explorer

    def is_configured(self) -> bool:
        """Check if the generator has all required components configured.

        Returns:
            True if all required components are set, False otherwise
        """
        return (
            self.uncertainty_estimator is not None
            or self.parameter_explorer is not None
            or self.kdtree_explorer is not None
        )

    def get_configuration_status(self) -> Dict[str, bool]:
        """Get the configuration status of all components.

        Returns:
            Dictionary showing which components are configured
        """
        return {
            "uncertainty_estimator": self.uncertainty_estimator is not None,
            "parameter_explorer": self.parameter_explorer is not None,
            "kdtree_explorer": self.kdtree_explorer is not None,
        }

    def _generate_parameter_candidates(
        self, num_candidates: int, brew_method: BrewMethod
    ) -> List[BrewingParameters]:
        """Generate candidate brewing parameters for uncertainty evaluation.

        Args:
            num_candidates: Number of candidates to generate
            brew_method: Target brewing method

        Returns:
            List of candidate brewing parameters
        """
        candidates = []
        parameter_ranges = ParameterRanges()

        for _ in range(num_candidates):
            # Generate random normalized parameters
            random_vector = np.random.random(6)  # 6 core dimensions

            # Add method-specific parameters if needed
            if brew_method in [BrewMethod.ESPRESSO, BrewMethod.AEROPRESS]:
                pressure_norm = np.random.random()
                random_vector = np.append(random_vector, pressure_norm)

            if brew_method == BrewMethod.POUR_OVER:
                bloom_norm = np.random.random()
                random_vector = np.append(random_vector, bloom_norm)

            # Convert to brewing parameters
            try:
                params = denormalize_parameters(
                    random_vector, parameter_ranges, brew_method
                )
                candidates.append(params)
            except Exception:
                # Skip invalid parameter combinations
                continue

        return candidates

    def _score_candidates_by_uncertainty(
        self,
        candidates: List[BrewingParameters],
        tokenizer,
        uncertainty_threshold: Optional[float] = None,
    ) -> List[Tuple[BrewingParameters, float]]:
        """Score brewing parameter candidates by uncertainty.

        Args:
            candidates: List of candidate brewing parameters
            tokenizer: Tokenizer for text conversion
            uncertainty_threshold: Minimum uncertainty threshold

        Returns:
            List of (parameters, uncertainty_score) tuples sorted by uncertainty descending
        """
        scored_candidates = []

        for params in candidates:
            try:
                # Convert parameters to text description
                text_description = self._parameters_to_text(params)

                # Tokenize the text
                inputs = tokenizer(
                    text_description,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )

                # Get uncertainty metrics
                uncertainty_metrics = self.uncertainty_estimator.estimate_uncertainty(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )

                # Extract total uncertainty score
                uncertainty_score = (
                    uncertainty_metrics["total_uncertainty"].mean().item()
                )

                # Apply threshold if specified
                if (
                    uncertainty_threshold is None
                    or uncertainty_score >= uncertainty_threshold
                ):
                    scored_candidates.append((params, uncertainty_score))

            except Exception:
                # Skip candidates that cause errors
                continue

        # Sort by uncertainty score (highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates

    def _parameters_to_text(self, params: BrewingParameters) -> str:
        """Convert brewing parameters to text description for uncertainty analysis.

        Args:
            params: Brewing parameters to convert

        Returns:
            Text description suitable for model input
        """
        # Format grind size
        grind_map = {
            GrindSize.VERY_FINE: "very fine",
            GrindSize.FINE: "fine",
            GrindSize.MEDIUM_FINE: "medium-fine",
            GrindSize.MEDIUM: "medium",
            GrindSize.MEDIUM_COARSE: "medium-coarse",
            GrindSize.COARSE: "coarse",
        }
        grind_text = grind_map.get(params.grind_size, "medium")

        # Format brew method
        method_map = {
            BrewMethod.POUR_OVER: "V60",
            BrewMethod.FRENCH_PRESS: "French Press",
            BrewMethod.ESPRESSO: "Espresso",
            BrewMethod.AEROPRESS: "AeroPress",
            BrewMethod.COLD_BREW: "Cold Brew",
        }
        method_text = method_map.get(params.brew_method, "V60")

        # Format brew time
        minutes = int(params.brew_time // 60)
        seconds = int(params.brew_time % 60)
        time_text = f"{minutes}:{seconds:02d}"

        # Build description
        description = f"{method_text}, {params.coffee_dose:.1f}g coffee, {params.water_amount:.0f}g water, {grind_text} grind, {time_text} brew time"

        # Add method-specific parameters
        if params.pressure is not None:
            description += f", {params.pressure:.1f} bar pressure"

        if params.bloom_time is not None:
            bloom_minutes = int(params.bloom_time // 60)
            bloom_seconds = int(params.bloom_time % 60)
            if bloom_minutes > 0:
                description += f", {bloom_minutes}:{bloom_seconds:02d} bloom time"
            else:
                description += f", {bloom_seconds}s bloom time"

        return description

    def _load_experiments_into_kdtree(self, experiments: List[Dict]) -> None:
        """Load existing experiments into the k-d tree explorer.

        Args:
            experiments: List of experiment dictionaries to load
        """
        if self.kdtree_explorer is None:
            return

        # Convert dictionaries to BrewingParameters and add to k-d tree
        for exp_dict in experiments:
            try:
                params = BrewingParameters.from_dict(exp_dict)
                self.kdtree_explorer.add_experiment(params)
            except Exception:
                # Skip invalid experiment data
                continue

    def _generate_sparsity_guided_experiments(
        self, num_experiments: int, brew_method: BrewMethod
    ) -> List[BrewingParameters]:
        """Generate experiments using sparsity-guided exploration.

        Args:
            num_experiments: Number of experiments to generate
            brew_method: Target brewing method

        Returns:
            List of brewing parameters targeting sparse regions
        """
        if self.kdtree_explorer is None:
            return []

        # Generate more candidates than needed and select the sparsest
        num_candidates = max(50, num_experiments * 10)
        candidates = self._generate_parameter_candidates(num_candidates, brew_method)

        # Score candidates by sparsity
        scored_candidates = []
        for params in candidates:
            try:
                sparsity_score = self.kdtree_explorer.calculate_sparsity(params)
                scored_candidates.append((params, sparsity_score))
            except Exception:
                # Skip candidates that cause errors
                continue

        # Sort by sparsity (highest first) and return top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_candidates[:num_experiments]
        return [params for params, _ in top_candidates]

    def _generate_basic_kdtree_experiments(
        self, num_experiments: int, brew_method: BrewMethod
    ) -> List[BrewingParameters]:
        """Generate experiments using basic k-d tree suggestion.

        Args:
            num_experiments: Number of experiments to generate
            brew_method: Target brewing method

        Returns:
            List of brewing parameters from k-d tree suggestions
        """
        if self.kdtree_explorer is None:
            return []

        # Use the k-d tree's built-in suggestion method
        suggestions = self.kdtree_explorer.suggest_experiments(num_experiments)

        # Filter suggestions to match the target brewing method
        filtered_suggestions = []
        for suggestion in suggestions:
            # Update the brewing method to match the target
            updated_suggestion = BrewingParameters(
                water_temp=suggestion.water_temp,
                coffee_dose=suggestion.coffee_dose,
                water_amount=suggestion.water_amount,
                grind_size=suggestion.grind_size,
                brew_time=suggestion.brew_time,
                brew_method=brew_method,
                pressure=(
                    suggestion.pressure
                    if brew_method in [BrewMethod.ESPRESSO, BrewMethod.AEROPRESS]
                    else None
                ),
                bloom_time=(
                    suggestion.bloom_time
                    if brew_method == BrewMethod.POUR_OVER
                    else None
                ),
            )
            filtered_suggestions.append(updated_suggestion)

        return filtered_suggestions

    def get_exploration_stats(self) -> Dict:
        """Get statistics about the current exploration state.

        Returns:
            Dictionary with exploration statistics
        """
        if self.kdtree_explorer is None:
            return {
                "total_experiments": 0,
                "avg_distance": 0.0,
                "coverage_estimate": 0.0,
                "kdtree_configured": False,
            }

        stats = self.kdtree_explorer.get_stats()
        stats["kdtree_configured"] = True
        return stats
