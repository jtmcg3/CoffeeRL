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
    normalize_parameters,
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
            tokenizer: Tokenizer for text conversion (required for uncertainty
                estimation)
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
            existing_experiments: Optional list of existing experiments to load into
                k-d tree

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
        self, candidates: List[BrewingParameters], tokenizer=None
    ) -> List[Tuple[BrewingParameters, float]]:
        """Score experiment candidates based on information gain potential and feasibility.

        Args:
            candidates: List of candidate brewing parameters
            tokenizer: Optional tokenizer for uncertainty scoring

        Returns:
            List of (parameters, score) tuples sorted by score descending
        """
        if not candidates:
            return []

        scored_candidates = []

        for params in candidates:
            # Calculate information gain score (0-1)
            info_gain_score = self._calculate_information_gain_score(params, tokenizer)

            # Calculate feasibility score (0-1)
            feasibility_score = self._calculate_feasibility_score(params)

            # Combine scores with weights (60% info gain, 40% feasibility)
            final_score = self._combine_scores(info_gain_score, feasibility_score)

            scored_candidates.append((params, final_score))

        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    def _calculate_information_gain_score(
        self, params: BrewingParameters, tokenizer=None
    ) -> float:
        """Calculate information gain potential score for brewing parameters.

        Args:
            params: Brewing parameters to score
            tokenizer: Optional tokenizer for uncertainty estimation

        Returns:
            Information gain score (0-1, higher is better)
        """
        info_gain = 0.0

        # Get uncertainty score if available
        if self.uncertainty_estimator is not None and tokenizer is not None:
            try:
                text_description = self._parameters_to_text(params)
                inputs = tokenizer(
                    text_description,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                uncertainty_metrics = self.uncertainty_estimator.estimate_uncertainty(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                uncertainty_score = (
                    uncertainty_metrics["total_uncertainty"].mean().item()
                )
                # Normalize uncertainty score to 0-1 range (assume max uncertainty ~3.0)
                info_gain += min(uncertainty_score / 3.0, 1.0) * 0.7
            except Exception:
                pass

                # Get exploration sparsity score if available
        if self.kdtree_explorer is not None:
            try:
                # Convert parameters to normalized vector for sparsity calculation
                parameter_ranges = ParameterRanges()
                normalized_vector = normalize_parameters(params, parameter_ranges)
                sparsity_score = self.kdtree_explorer.calculate_sparsity(
                    normalized_vector
                )
                info_gain += sparsity_score * 0.3
            except Exception:
                pass

        return min(info_gain, 1.0)

    def _calculate_feasibility_score(self, params: BrewingParameters) -> float:
        """Calculate feasibility score for brewing parameters.

        Args:
            params: Brewing parameters to evaluate

        Returns:
            Feasibility score (0-1, higher is better)
        """
        feasibility = 1.0

        # Check water temperature feasibility (optimal: 85-96°C)
        if params.water_temp < 80 or params.water_temp > 100:
            feasibility *= 0.3  # Very poor feasibility
        elif params.water_temp < 85 or params.water_temp > 96:
            feasibility *= 0.7  # Reduced feasibility

        # Check coffee dose to water ratio feasibility
        ratio = params.water_amount / params.coffee_dose
        if params.brew_method == BrewMethod.ESPRESSO:
            # Espresso ratio should be 1.5:1 to 3:1
            if ratio < 1.5 or ratio > 3.5:
                feasibility *= 0.4
            elif ratio < 2.0 or ratio > 3.0:
                feasibility *= 0.8
        else:
            # Pour over/other methods: 12:1 to 18:1
            if ratio < 10 or ratio > 20:
                feasibility *= 0.4
            elif ratio < 12 or ratio > 18:
                feasibility *= 0.8

        # Check brew time feasibility
        if params.brew_method == BrewMethod.ESPRESSO:
            # Espresso: 20-40 seconds
            if params.brew_time < 15 or params.brew_time > 60:
                feasibility *= 0.4
            elif params.brew_time < 20 or params.brew_time > 40:
                feasibility *= 0.8
        elif params.brew_method == BrewMethod.POUR_OVER:
            # Pour over: 3-6 minutes
            if params.brew_time < 120 or params.brew_time > 480:
                feasibility *= 0.4
            elif params.brew_time < 180 or params.brew_time > 360:
                feasibility *= 0.8

        # Check grind size appropriateness for method
        if params.brew_method == BrewMethod.ESPRESSO:
            if params.grind_size not in [GrindSize.VERY_FINE, GrindSize.FINE]:
                feasibility *= 0.6
        elif params.brew_method == BrewMethod.POUR_OVER:
            if params.grind_size not in [GrindSize.MEDIUM_FINE, GrindSize.MEDIUM]:
                feasibility *= 0.8

        return max(feasibility, 0.1)  # Minimum feasibility of 0.1

    def _combine_scores(
        self,
        info_gain_score: float,
        feasibility_score: float,
        info_gain_weight: float = 0.6,
        feasibility_weight: float = 0.4,
    ) -> float:
        """Combine information gain and feasibility scores.

        Args:
            info_gain_score: Information gain potential (0-1)
            feasibility_score: Experiment feasibility (0-1)
            info_gain_weight: Weight for information gain component
            feasibility_weight: Weight for feasibility component

        Returns:
            Combined score (0-1)
        """
        # Ensure weights sum to 1
        total_weight = info_gain_weight + feasibility_weight
        if total_weight > 0:
            info_gain_weight /= total_weight
            feasibility_weight /= total_weight

        combined_score = (
            info_gain_score * info_gain_weight + feasibility_score * feasibility_weight
        )

        return min(max(combined_score, 0.0), 1.0)

    def _validate_parameter_bounds(self, params: BrewingParameters) -> bool:
        """Validate that brewing parameters are within acceptable bounds.

        Args:
            params: Brewing parameters to validate

        Returns:
            True if parameters are within bounds, False otherwise
        """
        ranges = ParameterRanges()

        # Check core parameter bounds
        if not (ranges.water_temp_min <= params.water_temp <= ranges.water_temp_max):
            return False
        if not (ranges.coffee_dose_min <= params.coffee_dose <= ranges.coffee_dose_max):
            return False
        if not (
            ranges.water_amount_min <= params.water_amount <= ranges.water_amount_max
        ):
            return False
        if not (ranges.brew_time_min <= params.brew_time <= ranges.brew_time_max):
            return False

        # Check method-specific bounds
        if params.pressure is not None:
            if not (1.0 <= params.pressure <= 15.0):  # Bar pressure range
                return False

        if params.bloom_time is not None:
            if not (15.0 <= params.bloom_time <= 60.0):  # Bloom time range
                return False

        return True

    def _clamp_parameters(self, params: BrewingParameters) -> BrewingParameters:
        """Clamp brewing parameters to valid bounds.

        Args:
            params: Brewing parameters to clamp

        Returns:
            BrewingParameters with values clamped to valid ranges
        """
        ranges = ParameterRanges()

        # Clamp core parameters
        clamped_water_temp = max(
            ranges.water_temp_min, min(ranges.water_temp_max, params.water_temp)
        )
        clamped_coffee_dose = max(
            ranges.coffee_dose_min, min(ranges.coffee_dose_max, params.coffee_dose)
        )
        clamped_water_amount = max(
            ranges.water_amount_min, min(ranges.water_amount_max, params.water_amount)
        )
        clamped_brew_time = max(
            ranges.brew_time_min, min(ranges.brew_time_max, params.brew_time)
        )

        # Clamp method-specific parameters
        clamped_pressure = None
        if params.pressure is not None:
            clamped_pressure = max(1.0, min(15.0, params.pressure))

        clamped_bloom_time = None
        if params.bloom_time is not None:
            clamped_bloom_time = max(15.0, min(60.0, params.bloom_time))

        return BrewingParameters(
            water_temp=clamped_water_temp,
            coffee_dose=clamped_coffee_dose,
            water_amount=clamped_water_amount,
            grind_size=params.grind_size,  # Enum, already constrained
            brew_time=clamped_brew_time,
            brew_method=params.brew_method,
            pressure=clamped_pressure,
            bloom_time=clamped_bloom_time,
        )

    def generate_intelligent_experiments(
        self,
        num_experiments: int = 10,
        brew_method: BrewMethod = BrewMethod.POUR_OVER,
        tokenizer=None,
        uncertainty_weight: float = 0.6,
        exploration_weight: float = 0.4,
        uncertainty_threshold: Optional[float] = None,
        existing_experiments: Optional[List[Dict]] = None,
        enforce_bounds: bool = True,
        use_scoring: bool = True,
    ) -> List[BrewingParameters]:
        """Generate intelligent experiments using all available strategies.

        This is the main experiment generation method that orchestrates all components
        to produce high-quality experiment suggestions.

        Args:
            num_experiments: Number of experiments to generate
            brew_method: Target brewing method
            tokenizer: Tokenizer for uncertainty estimation (required for
                uncertainty-based generation)
            uncertainty_weight: Weight for uncertainty-based experiments (0-1)
            exploration_weight: Weight for exploration-based experiments (0-1)
            uncertainty_threshold: Minimum uncertainty threshold for candidates
            existing_experiments: Optional list of existing experiments for context
            enforce_bounds: Whether to enforce parameter bounds on generated experiments
            use_scoring: Whether to use the scoring system to rank candidates

        Returns:
            List of brewing parameters for intelligent experiments
        """
        if num_experiments <= 0:
            return []

        # Generate a larger pool of candidates for better selection
        candidate_pool_size = max(num_experiments * 3, 50)

        # Generate candidates using both strategies
        uncertainty_candidates = []
        exploration_candidates = []

        if self.uncertainty_estimator is not None and tokenizer is not None:
            uncertainty_candidates = self.generate_uncertainty_based_experiments(
                num_experiments=candidate_pool_size // 2,
                uncertainty_threshold=uncertainty_threshold,
                brew_method=brew_method,
                tokenizer=tokenizer,
            )

        if self.kdtree_explorer is not None:
            exploration_candidates = self.generate_exploration_based_experiments(
                num_experiments=candidate_pool_size // 2,
                brew_method=brew_method,
                existing_experiments=existing_experiments,
            )

        # Combine all candidates
        all_candidates = uncertainty_candidates + exploration_candidates

        # If no candidates generated, fall back to basic parameter generation
        if not all_candidates:
            all_candidates = self._generate_parameter_candidates(
                candidate_pool_size, brew_method
            )

        # Enforce parameter bounds if requested
        if enforce_bounds:
            valid_candidates = []
            for candidate in all_candidates:
                if self._validate_parameter_bounds(candidate):
                    valid_candidates.append(candidate)
                else:
                    # Clamp invalid parameters to bounds
                    clamped_candidate = self._clamp_parameters(candidate)
                    valid_candidates.append(clamped_candidate)
            all_candidates = valid_candidates

        # Use scoring system to select best candidates if requested
        if use_scoring and all_candidates:
            scored_candidates = self._score_experiment_candidates(
                all_candidates, tokenizer
            )
            selected_experiments = [
                params for params, _ in scored_candidates[:num_experiments]
            ]
        else:
            # Simple selection without scoring
            selected_experiments = all_candidates[:num_experiments]

        return selected_experiments

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
        description = (
            f"{method_text}, {params.coffee_dose:.1f}g coffee, "
            f"{params.water_amount:.0f}g water, {grind_text} grind, {time_text} brew time"
        )

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
