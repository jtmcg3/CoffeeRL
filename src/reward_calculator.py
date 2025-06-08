"""Reward Function Calculator for CoffeeRL.

This module implements the reward function that combines extraction yield accuracy,
brew time predictions, and user satisfaction ratings for reinforcement learning.
"""

from typing import Any, Dict, Optional


class RewardCalculator:
    """Calculates reward components for the reinforcement learning environment.

    This class handles the computation of various reward signals based on:
    - Extraction yield accuracy (18-22% optimal range)
    - Brew time prediction accuracy (15-second window)
    - User satisfaction ratings (5-point scale to -1.0 to 1.0 range)
    - Adaptive weighting based on data availability and reliability
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the reward calculator with configuration.

        Args:
            config: Dictionary containing reward scaling parameters and weights
        """
        # Default configuration
        default_config = {
            # Extraction yield parameters
            "yield_optimal_min": 18.0,
            "yield_optimal_max": 22.0,
            "yield_weight": 0.4,
            # Brew time parameters
            "brew_time_window": 15.0,  # seconds
            "brew_time_weight": 0.3,
            # User satisfaction parameters
            "satisfaction_weight": 0.3,
            # Adaptive weighting parameters
            "min_reliability_threshold": 0.5,
            "objective_bias": 0.7,  # Favor objective measurements initially
        }

        self.config = {**default_config, **(config or {})}

    def calculate_extraction_yield_reward(self, yield_percentage: float) -> float:
        """Calculate reward based on extraction yield accuracy.

        Uses a piecewise function that gives maximum reward within the 18-22% range
        and decreases linearly outside it.

        Args:
            yield_percentage: Extraction yield as percentage (0-100)

        Returns:
            float: Yield-based reward component (0.0 to 1.0)
        """
        # Input validation
        if yield_percentage is None or not isinstance(yield_percentage, (int, float)):
            return 0.0

        # Handle negative values
        if yield_percentage < 0:
            return 0.0

        # Clamp extremely high values (above 30% is unrealistic)
        yield_percentage = min(yield_percentage, 30.0)

        optimal_min = self.config["yield_optimal_min"]
        optimal_max = self.config["yield_optimal_max"]

        # Optimal range: maximum reward
        if optimal_min <= yield_percentage <= optimal_max:
            return 1.0

        # Below optimal range: linear decrease from optimal_min down to 0%
        elif yield_percentage < optimal_min:
            # Scale from 0.0 at 0% to 1.0 at optimal_min
            return max(0.0, yield_percentage / optimal_min)

        # Above optimal range: linear decrease from optimal_max to 30%
        else:  # yield_percentage > optimal_max
            # Scale from 1.0 at optimal_max to 0.0 at 30%
            max_yield = 30.0
            return max(
                0.0, 1.0 - (yield_percentage - optimal_max) / (max_yield - optimal_max)
            )

    def calculate_brew_time_reward(
        self, predicted_time: float, actual_time: float
    ) -> float:
        """Calculate reward based on brew time prediction accuracy.

        Uses a step function that gives full reward within 15-second window
        and zero outside.

        Args:
            predicted_time: Predicted brew time in seconds
            actual_time: Actual brew time in seconds

        Returns:
            float: Brew time-based reward component (0.0 or 1.0)
        """
        # Input validation
        if predicted_time is None or actual_time is None:
            return 0.0

        try:
            predicted_time = float(predicted_time)
            actual_time = float(actual_time)
        except (ValueError, TypeError):
            return 0.0

        # Handle negative values
        if predicted_time < 0 or actual_time < 0:
            return 0.0

        # Calculate absolute difference
        time_difference = abs(predicted_time - actual_time)

        # Apply step function with configurable accuracy window
        brew_time_window = self.config["brew_time_window"]
        if time_difference <= brew_time_window:
            return 1.0
        else:
            return 0.0

    def translate_user_satisfaction(self, rating: int) -> float:
        """Convert 5-point user satisfaction rating to -1.0 to 1.0 reward range.

        Linear transformation where:
        - 1 maps to -1.0
        - 3 maps to 0.0
        - 5 maps to 1.0

        Args:
            rating: User satisfaction rating (1-5)

        Returns:
            float: Satisfaction-based reward component (-1.0 to 1.0)
        """
        # Input validation
        if rating is None:
            return 0.0

        try:
            rating = int(rating)
        except (ValueError, TypeError):
            return 0.0

        # Check if rating is in valid range
        if rating < 1 or rating > 5:
            return 0.0

        # Linear transformation: (rating - 3) / 2
        # This maps: 1→-1.0, 2→-0.5, 3→0.0, 4→0.5, 5→1.0
        return (rating - 3) / 2.0

    def calculate_total_reward(
        self,
        yield_percentage: Optional[float] = None,
        predicted_time: Optional[float] = None,
        actual_time: Optional[float] = None,
        user_rating: Optional[int] = None,
        data_reliability: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate total reward combining all components with adaptive weighting.

        Args:
            yield_percentage: Extraction yield as percentage
            predicted_time: Predicted brew time in seconds
            actual_time: Actual brew time in seconds
            user_rating: User satisfaction rating (1-5)
            data_reliability: Dictionary with reliability scores for each component

        Returns:
            float: Total weighted reward
        """
        # Initialize default weights from config
        weights = {
            "yield": self.config["yield_weight"],
            "time": self.config["brew_time_weight"],
            "satisfaction": self.config["satisfaction_weight"],
        }

        # Initialize default reliability scores
        if data_reliability is None:
            data_reliability = {}

        reliability = {
            "yield": data_reliability.get("yield", 1.0),
            "time": data_reliability.get("time", 1.0),
            "satisfaction": data_reliability.get("satisfaction", 1.0),
        }

        # Calculate individual reward components
        rewards = {}
        available_components = []

        # Yield reward
        if yield_percentage is not None:
            rewards["yield"] = self.calculate_extraction_yield_reward(yield_percentage)
            available_components.append("yield")

        # Time reward (only if both predicted and actual times are available)
        if predicted_time is not None and actual_time is not None:
            rewards["time"] = self.calculate_brew_time_reward(
                predicted_time, actual_time
            )
            available_components.append("time")

        # Satisfaction reward
        if user_rating is not None:
            rewards["satisfaction"] = self.translate_user_satisfaction(user_rating)
            available_components.append("satisfaction")

        # Handle case where no data is available
        if not available_components:
            return 0.0

        # Apply reliability scores to weights and filter for available components
        adjusted_weights = {}
        for component in available_components:
            # Apply reliability threshold
            if reliability[component] >= self.config["min_reliability_threshold"]:
                adjusted_weights[component] = (
                    weights[component] * reliability[component]
                )
            else:
                # Low reliability data gets reduced weight
                adjusted_weights[component] = weights[component] * 0.1

        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight == 0:
            # Fallback: equal weights for available components
            adjusted_weights = {
                comp: 1.0 / len(available_components) for comp in available_components
            }
        else:
            adjusted_weights = {
                comp: weight / total_weight for comp, weight in adjusted_weights.items()
            }

        # Calculate weighted sum
        total_reward = sum(
            adjusted_weights[comp] * rewards[comp] for comp in available_components
        )

        return total_reward

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration.

        Returns:
            Dict containing current configuration parameters
        """
        return self.config.copy()

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration parameters.

        Args:
            new_config: Dictionary with new configuration values
        """
        self.config.update(new_config)
