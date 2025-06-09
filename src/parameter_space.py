#!/usr/bin/env python3
"""Parameter space exploration for coffee brewing using k-d trees.

This module defines the brewing parameter space and provides utilities for
efficient exploration using k-d tree data structures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class BrewMethod(Enum):
    """Supported brewing methods."""

    POUR_OVER = "pour_over"
    FRENCH_PRESS = "french_press"
    ESPRESSO = "espresso"
    AEROPRESS = "aeropress"
    COLD_BREW = "cold_brew"


class GrindSize(Enum):
    """Standardized grind size categories."""

    VERY_FINE = "very_fine"
    FINE = "fine"
    MEDIUM_FINE = "medium_fine"
    MEDIUM = "medium"
    MEDIUM_COARSE = "medium_coarse"
    COARSE = "coarse"


@dataclass
class ParameterRanges:
    """Defines valid ranges for brewing parameters."""

    # Core parameters (applicable to all brewing methods)
    water_temp_min: float = 80.0  # Celsius
    water_temp_max: float = 100.0

    coffee_dose_min: float = 10.0  # grams
    coffee_dose_max: float = 30.0

    water_amount_min: float = 150.0  # grams
    water_amount_max: float = 500.0

    brew_time_min: float = 60.0  # seconds
    brew_time_max: float = 480.0  # 8 minutes for cold brew

    # Grind size mapping to numerical values for k-d tree
    grind_size_mapping: Dict[GrindSize, float] = None

    def __post_init__(self):
        """Initialize grind size mapping after dataclass creation."""
        if self.grind_size_mapping is None:
            self.grind_size_mapping = {
                GrindSize.VERY_FINE: 1.0,
                GrindSize.FINE: 2.0,
                GrindSize.MEDIUM_FINE: 3.0,
                GrindSize.MEDIUM: 4.0,
                GrindSize.MEDIUM_COARSE: 5.0,
                GrindSize.COARSE: 6.0,
            }


@dataclass
class BrewingParameters:
    """Core brewing parameters for coffee experiments."""

    water_temp: float  # Celsius
    coffee_dose: float  # grams
    water_amount: float  # grams
    grind_size: GrindSize
    brew_time: float  # seconds
    brew_method: BrewMethod
    pressure: Optional[float] = None  # bar (for espresso/aeropress)
    bloom_time: Optional[float] = None  # seconds (for pour over)

    def to_numerical_vector(self, ranges: ParameterRanges) -> np.ndarray:
        """Convert brewing parameters to numerical vector.

        Args:
            ranges: Parameter ranges for grind size mapping

        Returns:
            Numerical vector representing the parameters
        """
        # Core dimensions
        grind_value = ranges.grind_size_mapping[self.grind_size]
        ratio = self.coffee_dose / self.water_amount

        vector = [
            self.water_temp,
            self.coffee_dose,
            self.water_amount,
            grind_value,
            self.brew_time,
            ratio,
        ]

        # Add method-specific parameters if present
        if self.pressure is not None:
            vector.append(self.pressure)
        if self.bloom_time is not None:
            vector.append(self.bloom_time)

        return np.array(vector)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrewingParameters":
        """Create BrewingParameters from dictionary.

        Args:
            data: Dictionary with parameter values

        Returns:
            BrewingParameters instance
        """
        # Convert string enums to enum objects
        grind_size = data["grind_size"]
        if isinstance(grind_size, str):
            grind_size = GrindSize(grind_size.lower())

        brew_method = data["brew_method"]
        if isinstance(brew_method, str):
            brew_method = BrewMethod(brew_method.lower())

        return cls(
            water_temp=float(data["water_temp"]),
            coffee_dose=float(data["coffee_dose"]),
            water_amount=float(data["water_amount"]),
            grind_size=grind_size,
            brew_time=float(data["brew_time"]),
            brew_method=brew_method,
            pressure=data.get("pressure"),
            bloom_time=data.get("bloom_time"),
        )


def get_core_dimensions() -> List[str]:
    """Get the names of core parameter dimensions.

    Returns:
        List of core dimension names
    """
    return [
        "water_temp",
        "coffee_dose",
        "water_amount",
        "grind_size",
        "brew_time",
        "coffee_water_ratio",
    ]


def get_method_specific_dimensions(brew_method: BrewMethod) -> List[str]:
    """Get method-specific parameter dimensions.

    Args:
        brew_method: Brewing method

    Returns:
        List of method-specific dimension names
    """
    method_dimensions = {
        BrewMethod.ESPRESSO: ["pressure"],
        BrewMethod.POUR_OVER: ["bloom_time"],
        BrewMethod.AEROPRESS: ["pressure"],
    }

    return method_dimensions.get(brew_method, [])


def normalize_parameters(
    params: BrewingParameters, ranges: ParameterRanges
) -> np.ndarray:
    """Normalize brewing parameters to [0, 1] range for k-d tree.

    Args:
        params: Brewing parameters to normalize
        ranges: Parameter ranges for normalization

    Returns:
        Normalized parameter vector
    """
    vector = params.to_numerical_vector(ranges)

    # Define normalization ranges for each dimension
    norm_ranges = [
        (ranges.water_temp_min, ranges.water_temp_max),  # water_temp
        (ranges.coffee_dose_min, ranges.coffee_dose_max),  # coffee_dose
        (ranges.water_amount_min, ranges.water_amount_max),  # water_amount
        (1.0, 6.0),  # grind_size (enum values)
        (ranges.brew_time_min, ranges.brew_time_max),  # brew_time
        (0.02, 0.2),  # coffee_water_ratio (typical range)
    ]

    # Add method-specific ranges if present
    if len(vector) > 6:
        if params.pressure is not None:
            norm_ranges.append((1.0, 15.0))  # pressure in bar
        if params.bloom_time is not None:
            norm_ranges.append((15.0, 60.0))  # bloom time in seconds

    # Normalize each dimension
    normalized = []
    for i, (value, (min_val, max_val)) in enumerate(zip(vector, norm_ranges)):
        normalized_value = (value - min_val) / (max_val - min_val)
        # Clamp to [0, 1] range
        normalized_value = max(0.0, min(1.0, normalized_value))
        normalized.append(normalized_value)

    return np.array(normalized)


def denormalize_parameters(
    normalized_vector: np.ndarray, ranges: ParameterRanges, brew_method: BrewMethod
) -> BrewingParameters:
    """Convert normalized vector back to brewing parameters.

    Args:
        normalized_vector: Normalized parameter vector
        ranges: Parameter ranges for denormalization
        brew_method: Target brewing method

    Returns:
        BrewingParameters instance
    """
    # Define denormalization ranges
    denorm_ranges = [
        (ranges.water_temp_min, ranges.water_temp_max),
        (ranges.coffee_dose_min, ranges.coffee_dose_max),
        (ranges.water_amount_min, ranges.water_amount_max),
        (1.0, 6.0),  # grind_size
        (ranges.brew_time_min, ranges.brew_time_max),
        (0.02, 0.2),  # coffee_water_ratio
    ]

    # Denormalize core parameters
    denormalized = []
    for i, (norm_val, (min_val, max_val)) in enumerate(
        zip(normalized_vector[:6], denorm_ranges)
    ):
        denorm_val = norm_val * (max_val - min_val) + min_val
        denormalized.append(denorm_val)

    # Convert grind size back to enum
    grind_value = round(denormalized[3])
    grind_value = max(1, min(6, grind_value))  # Clamp to valid range
    grind_mapping = {v: k for k, v in ranges.grind_size_mapping.items()}
    grind_size = grind_mapping.get(float(grind_value), GrindSize.MEDIUM)

    # Handle method-specific parameters
    pressure = None
    bloom_time = None

    if len(normalized_vector) > 6:
        if brew_method in [BrewMethod.ESPRESSO, BrewMethod.AEROPRESS]:
            pressure = normalized_vector[6] * (15.0 - 1.0) + 1.0
        if brew_method == BrewMethod.POUR_OVER and len(normalized_vector) > 6:
            idx = 7 if pressure is not None else 6
            if len(normalized_vector) > idx:
                bloom_time = normalized_vector[idx] * (60.0 - 15.0) + 15.0

    return BrewingParameters(
        water_temp=denormalized[0],
        coffee_dose=denormalized[1],
        water_amount=denormalized[2],
        grind_size=grind_size,
        brew_time=denormalized[4],
        brew_method=brew_method,
        pressure=pressure,
        bloom_time=bloom_time,
    )
