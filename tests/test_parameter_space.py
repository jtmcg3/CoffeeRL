#!/usr/bin/env python3
"""Tests for parameter_space module."""

import numpy as np
import pytest

from src.parameter_space import (
    BrewingParameters,
    BrewMethod,
    GrindSize,
    ParameterRanges,
    denormalize_parameters,
    normalize_parameters,
)


class TestBrewingParameters:
    """Test BrewingParameters dataclass."""

    def test_basic_creation(self):
        """Test basic parameter creation."""
        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )

        assert params.water_temp == 93.0
        assert params.coffee_dose == 18.0
        assert params.water_amount == 300.0
        assert params.grind_size == GrindSize.MEDIUM
        assert params.brew_time == 240.0
        assert params.brew_method == BrewMethod.POUR_OVER
        assert params.pressure is None
        assert params.bloom_time is None

    def test_espresso_with_pressure(self):
        """Test espresso parameters with pressure."""
        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=36.0,
            grind_size=GrindSize.FINE,
            brew_time=30.0,
            brew_method=BrewMethod.ESPRESSO,
            pressure=9.0,
        )

        assert params.pressure == 9.0
        assert params.brew_method == BrewMethod.ESPRESSO

    def test_pour_over_with_bloom(self):
        """Test pour over with bloom time."""
        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
            bloom_time=30.0,
        )

        assert params.bloom_time == 30.0

    def test_coffee_water_ratio_calculation(self):
        """Test coffee to water ratio calculation in vector."""
        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
        )

        ranges = ParameterRanges()
        vector = params.to_numerical_vector(ranges)
        expected_ratio = 18.0 / 300.0
        assert abs(vector[5] - expected_ratio) < 1e-6  # ratio is at index 5

    def test_from_dict_basic(self):
        """Test creation from dictionary."""
        data = {
            "water_temp": 93.0,
            "coffee_dose": 18.0,
            "water_amount": 300.0,
            "grind_size": "medium",
            "brew_time": 240.0,
            "brew_method": "pour_over",
        }

        params = BrewingParameters.from_dict(data)
        assert params.water_temp == 93.0
        assert params.grind_size == GrindSize.MEDIUM
        assert params.brew_method == BrewMethod.POUR_OVER

    def test_from_dict_with_optional_params(self):
        """Test creation from dictionary with optional parameters."""
        data = {
            "water_temp": 93.0,
            "coffee_dose": 18.0,
            "water_amount": 36.0,
            "grind_size": "fine",
            "brew_time": 30.0,
            "brew_method": "espresso",
            "pressure": 9.0,
        }

        params = BrewingParameters.from_dict(data)
        assert params.pressure == 9.0
        assert params.brew_method == BrewMethod.ESPRESSO

    def test_from_dict_invalid_grind_size(self):
        """Test handling of invalid grind size."""
        data = {
            "water_temp": 93.0,
            "coffee_dose": 18.0,
            "water_amount": 300.0,
            "grind_size": "invalid",
            "brew_time": 240.0,
            "brew_method": "pour_over",
        }

        params = BrewingParameters.from_dict(data)
        assert params.grind_size == GrindSize.MEDIUM  # Default fallback

    def test_from_dict_invalid_brew_method(self):
        """Test handling of invalid brew method."""
        data = {
            "water_temp": 93.0,
            "coffee_dose": 18.0,
            "water_amount": 300.0,
            "grind_size": "medium",
            "brew_time": 240.0,
            "brew_method": "invalid",
        }

        params = BrewingParameters.from_dict(data)
        assert params.brew_method == BrewMethod.POUR_OVER  # Default fallback

    def test_to_numerical_vector(self):
        """Test conversion to numerical vector."""
        params = BrewingParameters(
            water_temp=93.0,
            coffee_dose=18.0,
            water_amount=300.0,
            grind_size=GrindSize.MEDIUM,
            brew_time=240.0,
            brew_method=BrewMethod.POUR_OVER,
            bloom_time=30.0,
        )

        ranges = ParameterRanges()
        vector = params.to_numerical_vector(ranges)

        assert len(vector) == 7  # 6 core + 1 optional (bloom_time)
        assert vector[0] == 93.0  # water_temp
        assert vector[1] == 18.0  # coffee_dose
        assert vector[2] == 300.0  # water_amount
        assert vector[3] == 4.0  # grind_size (MEDIUM = 4.0)
        assert vector[4] == 240.0  # brew_time
        assert abs(vector[5] - (18.0 / 300.0)) < 1e-6  # ratio
        assert vector[6] == 30.0  # bloom_time


class TestParameterRanges:
    """Test ParameterRanges dataclass."""

    def test_default_ranges(self):
        """Test default parameter ranges."""
        ranges = ParameterRanges()

        assert ranges.water_temp_min == 80.0
        assert ranges.water_temp_max == 100.0
        assert ranges.coffee_dose_min == 10.0
        assert ranges.coffee_dose_max == 30.0
        assert ranges.water_amount_min == 150.0
        assert ranges.water_amount_max == 500.0
        assert ranges.brew_time_min == 60.0
        assert ranges.brew_time_max == 480.0

    def test_custom_ranges(self):
        """Test custom parameter ranges."""
        ranges = ParameterRanges(
            water_temp_min=85.0,
            water_temp_max=95.0,
            coffee_dose_min=15.0,
            coffee_dose_max=25.0,
        )

        assert ranges.water_temp_min == 85.0
        assert ranges.water_temp_max == 95.0
        assert ranges.coffee_dose_min == 15.0
        assert ranges.coffee_dose_max == 25.0

    def test_normalize_parameters(self):
        """Test parameter normalization."""
        ranges = ParameterRanges()
        params = BrewingParameters(
            water_temp=90.0,  # Mid-range
            coffee_dose=20.0,  # Mid-range
            water_amount=325.0,  # Mid-range
            grind_size=GrindSize.MEDIUM,
            brew_time=270.0,  # Mid-range
            brew_method=BrewMethod.POUR_OVER,
        )

        normalized = normalize_parameters(params, ranges)

        # All values should be around 0.5 for mid-range
        assert abs(normalized[0] - 0.5) < 0.1  # water_temp
        assert abs(normalized[1] - 0.5) < 0.1  # coffee_dose
        assert abs(normalized[2] - 0.5) < 0.1  # water_amount
        assert abs(normalized[4] - 0.5) < 0.1  # brew_time

    def test_normalize_edge_values(self):
        """Test normalization of edge values."""
        ranges = ParameterRanges()
        params = BrewingParameters(
            water_temp=80.0,  # Min value
            coffee_dose=30.0,  # Max value
            water_amount=150.0,  # Min value
            grind_size=GrindSize.VERY_FINE,  # Min grind
            brew_time=480.0,  # Max value
            brew_method=BrewMethod.POUR_OVER,
        )

        normalized = normalize_parameters(params, ranges)

        assert normalized[0] == 0.0  # Min water_temp
        assert normalized[1] == 1.0  # Max coffee_dose
        assert normalized[2] == 0.0  # Min water_amount
        assert normalized[3] == 0.0  # Min grind_size
        assert normalized[4] == 1.0  # Max brew_time

    def test_denormalize_parameters(self):
        """Test parameter denormalization."""
        ranges = ParameterRanges()
        normalized = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        params = denormalize_parameters(normalized, ranges, BrewMethod.POUR_OVER)

        # Should get mid-range values
        assert params.water_temp == 90.0
        assert params.coffee_dose == 20.0
        assert params.water_amount == 325.0
        assert params.grind_size == GrindSize.MEDIUM
        assert params.brew_time == 270.0

    def test_denormalize_with_optional_params(self):
        """Test denormalization with optional parameters."""
        ranges = ParameterRanges()
        # Include pressure value for espresso (7th element)
        normalized = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        params = denormalize_parameters(normalized, ranges, BrewMethod.ESPRESSO)

        # Should include pressure for espresso
        assert params.pressure is not None
        assert abs(params.pressure - 8.0) < 0.1  # Mid-range of 1-15

    def test_parameter_ranges_initialization(self):
        """Test parameter ranges initialization."""
        ranges = ParameterRanges()

        # Test that grind size mapping is created
        assert ranges.grind_size_mapping is not None
        assert len(ranges.grind_size_mapping) == 6
        assert ranges.grind_size_mapping[GrindSize.VERY_FINE] == 1.0
        assert ranges.grind_size_mapping[GrindSize.COARSE] == 6.0


class TestEnums:
    """Test enum classes."""

    def test_grind_size_values(self):
        """Test GrindSize enum values."""
        assert GrindSize.VERY_FINE.value == "very_fine"
        assert GrindSize.FINE.value == "fine"
        assert GrindSize.MEDIUM_FINE.value == "medium_fine"
        assert GrindSize.MEDIUM.value == "medium"
        assert GrindSize.MEDIUM_COARSE.value == "medium_coarse"
        assert GrindSize.COARSE.value == "coarse"

    def test_brew_method_values(self):
        """Test BrewMethod enum values."""
        assert BrewMethod.POUR_OVER.value == "pour_over"
        assert BrewMethod.FRENCH_PRESS.value == "french_press"
        assert BrewMethod.ESPRESSO.value == "espresso"
        assert BrewMethod.AEROPRESS.value == "aeropress"
        assert BrewMethod.COLD_BREW.value == "cold_brew"

    def test_grind_size_mapping(self):
        """Test that grind sizes have numerical mappings."""
        ranges = ParameterRanges()

        # Test that all grind sizes have mappings
        for grind_size in GrindSize:
            assert grind_size in ranges.grind_size_mapping
            assert isinstance(ranges.grind_size_mapping[grind_size], float)

        # Test ordering
        assert (
            ranges.grind_size_mapping[GrindSize.VERY_FINE]
            < ranges.grind_size_mapping[GrindSize.FINE]
        )
        assert (
            ranges.grind_size_mapping[GrindSize.FINE]
            < ranges.grind_size_mapping[GrindSize.MEDIUM]
        )
        assert (
            ranges.grind_size_mapping[GrindSize.MEDIUM]
            < ranges.grind_size_mapping[GrindSize.COARSE]
        )


if __name__ == "__main__":
    pytest.main([__file__])
