"""Tests for the RewardCalculator module."""

from src.reward_calculator import RewardCalculator


class TestRewardCalculatorStructure:
    """Test the basic structure and initialization of RewardCalculator."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        calculator = RewardCalculator()

        # Check that default config is set
        config = calculator.get_config()
        assert config["yield_optimal_min"] == 18.0
        assert config["yield_optimal_max"] == 22.0
        assert config["yield_weight"] == 0.4
        assert config["brew_time_window"] == 15.0
        assert config["brew_time_weight"] == 0.3
        assert config["satisfaction_weight"] == 0.3

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {"yield_optimal_min": 20.0, "yield_weight": 0.5}
        calculator = RewardCalculator(custom_config)

        config = calculator.get_config()
        # Custom values should override defaults
        assert config["yield_optimal_min"] == 20.0
        assert config["yield_weight"] == 0.5
        # Other defaults should remain
        assert config["yield_optimal_max"] == 22.0
        assert config["brew_time_weight"] == 0.3

    def test_update_config(self):
        """Test updating configuration after initialization."""
        calculator = RewardCalculator()

        new_config = {"yield_weight": 0.6}
        calculator.update_config(new_config)

        config = calculator.get_config()
        assert config["yield_weight"] == 0.6

    def test_methods_exist_and_return_correct_types(self):
        """Test that all required methods exist and return expected types."""
        calculator = RewardCalculator()

        # Test extraction yield reward method
        yield_reward = calculator.calculate_extraction_yield_reward(20.0)
        assert isinstance(yield_reward, float)
        assert 0.0 <= yield_reward <= 1.0  # Should be in valid range

        # Test brew time reward method
        time_reward = calculator.calculate_brew_time_reward(120.0, 125.0)
        assert isinstance(time_reward, float)
        assert time_reward in [0.0, 1.0]  # Step function returns 0 or 1

        # Test user satisfaction translation method
        satisfaction_reward = calculator.translate_user_satisfaction(3)
        assert isinstance(satisfaction_reward, float)
        assert -1.0 <= satisfaction_reward <= 1.0  # Should be in valid range

        # Test total reward calculation method
        total_reward = calculator.calculate_total_reward(
            yield_percentage=20.0,
            predicted_time=120.0,
            actual_time=125.0,
            user_rating=4,
        )
        assert isinstance(total_reward, float)
        assert total_reward >= 0.0  # Should be non-negative

    def test_method_signatures(self):
        """Test that methods accept the expected parameter types."""
        calculator = RewardCalculator()

        # Test with various parameter types
        calculator.calculate_extraction_yield_reward(20.5)
        calculator.calculate_brew_time_reward(120.5, 125.3)
        calculator.translate_user_satisfaction(4)

        # Test total reward with optional parameters
        calculator.calculate_total_reward()
        calculator.calculate_total_reward(yield_percentage=20.0)
        calculator.calculate_total_reward(
            yield_percentage=20.0,
            predicted_time=120.0,
            actual_time=125.0,
            user_rating=4,
            data_reliability={"yield": 0.9, "time": 0.8},
        )


class TestExtractionYieldReward:
    """Test the extraction yield reward calculation."""

    def test_optimal_range_yields_max_reward(self):
        """Test that yields in the optimal range (18-22%) give maximum reward."""
        calculator = RewardCalculator()

        # Test various values within optimal range
        assert calculator.calculate_extraction_yield_reward(18.0) == 1.0
        assert calculator.calculate_extraction_yield_reward(20.0) == 1.0
        assert calculator.calculate_extraction_yield_reward(22.0) == 1.0
        assert calculator.calculate_extraction_yield_reward(19.5) == 1.0

    def test_below_optimal_range_linear_decrease(self):
        """Test that yields below optimal range decrease linearly."""
        calculator = RewardCalculator()

        # Test specific points
        assert calculator.calculate_extraction_yield_reward(0.0) == 0.0
        assert calculator.calculate_extraction_yield_reward(9.0) == 0.5  # 9/18 = 0.5
        assert calculator.calculate_extraction_yield_reward(18.0) == 1.0

        # Test that it's actually linear
        reward_9 = calculator.calculate_extraction_yield_reward(9.0)
        reward_13_5 = calculator.calculate_extraction_yield_reward(13.5)
        assert abs(reward_13_5 - 0.75) < 1e-10  # 13.5/18 = 0.75
        assert reward_13_5 > reward_9

    def test_above_optimal_range_linear_decrease(self):
        """Test that yields above optimal range decrease linearly."""
        calculator = RewardCalculator()

        # Test specific points
        assert calculator.calculate_extraction_yield_reward(22.0) == 1.0
        assert calculator.calculate_extraction_yield_reward(30.0) == 0.0

        # Test midpoint: 26% should give 0.5 reward
        # (30 - 26) / (30 - 22) = 4/8 = 0.5, so reward = 1.0 - 0.5 = 0.5
        reward_26 = calculator.calculate_extraction_yield_reward(26.0)
        assert abs(reward_26 - 0.5) < 1e-10

    def test_edge_cases(self):
        """Test edge cases and input validation."""
        calculator = RewardCalculator()

        # Test negative values
        assert calculator.calculate_extraction_yield_reward(-5.0) == 0.0

        # Test None input
        assert calculator.calculate_extraction_yield_reward(None) == 0.0

        # Test extremely high values (should be clamped to 30%)
        assert calculator.calculate_extraction_yield_reward(50.0) == 0.0
        assert calculator.calculate_extraction_yield_reward(100.0) == 0.0

    def test_custom_optimal_range(self):
        """Test with custom optimal range configuration."""
        custom_config = {"yield_optimal_min": 20.0, "yield_optimal_max": 24.0}
        calculator = RewardCalculator(custom_config)

        # Test new optimal range
        assert calculator.calculate_extraction_yield_reward(20.0) == 1.0
        assert calculator.calculate_extraction_yield_reward(22.0) == 1.0
        assert calculator.calculate_extraction_yield_reward(24.0) == 1.0

        # Test outside new range
        assert calculator.calculate_extraction_yield_reward(18.0) < 1.0
        assert calculator.calculate_extraction_yield_reward(26.0) < 1.0


class TestBrewTimeReward:
    """Test the brew time reward calculation."""

    def test_exact_prediction_gives_max_reward(self):
        """Test that exact time predictions give maximum reward."""
        calculator = RewardCalculator()

        # Test exact matches
        assert calculator.calculate_brew_time_reward(120.0, 120.0) == 1.0
        assert calculator.calculate_brew_time_reward(90.0, 90.0) == 1.0
        assert calculator.calculate_brew_time_reward(180.0, 180.0) == 1.0

    def test_within_window_gives_max_reward(self):
        """Test that predictions within 15-second window give maximum reward."""
        calculator = RewardCalculator()

        # Test predictions within window (default 15 seconds)
        assert calculator.calculate_brew_time_reward(120.0, 135.0) == 1.0  # 15 sec diff
        assert calculator.calculate_brew_time_reward(120.0, 105.0) == 1.0  # 15 sec diff
        assert calculator.calculate_brew_time_reward(120.0, 130.0) == 1.0  # 10 sec diff
        assert calculator.calculate_brew_time_reward(120.0, 110.0) == 1.0  # 10 sec diff
        assert calculator.calculate_brew_time_reward(120.0, 125.0) == 1.0  # 5 sec diff

    def test_outside_window_gives_zero_reward(self):
        """Test that predictions outside 15-second window give zero reward."""
        calculator = RewardCalculator()

        # Test predictions outside window
        assert calculator.calculate_brew_time_reward(120.0, 136.0) == 0.0  # 16 sec diff
        assert calculator.calculate_brew_time_reward(120.0, 104.0) == 0.0  # 16 sec diff
        assert calculator.calculate_brew_time_reward(120.0, 150.0) == 0.0  # 30 sec diff
        assert calculator.calculate_brew_time_reward(120.0, 90.0) == 0.0  # 30 sec diff

    def test_edge_cases(self):
        """Test edge cases and input validation."""
        calculator = RewardCalculator()

        # Test None inputs
        assert calculator.calculate_brew_time_reward(None, 120.0) == 0.0
        assert calculator.calculate_brew_time_reward(120.0, None) == 0.0
        assert calculator.calculate_brew_time_reward(None, None) == 0.0

        # Test negative values
        assert calculator.calculate_brew_time_reward(-10.0, 120.0) == 0.0
        assert calculator.calculate_brew_time_reward(120.0, -10.0) == 0.0

        # Test invalid types (should be converted to float)
        assert calculator.calculate_brew_time_reward("120", "135") == 1.0
        assert calculator.calculate_brew_time_reward(120, 135) == 1.0

        # Test invalid string inputs
        assert calculator.calculate_brew_time_reward("invalid", 120.0) == 0.0
        assert calculator.calculate_brew_time_reward(120.0, "invalid") == 0.0

    def test_custom_time_window(self):
        """Test with custom time window configuration."""
        custom_config = {"brew_time_window": 10.0}  # 10-second window instead of 15
        calculator = RewardCalculator(custom_config)

        # Test within new window
        assert calculator.calculate_brew_time_reward(120.0, 130.0) == 1.0  # 10 sec diff
        assert calculator.calculate_brew_time_reward(120.0, 110.0) == 1.0  # 10 sec diff

        # Test outside new window (but within old window)
        assert calculator.calculate_brew_time_reward(120.0, 135.0) == 0.0  # 15 sec diff
        assert calculator.calculate_brew_time_reward(120.0, 105.0) == 0.0  # 15 sec diff

    def test_boundary_conditions(self):
        """Test exact boundary conditions."""
        calculator = RewardCalculator()

        # Test exactly at the boundary (15.0 seconds)
        assert calculator.calculate_brew_time_reward(120.0, 135.0) == 1.0
        assert calculator.calculate_brew_time_reward(120.0, 105.0) == 1.0

        # Test just outside the boundary
        assert calculator.calculate_brew_time_reward(120.0, 135.1) == 0.0
        assert calculator.calculate_brew_time_reward(120.0, 104.9) == 0.0


class TestUserSatisfactionTranslation:
    """Test the user satisfaction to reward scale translation."""

    def test_all_valid_ratings(self):
        """Test that all valid ratings (1-5) map correctly to reward scale."""
        calculator = RewardCalculator()

        # Test exact mappings as specified
        assert calculator.translate_user_satisfaction(1) == -1.0
        assert calculator.translate_user_satisfaction(2) == -0.5
        assert calculator.translate_user_satisfaction(3) == 0.0
        assert calculator.translate_user_satisfaction(4) == 0.5
        assert calculator.translate_user_satisfaction(5) == 1.0

    def test_linear_transformation_formula(self):
        """Test that the linear transformation formula (rating - 3) / 2 is correct."""
        calculator = RewardCalculator()

        # Verify the formula directly
        for rating in range(1, 6):
            expected = (rating - 3) / 2.0
            actual = calculator.translate_user_satisfaction(rating)
            assert abs(actual - expected) < 1e-10

    def test_invalid_ratings(self):
        """Test that invalid ratings return 0.0."""
        calculator = RewardCalculator()

        # Test ratings outside valid range
        assert calculator.translate_user_satisfaction(0) == 0.0
        assert calculator.translate_user_satisfaction(6) == 0.0
        assert calculator.translate_user_satisfaction(-1) == 0.0
        assert calculator.translate_user_satisfaction(10) == 0.0

    def test_edge_cases(self):
        """Test edge cases and input validation."""
        calculator = RewardCalculator()

        # Test None input
        assert calculator.translate_user_satisfaction(None) == 0.0

        # Test string inputs that can be converted to int
        assert calculator.translate_user_satisfaction("3") == 0.0
        assert calculator.translate_user_satisfaction("5") == 1.0

        # Test invalid string inputs
        assert calculator.translate_user_satisfaction("invalid") == 0.0
        assert calculator.translate_user_satisfaction("3.5") == 0.0  # Float string

        # Test float inputs (should be converted to int)
        assert calculator.translate_user_satisfaction(3.0) == 0.0
        assert (
            calculator.translate_user_satisfaction(3.9) == 0.0
        )  # Should truncate to 3
        assert (
            calculator.translate_user_satisfaction(4.1) == 0.5
        )  # Should truncate to 4

    def test_reward_range_bounds(self):
        """Test that all valid outputs are within the expected range."""
        calculator = RewardCalculator()

        for rating in range(1, 6):
            reward = calculator.translate_user_satisfaction(rating)
            assert -1.0 <= reward <= 1.0

    def test_monotonic_increase(self):
        """Test that higher ratings produce higher rewards."""
        calculator = RewardCalculator()

        rewards = [
            calculator.translate_user_satisfaction(rating) for rating in range(1, 6)
        ]

        # Check that rewards are in ascending order
        for i in range(1, len(rewards)):
            assert rewards[i] > rewards[i - 1]


class TestTotalRewardCalculation:
    """Test the total reward calculation with weighted combination."""

    def test_all_components_available(self):
        """Test total reward calculation when all components are available."""
        calculator = RewardCalculator()

        # Test with optimal values for all components
        total_reward = calculator.calculate_total_reward(
            yield_percentage=20.0,  # Optimal yield (reward = 1.0)
            predicted_time=120.0,  # Within 15-second window
            actual_time=125.0,  # (reward = 1.0)
            user_rating=5,  # Maximum satisfaction (reward = 1.0)
        )

        # With default weights (0.4, 0.3, 0.3) and all rewards = 1.0
        # Expected: 0.4 * 1.0 + 0.3 * 1.0 + 0.3 * 1.0 = 1.0
        assert abs(total_reward - 1.0) < 1e-10

    def test_partial_components_available(self):
        """Test total reward calculation with only some components available."""
        calculator = RewardCalculator()

        # Test with only yield and satisfaction
        total_reward = calculator.calculate_total_reward(
            yield_percentage=20.0,  # Optimal yield (reward = 1.0)
            user_rating=5,  # Maximum satisfaction (reward = 1.0)
        )

        # Only yield (0.4) and satisfaction (0.3) weights are used
        # Normalized weights: yield = 0.4/0.7 ≈ 0.571, satisfaction = 0.3/0.7 ≈ 0.429
        expected = (0.4 / 0.7) * 1.0 + (0.3 / 0.7) * 1.0
        assert abs(total_reward - expected) < 1e-10

    def test_no_components_available(self):
        """Test that no available components returns 0.0."""
        calculator = RewardCalculator()

        total_reward = calculator.calculate_total_reward()
        assert total_reward == 0.0

    def test_mixed_reward_values(self):
        """Test with mixed reward values (not all optimal)."""
        calculator = RewardCalculator()

        total_reward = calculator.calculate_total_reward(
            yield_percentage=15.0,  # Below optimal (reward ≈ 0.833)
            predicted_time=120.0,  # Outside window
            actual_time=140.0,  # (reward = 0.0)
            user_rating=3,  # Neutral satisfaction (reward = 0.0)
        )

        # Calculate expected yield reward: 15/18 ≈ 0.833
        yield_reward = 15.0 / 18.0

        # With default weights and normalization
        expected = 0.4 * yield_reward + 0.3 * 0.0 + 0.3 * 0.0
        assert abs(total_reward - expected) < 1e-10

    def test_data_reliability_adjustment(self):
        """Test that data reliability affects weighting."""
        calculator = RewardCalculator()

        # Test with low reliability for yield component
        total_reward_low_reliability = calculator.calculate_total_reward(
            yield_percentage=20.0,
            user_rating=5,
            data_reliability={
                "yield": 0.3,
                "satisfaction": 1.0,
            },  # Low yield reliability
        )

        # Test with high reliability for all components
        total_reward_high_reliability = calculator.calculate_total_reward(
            yield_percentage=20.0,
            user_rating=5,
            data_reliability={"yield": 1.0, "satisfaction": 1.0},
        )

        # Low reliability should result in different (likely lower) total reward
        assert total_reward_low_reliability != total_reward_high_reliability

    def test_reliability_threshold(self):
        """Test that very low reliability data gets reduced weight."""
        calculator = RewardCalculator()

        # Test with different reward values to make reliability effect more visible
        # Use suboptimal yield (15%) and poor satisfaction (rating 2)
        total_reward_low = calculator.calculate_total_reward(
            yield_percentage=15.0,  # Below optimal (reward ≈ 0.833)
            user_rating=2,  # Poor satisfaction (reward = -0.5)
            data_reliability={
                "yield": 0.2,
                "satisfaction": 1.0,
            },  # Low yield reliability
        )

        # Test with high reliability for comparison
        total_reward_high = calculator.calculate_total_reward(
            yield_percentage=15.0,  # Same yield
            user_rating=2,  # Same satisfaction
            data_reliability={"yield": 1.0, "satisfaction": 1.0},  # High reliability
        )

        # With low reliability on yield, the satisfaction component should get more weight
        # This should result in a different (likely lower) total reward
        assert abs(total_reward_low - total_reward_high) > 1e-10
        assert total_reward_low != total_reward_high

    def test_custom_weights(self):
        """Test with custom weight configuration."""
        custom_config = {
            "yield_weight": 0.6,
            "brew_time_weight": 0.2,
            "satisfaction_weight": 0.2,
        }
        calculator = RewardCalculator(custom_config)

        total_reward = calculator.calculate_total_reward(
            yield_percentage=20.0,  # reward = 1.0
            predicted_time=120.0,  # reward = 1.0
            actual_time=125.0,
            user_rating=5,  # reward = 1.0
        )

        # With custom weights: 0.6 * 1.0 + 0.2 * 1.0 + 0.2 * 1.0 = 1.0
        assert abs(total_reward - 1.0) < 1e-10

    def test_weight_normalization(self):
        """Test that weights are properly normalized."""
        calculator = RewardCalculator()

        # Test with only one component available
        total_reward = calculator.calculate_total_reward(yield_percentage=20.0)

        # Should get full reward since only one component is available
        assert abs(total_reward - 1.0) < 1e-10

    def test_edge_case_all_zero_weights(self):
        """Test edge case where all adjusted weights become zero."""
        calculator = RewardCalculator()

        # Test with all components having very low reliability
        total_reward = calculator.calculate_total_reward(
            yield_percentage=20.0,
            user_rating=5,
            data_reliability={"yield": 0.0, "satisfaction": 0.0},
        )

        # Should still get some reward due to fallback equal weighting
        assert total_reward > 0.0
