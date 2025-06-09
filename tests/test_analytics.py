"""
Test suite for analytics module.

Tests all analytics functions including completion rates, timing metrics,
correlation analysis, and performance statistics.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.analytics import Analytics, BrewingMethodStats, ExperimentStats
from src.database import (
    Base,
    DatabaseManager,
    Experiment,
    ExperimentResult,
    UserInteraction,
)


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


@pytest.fixture
def db_manager(in_memory_db):
    """Create a database manager with in-memory database."""
    manager = DatabaseManager("sqlite:///:memory:")
    # Use the same engine and session factory as the in_memory_db fixture
    manager.engine = in_memory_db.bind
    manager.SessionLocal = sessionmaker(bind=in_memory_db.bind)
    # Create tables using the manager's method
    manager.create_tables()
    return manager


@pytest.fixture
def analytics(db_manager):
    """Create analytics instance with test database."""
    return Analytics(db_manager)


@pytest.fixture
def sample_experiments(in_memory_db):
    """Create sample experiments for testing."""
    now = datetime.utcnow()

    experiments = [
        # Completed V60 experiment
        Experiment(
            id=1,
            user_id="test_user",
            brew_method="V60",
            coffee_dose=20.0,
            water_amount=300.0,
            water_temperature=93.0,
            grind_size="medium",
            brew_time=240,
            status="completed",
            predicted_score=8.5,
            created_at=now - timedelta(days=1),
            started_at=now - timedelta(days=1, hours=1),
            completed_at=now - timedelta(days=1, hours=0, minutes=30),
        ),
        # Completed Espresso experiment
        Experiment(
            id=2,
            user_id="test_user",
            brew_method="Espresso",
            coffee_dose=18.0,
            water_amount=36.0,
            water_temperature=93.0,
            grind_size="fine",
            brew_time=30,
            status="completed",
            predicted_score=7.8,
            created_at=now - timedelta(days=2),
            started_at=now - timedelta(days=2, hours=1),
            completed_at=now - timedelta(days=2, hours=0, minutes=45),
        ),
        # Running V60 experiment
        Experiment(
            id=3,
            user_id="test_user",
            brew_method="V60",
            coffee_dose=20.0,
            water_amount=300.0,
            water_temperature=93.0,
            grind_size="medium",
            brew_time=240,
            status="running",
            predicted_score=8.0,
            created_at=now - timedelta(hours=2),
            started_at=now - timedelta(hours=1),
        ),
        # Cancelled French Press experiment
        Experiment(
            id=4,
            user_id="test_user",
            brew_method="French Press",
            coffee_dose=30.0,
            water_amount=500.0,
            water_temperature=93.0,
            grind_size="coarse",
            brew_time=240,
            status="cancelled",
            predicted_score=7.2,
            created_at=now - timedelta(days=5),
            started_at=now - timedelta(days=5, hours=1),
        ),
        # Old completed experiment (outside 3-day window)
        Experiment(
            id=5,
            user_id="test_user",
            brew_method="V60",
            coffee_dose=20.0,
            water_amount=300.0,
            water_temperature=93.0,
            grind_size="medium",
            brew_time=240,
            status="completed",
            predicted_score=8.2,
            created_at=now - timedelta(days=10),
            started_at=now - timedelta(days=10, hours=1),
            completed_at=now - timedelta(days=10, hours=0, minutes=20),
        ),
    ]

    for exp in experiments:
        in_memory_db.add(exp)

    # Add experiment results
    results = [
        ExperimentResult(
            experiment_id=1,
            user_id="test_user",
            taste_score=8.7,
            extraction_yield=22.5,
            notes="Excellent balance",
        ),
        ExperimentResult(
            experiment_id=2,
            user_id="test_user",
            taste_score=7.5,
            extraction_yield=20.1,
            notes="Slightly bitter",
        ),
        ExperimentResult(
            experiment_id=5,
            user_id="test_user",
            taste_score=8.0,
            extraction_yield=21.8,
            notes="Good extraction",
        ),
    ]

    for result in results:
        in_memory_db.add(result)

    # Add user interactions
    interactions = [
        UserInteraction(
            experiment_id=1,
            user_id="test_user",
            interaction_type="create_experiment",
            timestamp=now - timedelta(days=1),
        ),
        UserInteraction(
            experiment_id=1,
            user_id="test_user",
            interaction_type="start_experiment",
            timestamp=now - timedelta(days=1, hours=1),
        ),
        UserInteraction(
            experiment_id=1,
            user_id="test_user",
            interaction_type="complete_experiment",
            timestamp=now - timedelta(days=1, hours=0, minutes=30),
        ),
        UserInteraction(
            experiment_id=2,
            user_id="test_user",
            interaction_type="create_experiment",
            timestamp=now - timedelta(days=2),
        ),
        UserInteraction(
            experiment_id=3,
            user_id="test_user",
            interaction_type="create_experiment",
            timestamp=now - timedelta(hours=2),
        ),
    ]

    for interaction in interactions:
        in_memory_db.add(interaction)

    in_memory_db.commit()
    return experiments


class TestAnalytics:
    """Test suite for Analytics class."""

    def test_calculate_completion_rate_all_experiments(
        self, analytics, sample_experiments
    ):
        """Test completion rate calculation for all experiments."""
        # 3 completed out of 5 total = 60%
        completion_rate = analytics.calculate_completion_rate()
        assert completion_rate == 60.0

    def test_calculate_completion_rate_with_time_filter(
        self, analytics, sample_experiments
    ):
        """Test completion rate with time filter."""
        # Last 3 days: 2 completed out of 4 total = 50%
        completion_rate = analytics.calculate_completion_rate(days_back=3)
        assert completion_rate == 50.0

    def test_calculate_completion_rate_by_brewing_method(
        self, analytics, sample_experiments
    ):
        """Test completion rate filtered by brewing method."""
        # V60: 2 completed out of 3 total = 66.67%
        completion_rate = analytics.calculate_completion_rate(brewing_method="V60")
        assert abs(completion_rate - 66.67) < 0.01

        # Espresso: 1 completed out of 1 total = 100%
        completion_rate = analytics.calculate_completion_rate(brewing_method="Espresso")
        assert completion_rate == 100.0

        # French Press: 0 completed out of 1 total = 0%
        completion_rate = analytics.calculate_completion_rate(
            brewing_method="French Press"
        )
        assert completion_rate == 0.0

    def test_calculate_completion_rate_no_experiments(self, analytics):
        """Test completion rate when no experiments exist."""
        completion_rate = analytics.calculate_completion_rate()
        assert completion_rate == 0.0

    def test_calculate_average_completion_time(self, analytics, sample_experiments):
        """Test average completion time calculation."""
        # Experiment 1: 30 minutes = 0.5 hours
        # Experiment 2: 15 minutes = 0.25 hours
        # Experiment 5: 40 minutes = 0.67 hours
        # Average: (0.5 + 0.25 + 0.67) / 3 = 0.47 hours
        avg_time = analytics.calculate_average_completion_time()
        assert abs(avg_time - 0.47) < 0.01

    def test_calculate_average_completion_time_with_filter(
        self, analytics, sample_experiments
    ):
        """Test average completion time with time filter."""
        # Last 3 days: only experiments 1 and 2
        # Average: (0.5 + 0.25) / 2 = 0.375 hours
        avg_time = analytics.calculate_average_completion_time(days_back=3)
        assert abs(avg_time - 0.375) < 0.01

    def test_calculate_average_completion_time_no_completed(self, analytics):
        """Test average completion time when no completed experiments exist."""
        avg_time = analytics.calculate_average_completion_time()
        assert avg_time is None

    def test_calculate_prediction_accuracy(self, analytics, sample_experiments):
        """Test prediction accuracy calculation."""
        # Experiment 1: predicted=8.5, actual=8.7
        # Experiment 2: predicted=7.8, actual=7.5
        # Experiment 5: predicted=8.2, actual=8.0
        accuracy = analytics.calculate_prediction_accuracy()
        # Should be a positive correlation
        assert accuracy is not None
        assert -1.0 <= accuracy <= 1.0

    def test_calculate_prediction_accuracy_insufficient_data(
        self, analytics, in_memory_db
    ):
        """Test prediction accuracy with insufficient data."""
        # Add only one experiment with result
        exp = Experiment(
            id=1,
            user_id="test_user",
            brew_method="V60",
            coffee_dose=20.0,
            water_amount=300.0,
            water_temperature=93.0,
            grind_size="medium",
            brew_time=240,
            status="completed",
            predicted_score=8.0,
        )
        result = ExperimentResult(experiment_id=1, user_id="test_user", taste_score=8.2)
        in_memory_db.add(exp)
        in_memory_db.add(result)
        in_memory_db.commit()

        accuracy = analytics.calculate_prediction_accuracy()
        assert accuracy is None

    def test_get_experiment_statistics(self, analytics, sample_experiments):
        """Test comprehensive experiment statistics."""
        stats = analytics.get_experiment_statistics()

        assert isinstance(stats, ExperimentStats)
        assert stats.total_experiments == 5
        assert stats.completed_experiments == 3
        assert stats.completion_rate == 60.0
        assert stats.average_completion_time_hours is not None
        assert stats.prediction_accuracy is not None
        assert stats.most_popular_method == "V60"  # 3 V60 experiments

    def test_get_experiment_statistics_with_time_filter(
        self, analytics, sample_experiments
    ):
        """Test experiment statistics with time filter."""
        stats = analytics.get_experiment_statistics(days_back=3)

        assert stats.total_experiments == 4  # Excludes old experiment
        assert stats.completed_experiments == 2
        assert stats.completion_rate == 50.0

    def test_get_brewing_method_performance(self, analytics, sample_experiments):
        """Test brewing method performance statistics."""
        performance = analytics.get_brewing_method_performance()

        assert len(performance) == 3  # V60, Espresso, French Press

        # Should be sorted by total experiments (V60 first with 3)
        v60_stats = performance[0]
        assert v60_stats.method == "V60"
        assert v60_stats.total_experiments == 3
        assert abs(v60_stats.completion_rate - 66.67) < 0.01
        assert v60_stats.average_taste_score is not None

        # Check Espresso stats
        espresso_stats = next(s for s in performance if s.method == "Espresso")
        assert espresso_stats.total_experiments == 1
        assert espresso_stats.completion_rate == 100.0

        # Check French Press stats
        french_press_stats = next(s for s in performance if s.method == "French Press")
        assert french_press_stats.total_experiments == 1
        assert french_press_stats.completion_rate == 0.0
        assert french_press_stats.average_taste_score is None  # No results

    def test_get_user_engagement_metrics(self, analytics, sample_experiments):
        """Test user engagement metrics calculation."""
        metrics = analytics.get_user_engagement_metrics()

        assert metrics["total_interactions"] == 5
        assert "action_distribution" in metrics
        assert metrics["action_distribution"]["create_experiment"] == 3
        assert metrics["action_distribution"]["start_experiment"] == 1
        assert metrics["action_distribution"]["complete_experiment"] == 1
        assert (
            metrics["average_interactions_per_experiment"] == 1.0
        )  # 5 interactions / 5 experiments

    def test_get_user_engagement_metrics_with_time_filter(
        self, analytics, sample_experiments
    ):
        """Test user engagement metrics with time filter."""
        metrics = analytics.get_user_engagement_metrics(days_back=1)

        # Should only include interactions from last day (experiment 1)
        assert metrics["total_interactions"] == 3
        assert metrics["action_distribution"]["create_experiment"] == 1
        assert metrics["action_distribution"]["start_experiment"] == 1
        assert metrics["action_distribution"]["complete_experiment"] == 1

    def test_calculate_correlation_perfect_positive(self, analytics):
        """Test correlation calculation with perfect positive correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        correlation = analytics._calculate_correlation(x, y)
        assert abs(correlation - 1.0) < 0.001

    def test_calculate_correlation_perfect_negative(self, analytics):
        """Test correlation calculation with perfect negative correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        correlation = analytics._calculate_correlation(x, y)
        assert abs(correlation - (-1.0)) < 0.001

    def test_calculate_correlation_no_correlation(self, analytics):
        """Test correlation calculation with no correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 1.0, 1.0, 1.0, 1.0]  # Constant values
        correlation = analytics._calculate_correlation(x, y)
        assert correlation == 0.0

    def test_calculate_correlation_insufficient_data(self, analytics):
        """Test correlation calculation with insufficient data."""
        x = [1.0]
        y = [2.0]
        correlation = analytics._calculate_correlation(x, y)
        assert correlation == 0.0

    def test_calculate_correlation_mismatched_lengths(self, analytics):
        """Test correlation calculation with mismatched array lengths."""
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0]
        correlation = analytics._calculate_correlation(x, y)
        assert correlation == 0.0


class TestAnalyticsEdgeCases:
    """Test edge cases and error conditions for Analytics class."""

    def test_empty_database(self, analytics):
        """Test analytics functions with empty database."""
        assert analytics.calculate_completion_rate() == 0.0
        assert analytics.calculate_average_completion_time() is None
        assert analytics.calculate_prediction_accuracy() is None

        stats = analytics.get_experiment_statistics()
        assert stats.total_experiments == 0
        assert stats.completion_rate == 0.0
        assert stats.most_popular_method is None

        performance = analytics.get_brewing_method_performance()
        assert len(performance) == 0

        metrics = analytics.get_user_engagement_metrics()
        assert metrics["total_interactions"] == 0
        assert metrics["average_interactions_per_experiment"] == 0.0

    def test_experiments_without_results(self, analytics, in_memory_db):
        """Test analytics with experiments that have no results."""
        exp = Experiment(
            id=1,
            user_id="test_user",
            brew_method="V60",
            coffee_dose=20.0,
            water_amount=300.0,
            water_temperature=93.0,
            grind_size="medium",
            brew_time=240,
            status="completed",
            predicted_score=8.0,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow() - timedelta(hours=1),
            completed_at=datetime.utcnow(),
        )
        in_memory_db.add(exp)
        in_memory_db.commit()

        # Should handle missing results gracefully
        assert analytics.calculate_completion_rate() == 100.0
        assert analytics.calculate_average_completion_time() is not None
        assert (
            analytics.calculate_prediction_accuracy() is None
        )  # No results to correlate

    def test_experiments_without_timing_data(self, analytics, in_memory_db):
        """Test analytics with experiments missing timing data."""
        exp = Experiment(
            id=1,
            user_id="test_user",
            brew_method="V60",
            coffee_dose=20.0,
            water_amount=300.0,
            water_temperature=93.0,
            grind_size="medium",
            brew_time=240,
            status="completed",
            predicted_score=8.0,
            created_at=datetime.utcnow(),
            # Missing started_at and completed_at
        )
        in_memory_db.add(exp)
        in_memory_db.commit()

        # Should handle missing timing data gracefully
        assert analytics.calculate_completion_rate() == 100.0
        assert analytics.calculate_average_completion_time() is None

    @patch("src.analytics.datetime")
    def test_time_filter_edge_cases(self, mock_datetime, analytics, sample_experiments):
        """Test time filtering edge cases."""
        # Mock current time
        mock_now = datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.utcnow.return_value = mock_now

        # Test with very large days_back value
        completion_rate = analytics.calculate_completion_rate(days_back=365)
        assert completion_rate >= 0.0

        # Test with zero days_back
        completion_rate = analytics.calculate_completion_rate(days_back=0)
        assert completion_rate >= 0.0
