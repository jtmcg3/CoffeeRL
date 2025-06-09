"""
Tests for experiment tracking system.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base, DatabaseManager, UserInteraction
from src.experiment_tracker import ExperimentTracker


class TestExperimentTracker:
    """Test experiment tracker functionality."""

    @pytest.fixture
    def db_manager(self):
        """Create a database manager with in-memory SQLite."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)

        # Create a mock database manager
        manager = DatabaseManager.__new__(DatabaseManager)
        manager.engine = engine
        manager.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )

        return manager

    @pytest.fixture
    def tracker(self, db_manager):
        """Create an experiment tracker with test database."""
        return ExperimentTracker(db_manager)

    def test_create_experiment(self, tracker):
        """Test creating a new experiment."""
        experiment = tracker.create_experiment(
            user_id="test_user",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
            bloom_time=30,
            predicted_score=8.5,
            uncertainty_score=0.3,
        )

        assert experiment is not None
        assert experiment.id is not None
        assert experiment.user_id == "test_user"
        assert experiment.brew_method == "V60"
        assert experiment.coffee_dose == 18.0
        assert experiment.water_amount == 300.0
        assert experiment.water_temperature == 92.0
        assert experiment.grind_size == "medium"
        assert experiment.brew_time == 240
        assert experiment.bloom_time == 30
        assert experiment.predicted_score == 8.5
        assert experiment.uncertainty_score == 0.3
        assert experiment.status == "pending"
        assert experiment.created_at is not None

    def test_get_experiment(self, tracker):
        """Test retrieving an experiment by ID."""
        # Create an experiment first
        experiment = tracker.create_experiment(
            user_id="test_user",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
        )

        # Retrieve it
        retrieved = tracker.get_experiment(experiment.id)

        assert retrieved is not None
        assert retrieved.id == experiment.id
        assert retrieved.user_id == "test_user"
        assert retrieved.brew_method == "V60"

    def test_get_experiment_nonexistent(self, tracker):
        """Test retrieving a non-existent experiment."""
        result = tracker.get_experiment(99999)
        assert result is None

    def test_start_experiment(self, tracker):
        """Test starting an experiment."""
        # Create an experiment
        experiment = tracker.create_experiment(
            user_id="test_user",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
        )

        # Start it
        success = tracker.start_experiment(experiment.id, "test_user")
        assert success is True

        # Verify status changed
        updated_experiment = tracker.get_experiment(experiment.id)
        assert updated_experiment.status == "in_progress"
        assert updated_experiment.started_at is not None

    def test_start_experiment_invalid_status(self, tracker):
        """Test starting an experiment with invalid current status."""
        # Create and start an experiment
        experiment = tracker.create_experiment(
            user_id="test_user",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
        )
        tracker.start_experiment(experiment.id, "test_user")

        # Try to start it again (should fail)
        success = tracker.start_experiment(experiment.id, "test_user")
        assert success is False

    def test_complete_experiment(self, tracker):
        """Test completing an experiment."""
        # Create and start an experiment
        experiment = tracker.create_experiment(
            user_id="test_user",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
        )
        tracker.start_experiment(experiment.id, "test_user")

        # Complete it
        success = tracker.complete_experiment(experiment.id, "test_user")
        assert success is True

        # Verify status changed
        updated_experiment = tracker.get_experiment(experiment.id)
        assert updated_experiment.status == "completed"
        assert updated_experiment.completed_at is not None

    def test_complete_experiment_invalid_status(self, tracker):
        """Test completing an experiment with invalid current status."""
        # Create an experiment but don't start it
        experiment = tracker.create_experiment(
            user_id="test_user",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
        )

        # Try to complete it (should fail - not started)
        success = tracker.complete_experiment(experiment.id, "test_user")
        assert success is False

    def test_experiment_with_wrong_user(self, tracker):
        """Test operations with wrong user ID."""
        # Create an experiment
        experiment = tracker.create_experiment(
            user_id="test_user",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
        )

        # Try to start with different user
        success = tracker.start_experiment(experiment.id, "different_user")
        assert success is False

    def test_experiment_with_nonexistent_id(self, tracker):
        """Test operations with non-existent experiment ID."""
        success = tracker.start_experiment(99999, "test_user")
        assert success is False

        success = tracker.complete_experiment(99999, "test_user")
        assert success is False

    def test_interaction_logging(self, tracker, db_manager):
        """Test that interactions are logged correctly."""
        # Create an experiment
        experiment = tracker.create_experiment(
            user_id="test_user",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
        )

        # Start the experiment
        tracker.start_experiment(experiment.id, "test_user")

        # Check interactions were logged
        with db_manager.get_session_context() as session:
            interactions = (
                session.query(UserInteraction)
                .filter_by(experiment_id=experiment.id)
                .order_by(UserInteraction.timestamp.asc())
                .all()
            )

            assert len(interactions) == 2  # create + start

            # Check create interaction
            create_interaction = interactions[0]
            assert create_interaction.interaction_type == "create"
            assert create_interaction.user_id == "test_user"
            assert create_interaction.interaction_data["experiment_created"] is True

            # Check start interaction
            start_interaction = interactions[1]
            assert start_interaction.interaction_type == "status_change"
            assert start_interaction.user_id == "test_user"
            assert start_interaction.interaction_data["to_status"] == "in_progress"

    def test_espresso_experiment(self, tracker):
        """Test creating an espresso experiment with pressure."""
        experiment = tracker.create_experiment(
            user_id="test_user",
            brew_method="Espresso",
            coffee_dose=20.0,
            water_amount=40.0,
            water_temperature=93.0,
            grind_size="fine",
            brew_time=30,
            pressure=9.0,
            predicted_score=8.0,
        )

        assert experiment.brew_method == "Espresso"
        assert experiment.pressure == 9.0
        assert experiment.bloom_time is None  # Not used for espresso

    def test_experiment_lifecycle_complete(self, tracker, db_manager):
        """Test complete experiment lifecycle from creation to completion."""
        # Create experiment
        experiment = tracker.create_experiment(
            user_id="test_user",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
            bloom_time=30,
        )

        assert experiment.status == "pending"
        assert experiment.started_at is None
        assert experiment.completed_at is None

        # Start experiment
        success = tracker.start_experiment(experiment.id, "test_user")
        assert success is True

        updated_experiment = tracker.get_experiment(experiment.id)
        assert updated_experiment.status == "in_progress"
        assert updated_experiment.started_at is not None
        assert updated_experiment.completed_at is None

        # Complete experiment
        success = tracker.complete_experiment(experiment.id, "test_user")
        assert success is True

        final_experiment = tracker.get_experiment(experiment.id)
        assert final_experiment.status == "completed"
        assert final_experiment.started_at is not None
        assert final_experiment.completed_at is not None

        # Verify all interactions were logged
        with db_manager.get_session_context() as session:
            interactions = (
                session.query(UserInteraction)
                .filter_by(experiment_id=experiment.id)
                .all()
            )

            assert len(interactions) == 3  # create, start, complete
            interaction_types = [i.interaction_type for i in interactions]
            assert "create" in interaction_types
            assert "status_change" in interaction_types


class TestExperimentTrackerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker with mocked database manager."""
        mock_db_manager = MagicMock()
        return ExperimentTracker(mock_db_manager)

    def test_initialization_with_default_manager(self):
        """Test tracker initialization with default database manager."""
        with patch("src.experiment_tracker.get_database_manager") as mock_get_db:
            mock_db_manager = MagicMock()
            mock_get_db.return_value = mock_db_manager

            tracker = ExperimentTracker()

            mock_get_db.assert_called_once()
            assert tracker.db_manager == mock_db_manager

    def test_initialization_with_custom_manager(self):
        """Test tracker initialization with custom database manager."""
        custom_manager = MagicMock()
        tracker = ExperimentTracker(custom_manager)

        assert tracker.db_manager == custom_manager
