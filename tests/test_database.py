"""
Tests for database models and functionality.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import (
    Base,
    DatabaseManager,
    DatabaseSession,
    Experiment,
    ExperimentResult,
    UserInteraction,
    get_database_manager,
)


class TestDatabaseModels:
    """Test database model functionality."""

    @pytest.fixture
    def in_memory_db(self):
        """Create an in-memory SQLite database for testing."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        yield session
        session.close()

    def test_experiment_model_creation(self, in_memory_db):
        """Test creating an experiment model."""
        experiment = Experiment(
            user_id="test_user_123",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
            bloom_time=30,
            predicted_score=8.5,
            uncertainty_score=0.3,
            status="pending",
        )

        in_memory_db.add(experiment)
        in_memory_db.commit()

        # Verify the experiment was created
        saved_experiment = in_memory_db.query(Experiment).first()
        assert saved_experiment is not None
        assert saved_experiment.user_id == "test_user_123"
        assert saved_experiment.brew_method == "V60"
        assert saved_experiment.coffee_dose == 18.0
        assert saved_experiment.water_amount == 300.0
        assert saved_experiment.water_temperature == 92.0
        assert saved_experiment.grind_size == "medium"
        assert saved_experiment.brew_time == 240
        assert saved_experiment.bloom_time == 30
        assert saved_experiment.predicted_score == 8.5
        assert saved_experiment.uncertainty_score == 0.3
        assert saved_experiment.status == "pending"
        assert saved_experiment.created_at is not None

    def test_experiment_to_dict(self, in_memory_db):
        """Test experiment to_dict method."""
        experiment = Experiment(
            user_id="test_user_123",
            brew_method="Espresso",
            coffee_dose=20.0,
            water_amount=40.0,
            water_temperature=93.0,
            grind_size="fine",
            brew_time=30,
            pressure=9.0,
            predicted_score=7.8,
            status="completed",
        )

        in_memory_db.add(experiment)
        in_memory_db.commit()

        experiment_dict = experiment.to_dict()

        assert experiment_dict["user_id"] == "test_user_123"
        assert experiment_dict["brew_method"] == "Espresso"
        assert experiment_dict["coffee_dose"] == 20.0
        assert experiment_dict["pressure"] == 9.0
        assert experiment_dict["status"] == "completed"
        assert "created_at" in experiment_dict

    def test_user_interaction_model(self, in_memory_db):
        """Test creating a user interaction model."""
        # First create an experiment
        experiment = Experiment(
            user_id="test_user_123",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
        )
        in_memory_db.add(experiment)
        in_memory_db.commit()

        # Create interaction
        interaction = UserInteraction(
            experiment_id=experiment.id,
            user_id="test_user_123",
            interaction_type="start",
            interaction_data={"device": "mobile", "location": "kitchen"},
        )

        in_memory_db.add(interaction)
        in_memory_db.commit()

        # Verify the interaction was created
        saved_interaction = in_memory_db.query(UserInteraction).first()
        assert saved_interaction is not None
        assert saved_interaction.experiment_id == experiment.id
        assert saved_interaction.user_id == "test_user_123"
        assert saved_interaction.interaction_type == "start"
        assert saved_interaction.interaction_data["device"] == "mobile"
        assert saved_interaction.timestamp is not None

    def test_experiment_result_model(self, in_memory_db):
        """Test creating an experiment result model."""
        # First create an experiment
        experiment = Experiment(
            user_id="test_user_123",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
        )
        in_memory_db.add(experiment)
        in_memory_db.commit()

        # Create result
        result = ExperimentResult(
            experiment_id=experiment.id,
            user_id="test_user_123",
            taste_score=8.5,
            extraction_yield=22.5,
            tds=1.35,
            brew_ratio=16.7,
            notes="Excellent balance, slight acidity",
            tags=["fruity", "bright", "clean"],
            actual_brew_time=245,
            is_successful=True,
        )

        in_memory_db.add(result)
        in_memory_db.commit()

        # Verify the result was created
        saved_result = in_memory_db.query(ExperimentResult).first()
        assert saved_result is not None
        assert saved_result.experiment_id == experiment.id
        assert saved_result.user_id == "test_user_123"
        assert saved_result.taste_score == 8.5
        assert saved_result.extraction_yield == 22.5
        assert saved_result.tds == 1.35
        assert saved_result.brew_ratio == 16.7
        assert saved_result.notes == "Excellent balance, slight acidity"
        assert saved_result.tags == ["fruity", "bright", "clean"]
        assert saved_result.actual_brew_time == 245
        assert saved_result.is_successful is True
        assert saved_result.recorded_at is not None

    def test_model_relationships(self, in_memory_db):
        """Test relationships between models."""
        # Create experiment
        experiment = Experiment(
            user_id="test_user_123",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
        )
        in_memory_db.add(experiment)
        in_memory_db.commit()

        # Create interaction
        interaction = UserInteraction(
            experiment_id=experiment.id,
            user_id="test_user_123",
            interaction_type="start",
        )

        # Create result
        result = ExperimentResult(
            experiment_id=experiment.id,
            user_id="test_user_123",
            taste_score=8.0,
            is_successful=True,
        )

        in_memory_db.add_all([interaction, result])
        in_memory_db.commit()

        # Test relationships
        saved_experiment = in_memory_db.query(Experiment).first()
        assert len(saved_experiment.interactions) == 1
        assert len(saved_experiment.results) == 1
        assert saved_experiment.interactions[0].interaction_type == "start"
        assert saved_experiment.results[0].taste_score == 8.0

    def test_experiment_result_to_dict(self, in_memory_db):
        """Test experiment result to_dict method."""
        experiment = Experiment(
            user_id="test_user_123",
            brew_method="V60",
            coffee_dose=18.0,
            water_amount=300.0,
            water_temperature=92.0,
            grind_size="medium",
            brew_time=240,
        )
        in_memory_db.add(experiment)
        in_memory_db.commit()

        result = ExperimentResult(
            experiment_id=experiment.id,
            user_id="test_user_123",
            taste_score=8.5,
            extraction_yield=22.5,
            notes="Great coffee",
            tags=["fruity", "bright"],
        )
        in_memory_db.add(result)
        in_memory_db.commit()

        result_dict = result.to_dict()

        assert result_dict["experiment_id"] == experiment.id
        assert result_dict["user_id"] == "test_user_123"
        assert result_dict["taste_score"] == 8.5
        assert result_dict["extraction_yield"] == 22.5
        assert result_dict["notes"] == "Great coffee"
        assert result_dict["tags"] == ["fruity", "bright"]
        assert "recorded_at" in result_dict


class TestDatabaseManager:
    """Test database manager functionality."""

    @patch("src.database.create_engine")
    def test_database_manager_initialization(self, mock_create_engine):
        """Test database manager initialization."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")

        mock_create_engine.assert_called_once_with(
            "postgresql://test:test@localhost/test", echo=False
        )
        assert db_manager.engine == mock_engine

    @patch("src.database.create_engine")
    def test_create_tables(self, mock_create_engine):
        """Test table creation."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")

        with patch.object(Base.metadata, "create_all") as mock_create_all:
            db_manager.create_tables()
            mock_create_all.assert_called_once_with(bind=mock_engine)

    @patch("src.database.create_engine")
    def test_drop_tables(self, mock_create_engine):
        """Test table dropping."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")

        with patch.object(Base.metadata, "drop_all") as mock_drop_all:
            db_manager.drop_tables()
            mock_drop_all.assert_called_once_with(bind=mock_engine)

    @patch("src.database.create_engine")
    def test_get_session(self, mock_create_engine):
        """Test session creation."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")
        session = db_manager.get_session()

        assert session is not None

    @patch("src.database.create_engine")
    def test_get_session_context(self, mock_create_engine):
        """Test session context manager."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        db_manager = DatabaseManager("postgresql://test:test@localhost/test")
        session_context = db_manager.get_session_context()

        assert isinstance(session_context, DatabaseSession)


class TestDatabaseSession:
    """Test database session context manager."""

    def test_database_session_success(self):
        """Test successful database session."""
        mock_session = MagicMock()

        with DatabaseSession(mock_session) as session:
            assert session == mock_session

        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.rollback.assert_not_called()

    def test_database_session_exception(self):
        """Test database session with exception."""
        mock_session = MagicMock()

        try:
            with DatabaseSession(mock_session) as session:
                assert session == mock_session
                raise ValueError("Test exception")
        except ValueError:
            pass

        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.commit.assert_not_called()


class TestDatabaseUtilities:
    """Test database utility functions."""

    @patch("src.database.DatabaseManager")
    def test_get_database_manager_default_url(self, mock_db_manager):
        """Test getting database manager with default URL."""
        get_database_manager()

        mock_db_manager.assert_called_once_with("postgresql://localhost:5432/coffeerl")

    @patch("src.database.DatabaseManager")
    def test_get_database_manager_custom_url(self, mock_db_manager):
        """Test getting database manager with custom URL."""
        custom_url = "postgresql://custom:password@localhost:5433/custom_db"
        get_database_manager(custom_url)

        mock_db_manager.assert_called_once_with(custom_url)


class TestDatabaseIntegration:
    """Integration tests for database functionality."""

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

    def test_full_experiment_lifecycle(self, db_manager):
        """Test complete experiment lifecycle with all models."""
        with db_manager.get_session_context() as session:
            # Create experiment
            experiment = Experiment(
                user_id="integration_test_user",
                brew_method="V60",
                coffee_dose=18.0,
                water_amount=300.0,
                water_temperature=92.0,
                grind_size="medium",
                brew_time=240,
                bloom_time=30,
                predicted_score=8.0,
                uncertainty_score=0.4,
                status="pending",
            )
            session.add(experiment)
            session.flush()  # Get the ID

            # Add interaction
            interaction = UserInteraction(
                experiment_id=experiment.id,
                user_id="integration_test_user",
                interaction_type="start",
                interaction_data={"device": "web", "browser": "chrome"},
            )
            session.add(interaction)

            # Update experiment status
            experiment.status = "in_progress"
            experiment.started_at = datetime.utcnow()

            # Add result
            result = ExperimentResult(
                experiment_id=experiment.id,
                user_id="integration_test_user",
                taste_score=8.2,
                extraction_yield=21.8,
                tds=1.32,
                notes="Excellent cup with bright acidity",
                tags=["fruity", "clean", "balanced"],
                actual_brew_time=245,
                is_successful=True,
            )
            session.add(result)

            # Complete experiment
            experiment.status = "completed"
            experiment.completed_at = datetime.utcnow()

        # Verify everything was saved correctly
        with db_manager.get_session_context() as session:
            saved_experiment = (
                session.query(Experiment)
                .filter_by(user_id="integration_test_user")
                .first()
            )

            assert saved_experiment is not None
            assert saved_experiment.status == "completed"
            assert saved_experiment.started_at is not None
            assert saved_experiment.completed_at is not None

            assert len(saved_experiment.interactions) == 1
            assert saved_experiment.interactions[0].interaction_type == "start"

            assert len(saved_experiment.results) == 1
            assert saved_experiment.results[0].taste_score == 8.2
            assert saved_experiment.results[0].tags == ["fruity", "clean", "balanced"]
