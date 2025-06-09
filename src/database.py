"""
Database models and configuration for experiment tracking system.

This module provides SQLAlchemy models for tracking coffee brewing experiments,
user interactions, and results. Uses PostgreSQL for efficient data storage.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

Base = declarative_base()


class Experiment(Base):
    """Model for tracking coffee brewing experiments."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)

    # Experiment parameters
    brew_method = Column(String(50), nullable=False)
    coffee_dose = Column(Float, nullable=False)  # grams
    water_amount = Column(Float, nullable=False)  # grams
    water_temperature = Column(Float, nullable=False)  # celsius
    grind_size = Column(String(20), nullable=False)
    brew_time = Column(Integer, nullable=False)  # seconds
    pressure = Column(Float, nullable=True)  # bar (for espresso/aeropress)
    bloom_time = Column(Integer, nullable=True)  # seconds (for pour over)

    # Experiment metadata
    parameters_json = Column(JSON, nullable=True)  # Full parameter object
    predicted_score = Column(Float, nullable=True)
    uncertainty_score = Column(Float, nullable=True)

    # Lifecycle tracking
    status = Column(
        String(20), nullable=False, default="pending"
    )  # pending, in_progress, completed, cancelled
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    interactions = relationship(
        "UserInteraction", back_populates="experiment", cascade="all, delete-orphan"
    )
    results = relationship(
        "ExperimentResult", back_populates="experiment", cascade="all, delete-orphan"
    )

    # Indexes for efficient querying
    __table_args__ = (
        Index("idx_experiment_user_status", "user_id", "status"),
        Index("idx_experiment_created", "created_at"),
        Index("idx_experiment_brew_method", "brew_method"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "brew_method": self.brew_method,
            "coffee_dose": self.coffee_dose,
            "water_amount": self.water_amount,
            "water_temperature": self.water_temperature,
            "grind_size": self.grind_size,
            "brew_time": self.brew_time,
            "pressure": self.pressure,
            "bloom_time": self.bloom_time,
            "parameters_json": self.parameters_json,
            "predicted_score": self.predicted_score,
            "uncertainty_score": self.uncertainty_score,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }


class UserInteraction(Base):
    """Model for tracking user interactions with experiments."""

    __tablename__ = "user_interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)

    # Interaction details
    interaction_type = Column(
        String(50), nullable=False
    )  # view, start, pause, resume, complete, rate
    interaction_data = Column(JSON, nullable=True)  # Additional interaction data
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    experiment = relationship("Experiment", back_populates="interactions")

    # Indexes
    __table_args__ = (
        Index("idx_interaction_experiment", "experiment_id"),
        Index("idx_interaction_user_type", "user_id", "interaction_type"),
        Index("idx_interaction_timestamp", "timestamp"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert interaction to dictionary representation."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "user_id": self.user_id,
            "interaction_type": self.interaction_type,
            "interaction_data": self.interaction_data,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class ExperimentResult(Base):
    """Model for storing experiment results and outcomes."""

    __tablename__ = "experiment_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)

    # Result metrics
    taste_score = Column(Float, nullable=True)  # User-rated taste score (1-10)
    extraction_yield = Column(Float, nullable=True)  # Percentage
    tds = Column(Float, nullable=True)  # Total dissolved solids
    brew_ratio = Column(Float, nullable=True)  # Coffee to water ratio

    # Qualitative feedback
    notes = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # Array of tags

    # System metrics
    actual_brew_time = Column(Integer, nullable=True)  # Actual time taken
    temperature_profile = Column(JSON, nullable=True)  # Temperature over time

    # Metadata
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_successful = Column(Boolean, nullable=False, default=True)

    # Relationships
    experiment = relationship("Experiment", back_populates="results")

    # Indexes
    __table_args__ = (
        Index("idx_result_experiment", "experiment_id"),
        Index("idx_result_user", "user_id"),
        Index("idx_result_recorded", "recorded_at"),
        Index("idx_result_taste_score", "taste_score"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "user_id": self.user_id,
            "taste_score": self.taste_score,
            "extraction_yield": self.extraction_yield,
            "tds": self.tds,
            "brew_ratio": self.brew_ratio,
            "notes": self.notes,
            "tags": self.tags,
            "actual_brew_time": self.actual_brew_time,
            "temperature_profile": self.temperature_profile,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "is_successful": self.is_successful,
        }


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str):
        """Initialize database manager with connection URL."""
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self) -> None:
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def get_session_context(self):
        """Get a database session as a context manager."""
        return DatabaseSession(self.get_session())


class DatabaseSession:
    """Context manager for database sessions."""

    def __init__(self, session: Session):
        self.session = session

    def __enter__(self) -> Session:
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()


# Default database configuration
DEFAULT_DATABASE_URL = "postgresql://localhost:5432/coffeerl"


def get_database_manager(database_url: Optional[str] = None) -> DatabaseManager:
    """Get a database manager instance."""
    url = database_url or DEFAULT_DATABASE_URL
    return DatabaseManager(url)
