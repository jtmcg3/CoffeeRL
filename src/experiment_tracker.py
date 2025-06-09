"""
Experiment tracking system for managing coffee brewing experiment lifecycles.

This module provides high-level functions to create, read, update, and delete
experiments, user interactions, and results. Handles experiment lifecycle
management with proper state transitions and error handling.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .database import (
    DatabaseManager,
    Experiment,
    ExperimentResult,
    UserInteraction,
    get_database_manager,
)


class ExperimentTracker:
    """High-level interface for experiment tracking operations."""

    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """Initialize experiment tracker with database manager."""
        self.db_manager = database_manager or get_database_manager()

    # Experiment CRUD Operations

    def create_experiment(
        self,
        user_id: str,
        brew_method: str,
        coffee_dose: float,
        water_amount: float,
        water_temperature: float,
        grind_size: str,
        brew_time: int,
        pressure: Optional[float] = None,
        bloom_time: Optional[int] = None,
        parameters_json: Optional[Dict[str, Any]] = None,
        predicted_score: Optional[float] = None,
        uncertainty_score: Optional[float] = None,
    ) -> Experiment:
        """Create a new experiment."""
        with self.db_manager.get_session_context() as session:
            experiment = Experiment(
                user_id=user_id,
                brew_method=brew_method,
                coffee_dose=coffee_dose,
                water_amount=water_amount,
                water_temperature=water_temperature,
                grind_size=grind_size,
                brew_time=brew_time,
                pressure=pressure,
                bloom_time=bloom_time,
                parameters_json=parameters_json,
                predicted_score=predicted_score,
                uncertainty_score=uncertainty_score,
                status="pending",
            )
            session.add(experiment)
            session.flush()  # Get the ID

            # Log creation interaction
            self._log_interaction(
                session, experiment.id, user_id, "create", {"experiment_created": True}
            )

            # Refresh to ensure all attributes are loaded
            session.refresh(experiment)

            # Detach from session to avoid DetachedInstanceError
            session.expunge(experiment)

            return experiment

    def get_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """Get an experiment by ID."""
        with self.db_manager.get_session_context() as session:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if experiment:
                # Detach from session to avoid DetachedInstanceError
                session.expunge(experiment)
            return experiment

    def get_experiments_by_user(
        self, user_id: str, status: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Experiment]:
        """Get experiments for a specific user, optionally filtered by status."""
        with self.db_manager.get_session_context() as session:
            query = session.query(Experiment).filter_by(user_id=user_id)

            if status:
                query = query.filter_by(status=status)

            query = query.order_by(Experiment.created_at.desc())

            if limit:
                query = query.limit(limit)

            return query.all()

    def update_experiment(
        self, experiment_id: int, user_id: str, **updates
    ) -> Optional[Experiment]:
        """Update an experiment with new values."""
        with self.db_manager.get_session_context() as session:
            experiment = (
                session.query(Experiment)
                .filter_by(id=experiment_id, user_id=user_id)
                .first()
            )

            if not experiment:
                return None

            # Update fields
            for field, value in updates.items():
                if hasattr(experiment, field):
                    setattr(experiment, field, value)

            # Log update interaction
            self._log_interaction(
                session,
                experiment_id,
                user_id,
                "update",
                {"fields_updated": list(updates.keys())},
            )

            return experiment

    def delete_experiment(self, experiment_id: int, user_id: str) -> bool:
        """Delete an experiment and all related data."""
        with self.db_manager.get_session_context() as session:
            experiment = (
                session.query(Experiment)
                .filter_by(id=experiment_id, user_id=user_id)
                .first()
            )

            if not experiment:
                return False

            # Log deletion interaction before deleting
            self._log_interaction(
                session, experiment_id, user_id, "delete", {"experiment_deleted": True}
            )

            session.delete(experiment)
            return True

    # Experiment Lifecycle Management

    def start_experiment(self, experiment_id: int, user_id: str) -> bool:
        """Start an experiment (transition from pending to in_progress)."""
        return self._transition_experiment_status(
            experiment_id,
            user_id,
            "pending",
            "in_progress",
            started_at=datetime.utcnow(),
        )

    def pause_experiment(self, experiment_id: int, user_id: str) -> bool:
        """Pause an experiment (transition from in_progress to paused)."""
        return self._transition_experiment_status(
            experiment_id, user_id, "in_progress", "paused"
        )

    def resume_experiment(self, experiment_id: int, user_id: str) -> bool:
        """Resume an experiment (transition from paused to in_progress)."""
        return self._transition_experiment_status(
            experiment_id, user_id, "paused", "in_progress"
        )

    def complete_experiment(self, experiment_id: int, user_id: str) -> bool:
        """Complete an experiment (transition to completed)."""
        return self._transition_experiment_status(
            experiment_id,
            user_id,
            ["in_progress", "paused"],
            "completed",
            completed_at=datetime.utcnow(),
        )

    def cancel_experiment(self, experiment_id: int, user_id: str) -> bool:
        """Cancel an experiment (transition to cancelled)."""
        return self._transition_experiment_status(
            experiment_id, user_id, ["pending", "in_progress", "paused"], "cancelled"
        )

    def _transition_experiment_status(
        self,
        experiment_id: int,
        user_id: str,
        from_status: str | List[str],
        to_status: str,
        **additional_updates,
    ) -> bool:
        """Helper method to transition experiment status with validation."""
        with self.db_manager.get_session_context() as session:
            experiment = (
                session.query(Experiment)
                .filter_by(id=experiment_id, user_id=user_id)
                .first()
            )

            if not experiment:
                return False

            # Validate current status
            valid_from_statuses = (
                [from_status] if isinstance(from_status, str) else from_status
            )
            if experiment.status not in valid_from_statuses:
                return False

            # Update status and additional fields
            experiment.status = to_status
            for field, value in additional_updates.items():
                setattr(experiment, field, value)

            # Log status transition
            self._log_interaction(
                session,
                experiment_id,
                user_id,
                "status_change",
                {
                    "from_status": experiment.status,
                    "to_status": to_status,
                },
            )

            return True

    # User Interaction Management

    def log_user_interaction(
        self,
        experiment_id: int,
        user_id: str,
        interaction_type: str,
        interaction_data: Optional[Dict[str, Any]] = None,
    ) -> UserInteraction:
        """Log a user interaction with an experiment."""
        with self.db_manager.get_session_context() as session:
            return self._log_interaction(
                session, experiment_id, user_id, interaction_type, interaction_data
            )

    def get_user_interactions(
        self,
        experiment_id: Optional[int] = None,
        user_id: Optional[str] = None,
        interaction_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[UserInteraction]:
        """Get user interactions with optional filtering."""
        with self.db_manager.get_session_context() as session:
            query = session.query(UserInteraction)

            if experiment_id:
                query = query.filter_by(experiment_id=experiment_id)
            if user_id:
                query = query.filter_by(user_id=user_id)
            if interaction_type:
                query = query.filter_by(interaction_type=interaction_type)

            query = query.order_by(UserInteraction.timestamp.desc())

            if limit:
                query = query.limit(limit)

            return query.all()

    def _log_interaction(
        self,
        session: Session,
        experiment_id: int,
        user_id: str,
        interaction_type: str,
        interaction_data: Optional[Dict[str, Any]] = None,
    ) -> UserInteraction:
        """Internal method to log interactions within a session."""
        interaction = UserInteraction(
            experiment_id=experiment_id,
            user_id=user_id,
            interaction_type=interaction_type,
            interaction_data=interaction_data or {},
        )
        session.add(interaction)
        session.flush()
        return interaction

    # Experiment Result Management

    def record_experiment_result(
        self,
        experiment_id: int,
        user_id: str,
        taste_score: Optional[float] = None,
        extraction_yield: Optional[float] = None,
        tds: Optional[float] = None,
        brew_ratio: Optional[float] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        actual_brew_time: Optional[int] = None,
        temperature_profile: Optional[Dict[str, Any]] = None,
        is_successful: bool = True,
    ) -> Optional[ExperimentResult]:
        """Record the result of an experiment."""
        with self.db_manager.get_session_context() as session:
            # Verify experiment exists and belongs to user
            experiment = (
                session.query(Experiment)
                .filter_by(id=experiment_id, user_id=user_id)
                .first()
            )

            if not experiment:
                return None

            result = ExperimentResult(
                experiment_id=experiment_id,
                user_id=user_id,
                taste_score=taste_score,
                extraction_yield=extraction_yield,
                tds=tds,
                brew_ratio=brew_ratio,
                notes=notes,
                tags=tags,
                actual_brew_time=actual_brew_time,
                temperature_profile=temperature_profile,
                is_successful=is_successful,
            )
            session.add(result)
            session.flush()

            # Log result recording interaction
            self._log_interaction(
                session,
                experiment_id,
                user_id,
                "record_result",
                {
                    "result_id": result.id,
                    "taste_score": taste_score,
                    "is_successful": is_successful,
                },
            )

            return result

    def get_experiment_results(
        self,
        experiment_id: Optional[int] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ExperimentResult]:
        """Get experiment results with optional filtering."""
        with self.db_manager.get_session_context() as session:
            query = session.query(ExperimentResult)

            if experiment_id:
                query = query.filter_by(experiment_id=experiment_id)
            if user_id:
                query = query.filter_by(user_id=user_id)

            query = query.order_by(ExperimentResult.recorded_at.desc())

            if limit:
                query = query.limit(limit)

            return query.all()

    def update_experiment_result(
        self, result_id: int, user_id: str, **updates
    ) -> Optional[ExperimentResult]:
        """Update an experiment result."""
        with self.db_manager.get_session_context() as session:
            result = (
                session.query(ExperimentResult)
                .filter_by(id=result_id, user_id=user_id)
                .first()
            )

            if not result:
                return None

            # Update fields
            for field, value in updates.items():
                if hasattr(result, field):
                    setattr(result, field, value)

            # Log update interaction
            self._log_interaction(
                session,
                result.experiment_id,
                user_id,
                "update_result",
                {
                    "result_id": result_id,
                    "fields_updated": list(updates.keys()),
                },
            )

            return result

    # Utility Methods

    def get_experiment_summary(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a comprehensive summary of an experiment including interactions and results.
        """
        with self.db_manager.get_session_context() as session:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()

            if not experiment:
                return None

            # Get related data
            interactions = (
                session.query(UserInteraction)
                .filter_by(experiment_id=experiment_id)
                .order_by(UserInteraction.timestamp.asc())
                .all()
            )

            results = (
                session.query(ExperimentResult)
                .filter_by(experiment_id=experiment_id)
                .order_by(ExperimentResult.recorded_at.asc())
                .all()
            )

            return {
                "experiment": experiment.to_dict(),
                "interactions": [interaction.to_dict() for interaction in interactions],
                "results": [result.to_dict() for result in results],
                "summary": {
                    "total_interactions": len(interactions),
                    "total_results": len(results),
                    "duration_seconds": self._calculate_experiment_duration(experiment),
                    "has_results": len(results) > 0,
                    "latest_taste_score": results[-1].taste_score if results else None,
                },
            }

    def _calculate_experiment_duration(self, experiment: Experiment) -> Optional[int]:
        """Calculate experiment duration in seconds."""
        if experiment.started_at and experiment.completed_at:
            return int(
                (experiment.completed_at - experiment.started_at).total_seconds()
            )
        return None

    def get_user_experiment_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user's experiments."""
        with self.db_manager.get_session_context() as session:
            experiments = session.query(Experiment).filter_by(user_id=user_id).all()
            results = session.query(ExperimentResult).filter_by(user_id=user_id).all()

            # Calculate statistics
            total_experiments = len(experiments)
            completed_experiments = len(
                [e for e in experiments if e.status == "completed"]
            )
            avg_taste_score = None

            if results:
                taste_scores = [
                    r.taste_score for r in results if r.taste_score is not None
                ]
                if taste_scores:
                    avg_taste_score = sum(taste_scores) / len(taste_scores)

            status_counts = {}
            for experiment in experiments:
                status_counts[experiment.status] = (
                    status_counts.get(experiment.status, 0) + 1
                )

            return {
                "user_id": user_id,
                "total_experiments": total_experiments,
                "completed_experiments": completed_experiments,
                "completion_rate": (
                    completed_experiments / total_experiments
                    if total_experiments > 0
                    else 0
                ),
                "average_taste_score": avg_taste_score,
                "status_distribution": status_counts,
                "total_results": len(results),
            }


# Convenience functions for common operations


def create_experiment_tracker(database_url: Optional[str] = None) -> ExperimentTracker:
    """Create an experiment tracker with optional custom database URL."""
    if database_url:
        db_manager = DatabaseManager(database_url)
    else:
        db_manager = get_database_manager()

    return ExperimentTracker(db_manager)


def quick_create_experiment(
    user_id: str,
    brew_method: str,
    coffee_dose: float,
    water_amount: float,
    water_temperature: float,
    grind_size: str,
    brew_time: int,
    **kwargs,
) -> Experiment:
    """Quick function to create an experiment with minimal parameters."""
    tracker = create_experiment_tracker()
    return tracker.create_experiment(
        user_id=user_id,
        brew_method=brew_method,
        coffee_dose=coffee_dose,
        water_amount=water_amount,
        water_temperature=water_temperature,
        grind_size=grind_size,
        brew_time=brew_time,
        **kwargs,
    )
