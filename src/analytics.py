"""
Analytics module for experiment tracking and performance metrics.

Provides functions to calculate completion rates, timing metrics, and correlation
analysis for the coffee brewing experiment system.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, case, func

from .database import DatabaseManager, Experiment, ExperimentResult, UserInteraction


@dataclass
class ExperimentStats:
    """Container for experiment statistics."""

    total_experiments: int
    completed_experiments: int
    completion_rate: float
    average_completion_time_hours: Optional[float]
    prediction_accuracy: Optional[float]
    most_popular_method: Optional[str]


@dataclass
class BrewingMethodStats:
    """Container for brewing method performance statistics."""

    method: str
    total_experiments: int
    completion_rate: float
    average_taste_score: Optional[float]
    average_extraction_yield: Optional[float]


class Analytics:
    """Analytics engine for experiment tracking system."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize analytics with database manager."""
        self.db_manager = db_manager

    def calculate_completion_rate(
        self, days_back: Optional[int] = None, brewing_method: Optional[str] = None
    ) -> float:
        """
        Calculate the percentage of experiments that reach 'completed' status.

        Args:
            days_back: Only consider experiments from the last N days
            brewing_method: Filter by specific brewing method

        Returns:
            Completion rate as percentage (0.0 to 100.0)
        """
        with self.db_manager.get_session() as session:
            query = session.query(Experiment)

            # Apply time filter if specified
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(Experiment.created_at >= cutoff_date)

            # Apply brewing method filter if specified
            if brewing_method:
                query = query.filter(Experiment.brew_method == brewing_method)

            total_experiments = query.count()
            if total_experiments == 0:
                return 0.0

            completed_experiments = query.filter(
                Experiment.status == "completed"
            ).count()

            return (completed_experiments / total_experiments) * 100.0

    def calculate_average_completion_time(
        self, days_back: Optional[int] = None, brewing_method: Optional[str] = None
    ) -> Optional[float]:
        """
        Calculate average time from 'running' to 'completed' status in hours.

        Args:
            days_back: Only consider experiments from the last N days
            brewing_method: Filter by specific brewing method

        Returns:
            Average completion time in hours, or None if no completed experiments
        """
        with self.db_manager.get_session() as session:
            query = session.query(Experiment).filter(
                and_(
                    Experiment.status == "completed",
                    Experiment.started_at.isnot(None),
                    Experiment.completed_at.isnot(None),
                )
            )

            # Apply time filter if specified
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(Experiment.created_at >= cutoff_date)

            # Apply brewing method filter if specified
            if brewing_method:
                query = query.filter(Experiment.brew_method == brewing_method)

            experiments = query.all()
            if not experiments:
                return None

            total_hours = 0.0
            for exp in experiments:
                duration = exp.completed_at - exp.started_at
                total_hours += duration.total_seconds() / 3600.0

            return total_hours / len(experiments)

    def calculate_prediction_accuracy(
        self, days_back: Optional[int] = None, brewing_method: Optional[str] = None
    ) -> Optional[float]:
        """
        Calculate correlation between predicted and actual taste scores.

        Args:
            days_back: Only consider experiments from the last N days
            brewing_method: Filter by specific brewing method

        Returns:
            Pearson correlation coefficient (-1.0 to 1.0), or None if insufficient data
        """
        with self.db_manager.get_session() as session:
            query = (
                session.query(Experiment.predicted_score, ExperimentResult.taste_score)
                .join(ExperimentResult, Experiment.id == ExperimentResult.experiment_id)
                .filter(
                    and_(
                        Experiment.predicted_score.isnot(None),
                        ExperimentResult.taste_score.isnot(None),
                    )
                )
            )

            # Apply time filter if specified
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(Experiment.created_at >= cutoff_date)

            # Apply brewing method filter if specified
            if brewing_method:
                query = query.filter(Experiment.brew_method == brewing_method)

            results = query.all()
            if len(results) < 2:
                return None

            predicted_scores = [r[0] for r in results]
            actual_scores = [r[1] for r in results]

            return self._calculate_correlation(predicted_scores, actual_scores)

    def get_experiment_statistics(
        self, days_back: Optional[int] = None
    ) -> ExperimentStats:
        """
        Get comprehensive experiment statistics.

        Args:
            days_back: Only consider experiments from the last N days

        Returns:
            ExperimentStats object with key metrics
        """
        with self.db_manager.get_session() as session:
            query = session.query(Experiment)

            # Apply time filter if specified
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(Experiment.created_at >= cutoff_date)

            total_experiments = query.count()
            completed_experiments = query.filter(
                Experiment.status == "completed"
            ).count()

            completion_rate = 0.0
            if total_experiments > 0:
                completion_rate = (completed_experiments / total_experiments) * 100.0

            # Get most popular brewing method
            method_counts = session.query(
                Experiment.brew_method, func.count(Experiment.id).label("count")
            ).group_by(Experiment.brew_method)

            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                method_counts = method_counts.filter(
                    Experiment.created_at >= cutoff_date
                )

            method_counts = method_counts.order_by(
                func.count(Experiment.id).desc()
            ).first()
            most_popular_method = method_counts[0] if method_counts else None

            return ExperimentStats(
                total_experiments=total_experiments,
                completed_experiments=completed_experiments,
                completion_rate=completion_rate,
                average_completion_time_hours=self.calculate_average_completion_time(
                    days_back
                ),
                prediction_accuracy=self.calculate_prediction_accuracy(days_back),
                most_popular_method=most_popular_method,
            )

    def get_brewing_method_performance(
        self, days_back: Optional[int] = None
    ) -> List[BrewingMethodStats]:
        """
        Get performance statistics by brewing method.

        Args:
            days_back: Only consider experiments from the last N days

        Returns:
            List of BrewingMethodStats objects
        """
        with self.db_manager.get_session() as session:
            # Get basic stats by method
            query = session.query(
                Experiment.brew_method,
                func.count(Experiment.id).label("total"),
                func.count(case([(Experiment.status == "completed", 1)])).label(
                    "completed"
                ),
            ).group_by(Experiment.brew_method)

            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(Experiment.created_at >= cutoff_date)

            method_stats = query.all()

            results = []
            for method, total, completed in method_stats:
                completion_rate = (completed / total * 100.0) if total > 0 else 0.0

                # Get average taste score for this method
                taste_query = (
                    session.query(func.avg(ExperimentResult.taste_score))
                    .join(Experiment, ExperimentResult.experiment_id == Experiment.id)
                    .filter(Experiment.brew_method == method)
                )

                if days_back:
                    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                    taste_query = taste_query.filter(
                        Experiment.created_at >= cutoff_date
                    )

                avg_taste_score = taste_query.scalar()

                # Get average extraction yield for this method
                yield_query = (
                    session.query(func.avg(ExperimentResult.extraction_yield))
                    .join(Experiment, ExperimentResult.experiment_id == Experiment.id)
                    .filter(Experiment.brew_method == method)
                )

                if days_back:
                    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                    yield_query = yield_query.filter(
                        Experiment.created_at >= cutoff_date
                    )

                avg_extraction_yield = yield_query.scalar()

                results.append(
                    BrewingMethodStats(
                        method=method,
                        total_experiments=total,
                        completion_rate=completion_rate,
                        average_taste_score=avg_taste_score,
                        average_extraction_yield=avg_extraction_yield,
                    )
                )

            return sorted(results, key=lambda x: x.total_experiments, reverse=True)

    def get_user_engagement_metrics(
        self, days_back: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate user engagement metrics from interaction data.

        Args:
            days_back: Only consider interactions from the last N days

        Returns:
            Dictionary with engagement metrics
        """
        with self.db_manager.get_session() as session:
            query = session.query(UserInteraction)

            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(UserInteraction.timestamp >= cutoff_date)

            total_interactions = query.count()

            # Count interactions by action type
            action_counts = session.query(
                UserInteraction.interaction_type,
                func.count(UserInteraction.id).label("count"),
            ).group_by(UserInteraction.interaction_type)

            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                action_counts = action_counts.filter(
                    UserInteraction.timestamp >= cutoff_date
                )

            action_distribution = {
                action: count for action, count in action_counts.all()
            }

            # Calculate average interactions per experiment
            experiment_count = session.query(Experiment).count()
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                experiment_count = (
                    session.query(Experiment)
                    .filter(Experiment.created_at >= cutoff_date)
                    .count()
                )

            avg_interactions_per_experiment = (
                total_interactions / experiment_count if experiment_count > 0 else 0.0
            )

            return {
                "total_interactions": total_interactions,
                "action_distribution": action_distribution,
                "average_interactions_per_experiment": avg_interactions_per_experiment,
            }

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient between two lists.

        Args:
            x: First variable values
            y: Second variable values

        Returns:
            Correlation coefficient (-1.0 to 1.0)
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = (
            (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
        ) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator
