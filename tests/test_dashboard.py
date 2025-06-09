"""
Tests for the Plotly Dash dashboard.

Tests dashboard components, callbacks, and data visualization functionality.
"""

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.analytics import BrewingMethodStats, ExperimentStats
from src.dashboard import analytics, app


class TestDashboardComponents:
    """Test dashboard layout and component structure."""

    def test_app_initialization(self):
        """Test that the Dash app initializes correctly."""
        assert app.title == "CoffeRL Analytics Dashboard"
        assert app.layout is not None

    def test_layout_structure(self):
        """Test that the layout contains expected components."""
        layout = app.layout

        # Check that layout is a Div
        assert hasattr(layout, "children")

        # Check for main sections
        children = layout.children
        assert len(children) >= 4  # header, filters, metrics, charts

    def test_filter_components(self):
        """Test that filter dropdowns are properly configured."""
        layout = app.layout

        # Find dropdown components
        dropdowns = []

        def find_dropdowns(component):
            if hasattr(component, "id"):
                if component.id in ["time-period-dropdown", "brewing-method-dropdown"]:
                    dropdowns.append(component)
            if hasattr(component, "children"):
                if isinstance(component.children, list):
                    for child in component.children:
                        find_dropdowns(child)
                else:
                    find_dropdowns(component.children)

        find_dropdowns(layout)
        assert len(dropdowns) == 2

        # Check time period dropdown options
        time_dropdown = next(d for d in dropdowns if d.id == "time-period-dropdown")
        assert len(time_dropdown.options) == 4
        assert time_dropdown.value == 30

        # Check brewing method dropdown options
        method_dropdown = next(
            d for d in dropdowns if d.id == "brewing-method-dropdown"
        )
        assert len(method_dropdown.options) == 5
        assert method_dropdown.value is None


class TestDashboardCallbacks:
    """Test dashboard callback functions."""

    @pytest.fixture
    def mock_analytics(self):
        """Mock analytics for testing."""
        with patch("src.dashboard.analytics") as mock:
            # Mock experiment statistics
            mock.get_experiment_statistics.return_value = ExperimentStats(
                total_experiments=10,
                completed_experiments=8,
                completion_rate=80.0,
                average_completion_time_hours=2.5,
                prediction_accuracy=0.85,
                most_popular_method="V60",
            )

            # Mock other methods
            mock.calculate_completion_rate.return_value = 80.0
            mock.calculate_average_completion_time.return_value = 2.5
            mock.calculate_prediction_accuracy.return_value = 0.85
            mock.get_brewing_method_performance.return_value = [
                BrewingMethodStats(
                    method="V60",
                    total_experiments=5,
                    completion_rate=80.0,
                    average_taste_score=8.2,
                    average_extraction_yield=18.5,
                ),
                BrewingMethodStats(
                    method="Espresso",
                    total_experiments=3,
                    completion_rate=66.7,
                    average_taste_score=7.8,
                    average_extraction_yield=20.2,
                ),
            ]
            mock.get_user_engagement_metrics.return_value = {
                "total_interactions": 25,
                "unique_users": 3,
                "action_distribution": {
                    "create": 10,
                    "update": 8,
                    "complete": 5,
                    "view": 2,
                },
            }
            yield mock

    def test_update_metrics_cards(self, mock_analytics):
        """Test metrics cards callback."""
        from src.dashboard import update_metrics_cards

        # Test with valid inputs
        result = update_metrics_cards(30, "V60", 0)

        assert len(result) == 4
        assert all(hasattr(card, "children") for card in result)

        # Check that analytics methods were called
        mock_analytics.get_experiment_statistics.assert_called_once_with(30)
        mock_analytics.calculate_completion_rate.assert_called_once_with(30, "V60")
        mock_analytics.calculate_average_completion_time.assert_called_once_with(
            30, "V60"
        )
        mock_analytics.calculate_prediction_accuracy.assert_called_once_with(30, "V60")

    def test_update_metrics_cards_error_handling(self):
        """Test metrics cards error handling."""
        from src.dashboard import update_metrics_cards

        with patch("src.dashboard.analytics") as mock_analytics:
            mock_analytics.get_experiment_statistics.side_effect = Exception(
                "Database error"
            )

            result = update_metrics_cards(30, None, 0)

            assert len(result) == 1
            assert "Error loading metrics" in str(result[0].children)

    def test_update_completion_rate_chart(self, mock_analytics):
        """Test completion rate chart callback."""
        from src.dashboard import update_completion_rate_chart

        result = update_completion_rate_chart(7, 0)

        assert result is not None
        assert hasattr(result, "data")
        assert result.layout.title.text == "Completion Rate Trend"
        assert result.layout.xaxis.title.text == "Date"
        assert result.layout.yaxis.title.text == "Completion Rate (%)"

    def test_update_brewing_method_chart(self, mock_analytics):
        """Test brewing method performance chart callback."""
        from src.dashboard import update_brewing_method_chart

        result = update_brewing_method_chart(30, 0)

        assert result is not None
        assert hasattr(result, "data")
        assert len(result.data) == 2  # Bar chart + line chart
        assert result.layout.title.text == "Brewing Method Performance"

    def test_update_brewing_method_chart_no_data(self, mock_analytics):
        """Test brewing method chart with no data."""
        from src.dashboard import update_brewing_method_chart

        mock_analytics.get_brewing_method_performance.return_value = []

        result = update_brewing_method_chart(30, 0)

        assert result is not None
        # Should show "No data available" annotation
        assert len(result.layout.annotations) > 0

    def test_update_prediction_accuracy_chart(self, mock_analytics):
        """Test prediction accuracy chart callback."""
        from src.dashboard import update_prediction_accuracy_chart

        result = update_prediction_accuracy_chart(30, "V60", 0)

        assert result is not None
        assert hasattr(result, "data")
        assert len(result.data) == 2  # Scatter plot + perfect prediction line
        assert "Prediction Accuracy" in result.layout.title.text

    def test_update_prediction_accuracy_chart_no_data(self, mock_analytics):
        """Test prediction accuracy chart with insufficient data."""
        from src.dashboard import update_prediction_accuracy_chart

        mock_analytics.calculate_prediction_accuracy.return_value = None

        result = update_prediction_accuracy_chart(30, None, 0)

        assert result is not None
        assert len(result.layout.annotations) > 0
        assert "Insufficient data" in result.layout.annotations[0].text

    def test_update_user_engagement_chart(self, mock_analytics):
        """Test user engagement chart callback."""
        from src.dashboard import update_user_engagement_chart

        result = update_user_engagement_chart(30, 0)

        assert result is not None
        assert hasattr(result, "data")
        assert len(result.data) == 1  # Pie chart
        assert result.layout.title.text == "User Interaction Distribution"

    def test_update_user_engagement_chart_no_data(self, mock_analytics):
        """Test user engagement chart with no interaction data."""
        from src.dashboard import update_user_engagement_chart

        mock_analytics.get_user_engagement_metrics.return_value = {
            "total_interactions": 0,
            "unique_users": 0,
            "action_distribution": {},
        }

        result = update_user_engagement_chart(30, 0)

        assert result is not None
        assert len(result.layout.annotations) > 0
        assert "No interaction data" in result.layout.annotations[0].text


class TestDashboardIntegration:
    """Test dashboard integration with analytics system."""

    def test_database_connection(self):
        """Test that dashboard connects to database properly."""
        from src.dashboard import analytics, db_manager

        assert db_manager is not None
        assert analytics is not None
        assert hasattr(analytics, "db_manager")

    def test_environment_variable_handling(self):
        """Test that dashboard handles DATABASE_URL environment variable."""
        # Test default SQLite URL
        with patch.dict(os.environ, {}, clear=True):
            from src.dashboard import DATABASE_URL

            assert "sqlite:///" in DATABASE_URL

        # Test custom DATABASE_URL
        custom_url = "postgresql://user:pass@localhost/test"
        with patch.dict(os.environ, {"DATABASE_URL": custom_url}):
            # Reload the module to pick up new environment variable
            import importlib

            import src.dashboard

            importlib.reload(src.dashboard)
            assert src.dashboard.DATABASE_URL == custom_url


class TestDashboardUtilities:
    """Test dashboard utility functions."""

    def test_run_dashboard_function(self):
        """Test the run_dashboard function."""
        from src.dashboard import run_dashboard

        with patch.object(app, "run") as mock_run:
            run_dashboard(host="0.0.0.0", port=8080, debug=True)
            mock_run.assert_called_once_with(host="0.0.0.0", port=8080, debug=True)

    def test_run_dashboard_defaults(self):
        """Test run_dashboard with default parameters."""
        from src.dashboard import run_dashboard

        with patch.object(app, "run") as mock_run:
            run_dashboard()
            mock_run.assert_called_once_with(host="127.0.0.1", port=8050, debug=False)


class TestDashboardErrorHandling:
    """Test dashboard error handling and edge cases."""

    def test_callback_error_handling(self):
        """Test that callbacks handle errors gracefully."""
        from src.dashboard import update_metrics_cards

        with patch("src.dashboard.analytics") as mock_analytics:
            mock_analytics.get_experiment_statistics.side_effect = Exception(
                "Test error"
            )

            result = update_metrics_cards(None, None, 0)

            # Should return error message instead of crashing
            assert len(result) == 1
            assert "Error loading metrics" in str(result[0].children)

    def test_chart_error_handling(self):
        """Test that chart callbacks handle errors gracefully."""
        from src.dashboard import update_completion_rate_chart

        with patch("src.dashboard.analytics") as mock_analytics:
            mock_analytics.calculate_completion_rate.side_effect = Exception(
                "Chart error"
            )

            result = update_completion_rate_chart(30, 0)

            # Should return figure with error annotation
            assert result is not None
            assert len(result.layout.annotations) > 0
            assert "Error:" in result.layout.annotations[0].text

    @pytest.fixture
    def mock_analytics_simple(self):
        """Simple mock analytics for error handling tests."""
        with patch("src.dashboard.analytics") as mock:
            mock.get_experiment_statistics.return_value = ExperimentStats(
                total_experiments=5,
                completed_experiments=3,
                completion_rate=60.0,
                average_completion_time_hours=1.5,
                prediction_accuracy=0.75,
                most_popular_method="V60",
            )
            mock.calculate_completion_rate.return_value = 60.0
            yield mock

    def test_invalid_filter_values(self, mock_analytics_simple):
        """Test dashboard behavior with invalid filter values."""
        from src.dashboard import update_metrics_cards

        # Test with negative days
        result = update_metrics_cards(-5, "InvalidMethod", 0)

        # Should still work (analytics should handle invalid values)
        assert len(result) == 4

        # Verify analytics was called with the invalid values
        mock_analytics_simple.get_experiment_statistics.assert_called_with(-5)
        mock_analytics_simple.calculate_completion_rate.assert_called_with(
            -5, "InvalidMethod"
        )


if __name__ == "__main__":
    pytest.main([__file__])
