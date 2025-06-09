"""
Plotly Dash dashboard for experiment tracking analytics.

Provides interactive visualizations for completion rates, timing metrics,
prediction accuracy, and brewing method performance.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from .analytics import Analytics, BrewingMethodStats, ExperimentStats
from .database import get_database_manager

# Initialize the Dash app
app = dash.Dash(__name__, title="CoffeRL Analytics Dashboard")

# Get database manager
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///cofferl.db")
db_manager = get_database_manager(DATABASE_URL)
analytics = Analytics(db_manager)

# Dashboard layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("CoffeRL Analytics Dashboard", className="dashboard-title"),
                html.P(
                    "Real-time insights into coffee brewing experiments",
                    className="dashboard-subtitle",
                ),
            ],
            className="header",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Time Period:"),
                        dcc.Dropdown(
                            id="time-period-dropdown",
                            options=[
                                {"label": "All Time", "value": None},
                                {"label": "Last 7 Days", "value": 7},
                                {"label": "Last 30 Days", "value": 30},
                                {"label": "Last 90 Days", "value": 90},
                            ],
                            value=30,
                            className="dropdown",
                        ),
                    ],
                    className="filter-item",
                ),
                html.Div(
                    [
                        html.Label("Brewing Method:"),
                        dcc.Dropdown(
                            id="brewing-method-dropdown",
                            options=[
                                {"label": "All Methods", "value": None},
                                {"label": "V60", "value": "V60"},
                                {"label": "Espresso", "value": "Espresso"},
                                {"label": "French Press", "value": "French Press"},
                                {"label": "AeroPress", "value": "AeroPress"},
                            ],
                            value=None,
                            className="dropdown",
                        ),
                    ],
                    className="filter-item",
                ),
            ],
            className="filters",
        ),
        # Key metrics cards
        html.Div(id="metrics-cards", className="metrics-container"),
        # Charts section
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="completion-rate-chart")], className="chart-container"
                ),
                html.Div(
                    [dcc.Graph(id="brewing-method-performance")],
                    className="chart-container",
                ),
            ],
            className="charts-row",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="prediction-accuracy-chart")],
                    className="chart-container",
                ),
                html.Div(
                    [dcc.Graph(id="user-engagement-chart")], className="chart-container"
                ),
            ],
            className="charts-row",
        ),
        # Auto-refresh interval
        dcc.Interval(
            id="interval-component",
            interval=30 * 1000,
            n_intervals=0,  # Update every 30 seconds
        ),
    ],
    className="dashboard-container",
)


@callback(
    Output("metrics-cards", "children"),
    [
        Input("time-period-dropdown", "value"),
        Input("brewing-method-dropdown", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_metrics_cards(
    days_back: Optional[int], brewing_method: Optional[str], n_intervals: int
):
    """Update the key metrics cards."""
    try:
        # Get overall statistics
        stats = analytics.get_experiment_statistics(days_back)

        # Get filtered completion rate
        completion_rate = analytics.calculate_completion_rate(days_back, brewing_method)

        # Get average completion time
        avg_time = analytics.calculate_average_completion_time(
            days_back, brewing_method
        )
        avg_time_str = f"{avg_time:.1f}h" if avg_time else "N/A"

        # Get prediction accuracy
        accuracy = analytics.calculate_prediction_accuracy(days_back, brewing_method)
        accuracy_str = f"{accuracy:.2f}" if accuracy else "N/A"

        cards = [
            html.Div(
                [html.H3(f"{stats.total_experiments}"), html.P("Total Experiments")],
                className="metric-card",
            ),
            html.Div(
                [html.H3(f"{completion_rate:.1f}%"), html.P("Completion Rate")],
                className="metric-card",
            ),
            html.Div(
                [html.H3(avg_time_str), html.P("Avg. Completion Time")],
                className="metric-card",
            ),
            html.Div(
                [html.H3(accuracy_str), html.P("Prediction Accuracy")],
                className="metric-card",
            ),
        ]

        return cards

    except Exception as e:
        return [html.Div(f"Error loading metrics: {str(e)}", className="error-message")]


@callback(
    Output("completion-rate-chart", "figure"),
    [
        Input("time-period-dropdown", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_completion_rate_chart(days_back: Optional[int], n_intervals: int):
    """Update completion rate trend chart."""
    try:
        # Generate data for the last 30 days or specified period
        end_date = datetime.now()
        period_days = days_back or 30
        start_date = end_date - timedelta(days=period_days)

        # Calculate completion rates for each day
        dates = []
        rates = []

        for i in range(period_days):
            date = start_date + timedelta(days=i)
            # Calculate completion rate for experiments created up to this date
            rate = analytics.calculate_completion_rate(days_back=(period_days - i))
            dates.append(date)
            rates.append(rate)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rates,
                mode="lines+markers",
                name="Completion Rate",
                line=dict(color="#2E86AB", width=3),
                marker=dict(size=6),
            )
        )

        fig.update_layout(
            title="Completion Rate Trend",
            xaxis_title="Date",
            yaxis_title="Completion Rate (%)",
            template="plotly_white",
            height=400,
        )

        return fig

    except Exception as e:
        return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)


@callback(
    Output("brewing-method-performance", "figure"),
    [
        Input("time-period-dropdown", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_brewing_method_chart(days_back: Optional[int], n_intervals: int):
    """Update brewing method performance chart."""
    try:
        performance_stats = analytics.get_brewing_method_performance(days_back)

        if not performance_stats:
            return go.Figure().add_annotation(text="No data available", showarrow=False)

        methods = [stat.method for stat in performance_stats]
        completion_rates = [stat.completion_rate for stat in performance_stats]
        total_experiments = [stat.total_experiments for stat in performance_stats]
        avg_taste_scores = [stat.average_taste_score or 0 for stat in performance_stats]

        fig = go.Figure()

        # Add completion rate bars
        fig.add_trace(
            go.Bar(
                name="Completion Rate (%)",
                x=methods,
                y=completion_rates,
                yaxis="y",
                marker_color="#A23B72",
            )
        )

        # Add average taste score line
        fig.add_trace(
            go.Scatter(
                name="Avg Taste Score",
                x=methods,
                y=avg_taste_scores,
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#F18F01", width=3),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            title="Brewing Method Performance",
            xaxis_title="Brewing Method",
            yaxis=dict(title="Completion Rate (%)", side="left"),
            yaxis2=dict(title="Average Taste Score", side="right", overlaying="y"),
            template="plotly_white",
            height=400,
            legend=dict(x=0.01, y=0.99),
        )

        return fig

    except Exception as e:
        return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)


@callback(
    Output("prediction-accuracy-chart", "figure"),
    [
        Input("time-period-dropdown", "value"),
        Input("brewing-method-dropdown", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_prediction_accuracy_chart(
    days_back: Optional[int], brewing_method: Optional[str], n_intervals: int
):
    """Update prediction accuracy scatter plot."""
    try:
        # This would ideally get actual predicted vs actual data points
        # For now, we'll create a placeholder showing the correlation
        accuracy = analytics.calculate_prediction_accuracy(days_back, brewing_method)

        if accuracy is None:
            return go.Figure().add_annotation(
                text="Insufficient data for prediction accuracy", showarrow=False
            )

        # Create a sample scatter plot showing the correlation
        # In a real implementation, you'd fetch the actual predicted vs actual values
        import numpy as np

        np.random.seed(42)
        n_points = 50
        predicted = np.random.normal(8, 1, n_points)
        actual = predicted + np.random.normal(0, 0.5, n_points) * (1 - abs(accuracy))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=predicted,
                y=actual,
                mode="markers",
                name="Experiments",
                marker=dict(size=8, color="#C73E1D", opacity=0.7),
            )
        )

        # Add perfect prediction line
        min_val, max_val = min(min(predicted), min(actual)), max(
            max(predicted), max(actual)
        )
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(dash="dash", color="gray"),
            )
        )

        fig.update_layout(
            title=f"Prediction Accuracy (r = {accuracy:.3f})",
            xaxis_title="Predicted Taste Score",
            yaxis_title="Actual Taste Score",
            template="plotly_white",
            height=400,
        )

        return fig

    except Exception as e:
        return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)


@callback(
    Output("user-engagement-chart", "figure"),
    [
        Input("time-period-dropdown", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_user_engagement_chart(days_back: Optional[int], n_intervals: int):
    """Update user engagement metrics chart."""
    try:
        engagement_metrics = analytics.get_user_engagement_metrics(days_back)

        if not engagement_metrics["action_distribution"]:
            return go.Figure().add_annotation(
                text="No interaction data available", showarrow=False
            )

        actions = list(engagement_metrics["action_distribution"].keys())
        counts = list(engagement_metrics["action_distribution"].values())

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=actions,
                    values=counts,
                    hole=0.4,
                    marker_colors=[
                        "#2E86AB",
                        "#A23B72",
                        "#F18F01",
                        "#C73E1D",
                        "#6A994E",
                    ],
                )
            ]
        )

        fig.update_layout(
            title="User Interaction Distribution",
            template="plotly_white",
            height=400,
            annotations=[
                dict(
                    text=f"Total: {engagement_metrics['total_interactions']}",
                    x=0.5,
                    y=0.5,
                    font_size=16,
                    showarrow=False,
                )
            ],
        )

        return fig

    except Exception as e:
        return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)


# CSS styling
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
            }
            .dashboard-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, #2E86AB, #A23B72);
                color: white;
                border-radius: 10px;
            }
            .dashboard-title {
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }
            .dashboard-subtitle {
                margin: 10px 0 0 0;
                font-size: 1.1em;
                opacity: 0.9;
            }
            .filters {
                display: flex;
                gap: 20px;
                margin-bottom: 30px;
                padding: 20px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .filter-item {
                flex: 1;
            }
            .filter-item label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
                color: #333;
            }
            .dropdown {
                width: 100%;
            }
            .metrics-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 4px solid #2E86AB;
            }
            .metric-card h3 {
                margin: 0;
                font-size: 2em;
                color: #2E86AB;
                font-weight: 600;
            }
            .metric-card p {
                margin: 5px 0 0 0;
                color: #666;
                font-weight: 500;
            }
            .charts-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }
            .chart-container {
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 10px;
            }
            .error-message {
                color: #C73E1D;
                text-align: center;
                padding: 20px;
                background: #fff5f5;
                border-radius: 5px;
                border: 1px solid #fed7d7;
            }
            @media (max-width: 768px) {
                .charts-row {
                    grid-template-columns: 1fr;
                }
                .filters {
                    flex-direction: column;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def run_dashboard(host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
    """Run the dashboard server."""
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard(debug=True)
