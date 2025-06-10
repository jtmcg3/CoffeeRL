import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup to avoid import issues
from config.platform_config import detect_platform, get_platform_settings  # noqa: E402
from src.database import DatabaseManager, Experiment, ExperimentResult  # noqa: E402
from src.experiment_generator import ExperimentGenerator  # noqa: E402
from src.experiment_tracker import ExperimentTracker  # noqa: E402
from src.kdtree_explorer import SimpleKDTreeExplorer  # noqa: E402
from src.parameter_space import BrewMethod  # noqa: E402

# Global variables to store loaded model and experiment components
model = None
tokenizer = None
platform_info = None
model_size = None
experiment_generator = None
experiment_tracker = None


def initialize_experiment_system():
    """Initialize the experiment generation and tracking system."""
    global experiment_generator, experiment_tracker

    try:
        # Initialize database with SQLite
        sqlite_url = "sqlite:///cofferl.db"
        db_manager = DatabaseManager(sqlite_url)

        # Create tables if they don't exist
        db_manager.create_tables()

        # Initialize experiment tracker
        experiment_tracker = ExperimentTracker(db_manager)

        # Initialize experiment generator with k-d tree explorer
        kdtree_explorer = SimpleKDTreeExplorer()
        experiment_generator = ExperimentGenerator(kdtree_explorer=kdtree_explorer)

        print("✅ Experiment system initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Error initializing experiment system: {e}")
        return False


def format_number(value: float, decimals: int = 0) -> str:
    """Format numbers with specified decimal places."""
    if decimals == 0:
        return f"{value:.0f}"
    else:
        return f"{value:.{decimals}f}"


def generate_experiment_suggestions(num_experiments: int = 2) -> List[Dict]:
    """Generate experiment suggestions using the experiment generator."""
    if not experiment_generator or not experiment_tracker:
        return []

    try:
        # Get existing experiments to inform generation
        existing_data = []
        with experiment_tracker.db_manager.get_session_context() as session:
            existing_experiments = session.query(Experiment).all()
            for exp in existing_experiments:
                exp_dict = {
                    "water_temp": exp.water_temperature,
                    "coffee_dose": exp.coffee_dose,
                    "water_amount": exp.water_amount,
                    "grind_size": exp.grind_size,
                    "brew_time": exp.brew_time,
                    "brew_method": exp.brew_method,
                }
                if exp.pressure:
                    exp_dict["pressure"] = exp.pressure
                if exp.bloom_time:
                    exp_dict["bloom_time"] = exp.bloom_time
                existing_data.append(exp_dict)

        # Generate new experiments
        suggested_params = experiment_generator.generate_exploration_based_experiments(
            num_experiments=num_experiments,
            brew_method=BrewMethod.POUR_OVER,
            existing_experiments=existing_data,
        )

        # Convert to experiment dictionaries
        suggestions = []
        for i, params in enumerate(suggested_params):
            suggestion = {
                "id": f"generated_{i + 1}",
                "title": f"Exploration Experiment {i + 1}",
                "description": "AI-generated experiment targeting unexplored parameter regions",
                "brew_method": "V60",
                "difficulty": "Medium",
                "estimated_time": "15 minutes",
                "reward_points": 50,
                "parameters": {
                    "coffee_dose": format_number(params.coffee_dose, 0),
                    "water_amount": format_number(params.water_amount, 0),
                    "water_temperature": format_number(params.water_temp, 1),
                    "grind_size": params.grind_size.value.replace("_", "-"),
                    "brew_time": format_number(params.brew_time, 0),
                },
                "scientific_rationale": "This experiment targets a sparse region in the parameter space to maximize information gain and explore potentially optimal brewing conditions.",
                "status": "pending",
                "brewing_params": params,  # Store the actual BrewingParameters object
            }
            if params.bloom_time:
                suggestion["parameters"]["bloom_time"] = format_number(
                    params.bloom_time, 0
                )

            suggestions.append(suggestion)

        return suggestions

    except Exception as e:
        print(f"Error generating experiments: {e}")
        return []


def create_custom_experiment_card() -> str:
    """Create HTML for the custom experiment submission card with beautiful styling."""
    return """
    <div class="custom-experiment-card" style="
        border: 2px dashed #007bff;
        border-radius: 16px;
        padding: 24px;
        margin: 12px;
        background: linear-gradient(135deg, #f8fbff 0%, #e3f2fd 100%);
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.08);
        transition: all 0.3s ease;
        cursor: pointer;
        max-width: 420px;
        width: 100%;
        position: relative;
        overflow: hidden;
        text-align: center;
    " onmouseover="
        this.style.transform='translateY(-4px)';
        this.style.boxShadow='0 8px 25px rgba(0, 123, 255, 0.15)';
        this.style.borderColor='#0056b3';
        this.style.background='linear-gradient(135deg, #f0f8ff 0%, #dbeafe 100%)';
    " onmouseout="
        this.style.transform='translateY(0)';
        this.style.boxShadow='0 4px 12px rgba(0, 123, 255, 0.08)';
        this.style.borderColor='#007bff';
        this.style.background='linear-gradient(135deg, #f8fbff 0%, #e3f2fd 100%)';
    ">

        <!-- Decorative gradient overlay -->
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #007bff 0%, #00d4aa 50%, #007bff 100%);
        "></div>

        <!-- Header badge -->
        <div style="margin-bottom: 16px;">
            <span style="
                background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                color: white;
                padding: 8px 16px;
                border-radius: 25px;
                font-size: 13px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                box-shadow: 0 3px 6px rgba(0,123,255,0.2);
                display: inline-flex;
                align-items: center;
                gap: 8px;
            ">
                <span style="font-size: 16px;">🧪</span>
                Custom Experiment
            </span>
        </div>

        <!-- Main icon -->
        <div style="
            font-size: 48px;
            margin-bottom: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">☕</div>

        <!-- Title -->
        <h3 style="
            margin: 0 0 12px 0;
            color: #1a202c;
            font-size: 22px;
            font-weight: 700;
            line-height: 1.3;
        ">Submit Your Own Experiment</h3>

        <!-- Description -->
        <p style="
            color: #4a5568;
            margin: 0 0 20px 0;
            font-size: 15px;
            line-height: 1.6;
            font-weight: 400;
        ">Have your own brewing parameters to test? Submit them here and contribute to our research database.</p>

        <!-- Features list -->
        <div style="
            background: rgba(255, 255, 255, 0.7);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
            border: 1px solid rgba(0, 123, 255, 0.1);
        ">
            <div style="
                display: flex;
                flex-direction: column;
                gap: 8px;
                text-align: left;
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    color: #2d3748;
                    font-size: 14px;
                    font-weight: 500;
                ">
                    <span style="
                        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
                        color: white;
                        width: 20px;
                        height: 20px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 12px;
                        font-weight: bold;
                    ">✓</span>
                    Enter your brewing parameters
                </div>
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    color: #2d3748;
                    font-size: 14px;
                    font-weight: 500;
                ">
                    <span style="
                        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
                        color: white;
                        width: 20px;
                        height: 20px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 12px;
                        font-weight: bold;
                    ">✓</span>
                    Conduct the experiment
                </div>
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    color: #2d3748;
                    font-size: 14px;
                    font-weight: 500;
                ">
                    <span style="
                        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
                        color: white;
                        width: 20px;
                        height: 20px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 12px;
                        font-weight: bold;
                    ">✓</span>
                    Submit your results to help improve our AI models
                </div>
            </div>
        </div>

        <!-- Call to action -->
        <div style="
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 12px 16px;
            border-radius: 12px;
            margin-top: 16px;
            border: 1px solid rgba(252, 182, 159, 0.3);
        ">
            <div style="
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                color: #8b4513;
                font-size: 14px;
                font-weight: 600;
            ">
                <span style="font-size: 18px;">🏆</span>
                <span>Contribute to Coffee Science!</span>
            </div>
        </div>
    </div>
    """


def create_experiment_card(experiment: dict) -> str:
    """Create HTML for an experiment card with beautiful, readable styling."""
    difficulty_colors = {
        "Easy": "#28a745",
        "Medium": "#ffc107",
        "Hard": "#dc3545",
    }
    difficulty_color = difficulty_colors.get(experiment["difficulty"], "#6c757d")

    status_badge = (
        "✅ Completed" if experiment["status"] == "completed" else "🔬 Available"
    )
    status_color = "#28a745" if experiment["status"] == "completed" else "#007bff"

    # Format parameters with icons and better styling
    params = experiment["parameters"]
    param_items = []

    # Parameter mapping with icons and units
    param_config = {
        "coffee_dose": {
            "icon": "☕",
            "label": "Coffee Dose",
            "unit": "g",
            "color": "#8B4513",
        },
        "water_amount": {
            "icon": "💧",
            "label": "Water Amount",
            "unit": "g",
            "color": "#4A90E2",
        },
        "water_temperature": {
            "icon": "🌡️",
            "label": "Temperature",
            "unit": "°C",
            "color": "#FF6B6B",
        },
        "grind_size": {
            "icon": "⚙️",
            "label": "Grind Size",
            "unit": "",
            "color": "#9B59B6",
        },
        "brew_time": {
            "icon": "⏱️",
            "label": "Brew Time",
            "unit": "s",
            "color": "#F39C12",
        },
        "bloom_time": {
            "icon": "🌸",
            "label": "Bloom Time",
            "unit": "s",
            "color": "#E67E22",
        },
    }

    for key, value in params.items():
        if key in param_config:
            config = param_config[key]
            param_items.append(
                f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 8px 12px;
                    margin: 4px 0;
                    background: linear-gradient(135deg, {config["color"]}15 0%, {config["color"]}08 100%);
                    border-left: 3px solid {config["color"]};
                    border-radius: 6px;
                    transition: transform 0.2s ease;
                " onmouseover="this.style.transform='translateX(2px)'" onmouseout="this.style.transform='translateX(0)'">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 16px;">{config["icon"]}</span>
                        <span style="font-weight: 500; color: #2c3e50; font-size: 14px;">{config["label"]}</span>
                    </div>
                    <div style="
                        background: white;
                        padding: 4px 8px;
                        border-radius: 12px;
                        font-weight: bold;
                        color: {config["color"]};
                        font-size: 14px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    ">
                        {value}{config["unit"]}
                    </div>
                </div>
            """
            )

    params_html = "".join(param_items)

    # Add data attributes for JavaScript interaction
    experiment_id = experiment.get("id", "unknown")

    card_html = f"""
    <div class="experiment-card" data-experiment-id="{experiment_id}" style="
        border: 1px solid #e8ecf0;
        border-radius: 16px;
        padding: 24px;
        margin: 12px;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        cursor: pointer;
        max-width: 420px;
        width: 100%;
        position: relative;
        overflow: hidden;
    " onmouseover="
        this.style.transform='translateY(-4px)';
        this.style.boxShadow='0 8px 25px rgba(0, 0, 0, 0.12)';
        this.style.borderColor='#007bff';
    " onmouseout="
        this.style.transform='translateY(0)';
        this.style.boxShadow='0 4px 12px rgba(0, 0, 0, 0.08)';
        this.style.borderColor='#e8ecf0';
    ">

        <!-- Decorative gradient overlay -->
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #007bff 0%, #6f42c1 50%, #e83e8c 100%);
        "></div>

        <!-- Header with badges -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <span style="
                background: linear-gradient(135deg, {status_color} 0%, {status_color}dd 100%);
                color: white;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">{status_badge}</span>
            <span style="
                background: linear-gradient(135deg, {difficulty_color} 0%, {difficulty_color}dd 100%);
                color: white;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">{experiment["difficulty"]}</span>
        </div>

        <!-- Title -->
        <h3 style="
            margin: 0 0 12px 0;
            color: #1a202c;
            font-size: 20px;
            font-weight: 700;
            line-height: 1.3;
        ">{experiment["title"]}</h3>

        <!-- Description -->
        <p style="
            color: #64748b;
            margin: 0 0 20px 0;
            font-size: 14px;
            line-height: 1.5;
            font-weight: 400;
        ">{experiment["description"]}</p>

        <!-- Parameters Section -->
        <div style="margin-bottom: 20px;">
            <h4 style="
                margin: 0 0 12px 0;
                color: #2d3748;
                font-size: 16px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 8px;
            ">
                <span style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                ">⚡ Brewing Parameters</span>
            </h4>
            <div style="
                background: #f8fafc;
                border-radius: 12px;
                padding: 12px;
                border: 1px solid #e2e8f0;
            ">
                {params_html}
            </div>
        </div>

        <!-- Scientific Rationale -->
        <div style="margin-bottom: 20px;">
            <h4 style="
                margin: 0 0 8px 0;
                color: #2d3748;
                font-size: 14px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 6px;
            ">
                <span style="font-size: 16px;">🧪</span>
                <span style="
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                ">Scientific Rationale</span>
            </h4>
            <div style="
                background: linear-gradient(135deg, #667eea08 0%, #764ba208 100%);
                border-radius: 10px;
                padding: 12px;
                border-left: 3px solid #667eea;
            ">
                <p style="
                    margin: 0;
                    font-size: 13px;
                    color: #4a5568;
                    line-height: 1.4;
                    font-style: italic;
                ">{experiment["scientific_rationale"]}</p>
            </div>
        </div>

        <!-- Footer with stats -->
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-top: 16px;
            border-top: 1px solid #e2e8f0;
        ">
            <div style="
                display: flex;
                align-items: center;
                gap: 6px;
                color: #64748b;
                font-size: 12px;
                font-weight: 500;
            ">
                <span style="font-size: 14px;">⏱️</span>
                <span>{experiment["estimated_time"]}</span>
            </div>
            <div style="
                display: flex;
                align-items: center;
                gap: 6px;
                background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                padding: 6px 10px;
                border-radius: 16px;
                font-size: 12px;
                font-weight: 600;
                color: #8b4513;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            ">
                <span style="font-size: 14px;">🏆</span>
                <span>{experiment["reward_points"]} points</span>
            </div>
        </div>
    </div>
    """

    return card_html


def create_experiment_grid() -> str:
    """Create a grid of experiment cards with generated suggestions."""
    # Generate experiment suggestions
    suggested_experiments = generate_experiment_suggestions(2)

    # Create cards for suggested experiments
    cards_html = []
    for experiment in suggested_experiments:
        cards_html.append(create_experiment_card(experiment))

    # Add custom experiment card
    cards_html.append(create_custom_experiment_card())

    grid_html = f"""
    <div class="experiment-grid" style="
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        padding: 20px 0;
        max-width: 1200px;
        margin: 0 auto;
    ">
        {"".join(cards_html)}
    </div>

    <style>
        @media (max-width: 768px) {{
            .experiment-grid {{
                grid-template-columns: 1fr !important;
                gap: 15px !important;
                padding: 15px 10px !important;
            }}
            .experiment-card {{
                margin: 5px !important;
                padding: 15px !important;
            }}
        }}

        .custom-card:hover {{
            border-color: #0056b3 !important;
        }}
    </style>
    """

    return grid_html


def create_experiment_from_parameters(
    user_id: str,
    coffee_dose: float,
    water_amount: float,
    water_temperature: float,
    grind_size: str,
    brew_time: int,
    bloom_time: Optional[int] = None,
    notes: str = "",
) -> Tuple[bool, str, Optional[int]]:
    """Create a new experiment from user parameters."""
    if not experiment_tracker:
        return False, "Experiment system not initialized", None

    try:
        # Validate inputs
        if not user_id or user_id.strip() == "":
            return False, "User ID is required", None

        if coffee_dose < 10 or coffee_dose > 30:
            return False, "Coffee dose must be between 10-30g", None

        if water_amount < 150 or water_amount > 500:
            return False, "Water amount must be between 150-500g", None

        if water_temperature < 80 or water_temperature > 100:
            return False, "Water temperature must be between 80-100°C", None

        if brew_time < 60 or brew_time > 480:
            return False, "Brew time must be between 60-480 seconds", None

        # Create experiment
        experiment = experiment_tracker.create_experiment(
            user_id=user_id.strip(),
            brew_method="V60",
            coffee_dose=coffee_dose,
            water_amount=water_amount,
            water_temperature=water_temperature,
            grind_size=grind_size,
            brew_time=brew_time,
            bloom_time=bloom_time,
            parameters_json={
                "source": "custom_submission",
                "notes": notes,
                "difficulty": "Custom",
                "reward_points": 50,
            },
        )

        if experiment:
            return (
                True,
                f"Experiment created successfully with ID: {experiment.id}",
                experiment.id,
            )
        else:
            return False, "Failed to create experiment", None

    except Exception as e:
        return False, f"Error creating experiment: {str(e)}", None


def submit_experiment_result(
    experiment_id: int,
    user_id: str,
    taste_score: float,
    extraction_notes: str,
    brew_time_actual: int,
    temperature_consistency: str,
    overall_satisfaction: int,
    additional_notes: str,
) -> str:
    """Submit experiment result to the database."""
    if not experiment_tracker:
        return "❌ Error: Experiment system not initialized"

    try:
        # Validate required fields
        if not user_id or user_id.strip() == "":
            return "❌ Error: User ID is required. Please enter a unique identifier."

        if not extraction_notes or extraction_notes.strip() == "":
            return "❌ Error: Extraction notes are required. Please describe your brewing process."

        if experiment_id < 1:
            return "❌ Error: Please select a valid experiment ID."

        if taste_score < 1 or taste_score > 10:
            return "❌ Error: Taste score must be between 1 and 10."

        if brew_time_actual < 30 or brew_time_actual > 600:
            return "❌ Error: Brew time must be between 30 and 600 seconds."

        # Convert temperature consistency to a score
        temp_score_map = {
            "Excellent": 5,
            "Good": 4,
            "Fair": 3,
            "Poor": 2,
            "Very Poor": 1,
        }
        temp_score = temp_score_map.get(temperature_consistency, 3)

        # Record the experiment result
        result = experiment_tracker.record_experiment_result(
            experiment_id=experiment_id,
            user_id=user_id,
            taste_score=taste_score,
            actual_brew_time=brew_time_actual,
            notes=f"Extraction notes: {extraction_notes}\nAdditional notes: {additional_notes}",
            tags=[
                "web_submission",
                f"temp_consistency_{temperature_consistency.lower().replace(' ', '_')}",
            ],
            temperature_profile={"consistency_rating": temp_score},
            is_successful=True,
        )

        if result:
            # Log the interaction
            experiment_tracker.log_user_interaction(
                experiment_id=experiment_id,
                user_id=user_id,
                interaction_type="submit_result",
                interaction_data={
                    "taste_score": taste_score,
                    "overall_satisfaction": overall_satisfaction,
                    "submission_method": "web_form",
                },
            )

            return f"""
            ✅ **Result Submitted Successfully!**

            Thank you for participating in experiment #{experiment_id}. Your results have been recorded:
            - Taste Score: {taste_score}/10
            - Overall Satisfaction: {overall_satisfaction}/10
            - Temperature Consistency: {temperature_consistency}

            Your contribution helps improve our coffee brewing models!
            """
        else:
            return "❌ Error: Could not submit result. Please check the experiment ID and try again."

    except Exception as e:
        return f"❌ Error submitting result: {str(e)}"


def create_custom_experiment_form():
    """Create the custom experiment submission form."""
    with gr.Column(visible=False) as custom_form_column:
        gr.Markdown("### 🎯 Create Custom Experiment")
        gr.Markdown("Enter your brewing parameters to create a new experiment.")

        with gr.Row():
            user_id_custom = gr.Textbox(
                label="Your User ID",
                placeholder="Enter your unique user ID",
                info="Use a consistent ID to track your experiments",
            )

        with gr.Row():
            coffee_dose_custom = gr.Number(
                label="Coffee Dose (g)",
                value=18.0,
                minimum=10.0,
                maximum=30.0,
                step=0.5,
                info="Amount of coffee in grams",
            )
            water_amount_custom = gr.Number(
                label="Water Amount (g)",
                value=300.0,
                minimum=150.0,
                maximum=500.0,
                step=5.0,
                info="Amount of water in grams",
            )

        with gr.Row():
            water_temp_custom = gr.Number(
                label="Water Temperature (°C)",
                value=92.0,
                minimum=80.0,
                maximum=100.0,
                step=0.5,
                info="Water temperature in Celsius",
            )
            grind_size_custom = gr.Dropdown(
                label="Grind Size",
                choices=[
                    "very-fine",
                    "fine",
                    "medium-fine",
                    "medium",
                    "medium-coarse",
                    "coarse",
                ],
                value="medium-fine",
                info="Coffee grind size",
            )

        with gr.Row():
            brew_time_custom = gr.Number(
                label="Brew Time (seconds)",
                value=240,
                minimum=60,
                maximum=480,
                step=5,
                info="Total brewing time in seconds",
            )
            bloom_time_custom = gr.Number(
                label="Bloom Time (seconds)",
                value=30,
                minimum=15,
                maximum=60,
                step=5,
                info="Optional: Bloom time for pour over",
            )

        notes_custom = gr.Textbox(
            label="Notes",
            placeholder="Any additional notes about this experiment...",
            lines=2,
            info="Optional: Additional information about your experiment",
        )

        with gr.Row():
            create_experiment_btn = gr.Button("Create Experiment", variant="primary")
            cancel_custom_btn = gr.Button("Cancel", variant="secondary")

        custom_result_output = gr.Markdown(value="", visible=True)

        # Form submission logic
        def handle_custom_experiment_creation(
            user_id,
            coffee_dose,
            water_amount,
            water_temp,
            grind_size,
            brew_time,
            bloom_time,
            notes,
        ):
            success, message, exp_id = create_experiment_from_parameters(
                user_id,
                coffee_dose,
                water_amount,
                water_temp,
                grind_size,
                brew_time,
                bloom_time,
                notes,
            )

            if success:
                # Hide form and show success message
                return (
                    gr.update(visible=False),
                    f"✅ {message}\n\nYou can now conduct your experiment and submit results using experiment ID: {exp_id}",
                )
            else:
                return gr.update(), f"❌ {message}"

        create_experiment_btn.click(
            fn=handle_custom_experiment_creation,
            inputs=[
                user_id_custom,
                coffee_dose_custom,
                water_amount_custom,
                water_temp_custom,
                grind_size_custom,
                brew_time_custom,
                bloom_time_custom,
                notes_custom,
            ],
            outputs=[custom_form_column, custom_result_output],
        )

        # Cancel button logic
        cancel_custom_btn.click(
            fn=lambda: (gr.update(visible=False), ""),
            outputs=[custom_form_column, custom_result_output],
        )

    return custom_form_column, custom_result_output


def create_result_submission_form():
    """Create the experiment result submission form components."""
    with gr.Column(visible=False) as form_column:
        gr.Markdown("### 📝 Submit Experiment Results")
        gr.Markdown("Please fill out all fields to submit your experiment results.")

        with gr.Row():
            experiment_id_input = gr.Number(
                label="Experiment ID",
                value=1,
                minimum=1,
                step=1,
                info="Select the experiment you completed",
            )
            user_id_input = gr.Textbox(
                label="Your User ID",
                placeholder="Enter your unique user ID",
                info="Use a consistent ID to track your experiments",
            )

        with gr.Row():
            taste_score_input = gr.Slider(
                label="Taste Score",
                minimum=1,
                maximum=10,
                step=0.1,
                value=5.0,
                info="Rate the overall taste quality (1=poor, 10=excellent)",
            )
            overall_satisfaction_input = gr.Slider(
                label="Overall Satisfaction",
                minimum=1,
                maximum=10,
                step=1,
                value=5,
                info="How satisfied are you with this brew? (1=very unsatisfied, 10=very satisfied)",
            )

        extraction_notes_input = gr.Textbox(
            label="Extraction Notes",
            placeholder="Describe the extraction process, any issues, observations...",
            lines=3,
            info="Note any observations about the brewing process",
        )

        with gr.Row():
            brew_time_actual_input = gr.Number(
                label="Actual Brew Time (seconds)",
                value=240,
                minimum=30,
                maximum=600,
                step=1,
                info="How long did the actual brewing take?",
            )
            temperature_consistency_input = gr.Dropdown(
                label="Temperature Consistency",
                choices=["Excellent", "Good", "Fair", "Poor", "Very Poor"],
                value="Good",
                info="How consistent was your water temperature?",
            )

        additional_notes_input = gr.Textbox(
            label="Additional Notes",
            placeholder="Any other observations, equipment used, environmental factors...",
            lines=2,
            info="Optional: Any other relevant information",
        )

        with gr.Row():
            submit_result_btn = gr.Button("Submit Results", variant="primary")
            cancel_form_btn = gr.Button("Cancel", variant="secondary")

        result_output = gr.Markdown(value="", visible=True)

        # Form submission logic
        submit_result_btn.click(
            fn=submit_experiment_result,
            inputs=[
                experiment_id_input,
                user_id_input,
                taste_score_input,
                extraction_notes_input,
                brew_time_actual_input,
                temperature_consistency_input,
                overall_satisfaction_input,
                additional_notes_input,
            ],
            outputs=result_output,
        )

        # Cancel button logic
        cancel_form_btn.click(
            fn=lambda: (gr.update(visible=False), ""),
            outputs=[form_column, result_output],
        )

    return form_column, result_output, experiment_id_input, user_id_input


def get_experiment_history(user_id: str, status_filter: str = "All") -> tuple:
    """Get experiment history for a user with optional status filtering."""
    if not experiment_tracker:
        return [], "Experiment system not initialized"

    try:
        if not user_id or user_id.strip() == "":
            return [], "Please enter a User ID to view experiment history."

        # Get experiments for the user and convert to data within session context
        history_data = []
        with experiment_tracker.db_manager.get_session_context() as session:
            # Get experiments for the user
            query = session.query(Experiment).filter_by(user_id=user_id.strip())
            if status_filter != "All":
                query = query.filter_by(status=status_filter.lower())

            experiments = query.order_by(Experiment.created_at.desc()).all()

            if not experiments:
                return (
                    [],
                    f"No experiments found for user '{user_id}' with status '{status_filter}'.",
                )

            # Format data for display within session context
            for exp in experiments:
                # Get results for this experiment
                results = (
                    session.query(ExperimentResult)
                    .filter_by(experiment_id=exp.id, user_id=user_id)
                    .order_by(ExperimentResult.recorded_at.desc())
                    .all()
                )

                # Format the row
                status_emoji = {
                    "pending": "⏳",
                    "in_progress": "🔬",
                    "completed": "✅",
                    "cancelled": "❌",
                }.get(exp.status, "❓")

                taste_score = "N/A"
                if results:
                    latest_result = results[0]  # Most recent result
                    taste_score = (
                        f"{latest_result.taste_score:.1f}/10"
                        if latest_result.taste_score
                        else "N/A"
                    )

                history_data.append(
                    [
                        exp.id,
                        f"{status_emoji} {exp.status.title()}",
                        exp.brew_method,
                        f"{exp.coffee_dose:.0f}g",
                        f"{exp.water_amount:.0f}g",
                        f"{exp.water_temperature:.1f}°C",
                        exp.grind_size,
                        f"{exp.brew_time:.0f}s",
                        taste_score,
                        (
                            exp.created_at.strftime("%Y-%m-%d %H:%M")
                            if exp.created_at
                            else "N/A"
                        ),
                    ]
                )

        return (
            history_data,
            f"Found {len(history_data)} experiments for user '{user_id}'.",
        )

    except Exception as e:
        return [], f"Error retrieving experiment history: {str(e)}"


def get_user_stats(user_id: str) -> str:
    """Get user experiment statistics."""
    if not experiment_tracker:
        return "Experiment system not initialized"

    try:
        if not user_id or user_id.strip() == "":
            return "Please enter a User ID to view statistics."

        # Calculate stats within session context
        with experiment_tracker.db_manager.get_session_context() as session:
            # Get all experiments for user
            experiments = (
                session.query(Experiment).filter_by(user_id=user_id.strip()).all()
            )

            if not experiments:
                return f"No experiments found for user '{user_id}'."

            # Calculate basic stats
            total_experiments = len(experiments)
            completed_experiments = len(
                [e for e in experiments if e.status == "completed"]
            )
            completion_rate = (
                (completed_experiments / total_experiments) * 100
                if total_experiments > 0
                else 0
            )

            # Get all results for user
            results = (
                session.query(ExperimentResult).filter_by(user_id=user_id.strip()).all()
            )
            total_results = len(results)

            # Calculate average taste score
            taste_scores = [r.taste_score for r in results if r.taste_score is not None]
            average_taste_score = (
                sum(taste_scores) / len(taste_scores) if taste_scores else None
            )

            # Status distribution
            status_distribution = {}
            for exp in experiments:
                status_distribution[exp.status] = (
                    status_distribution.get(exp.status, 0) + 1
                )

        avg_score_text = (
            f"{average_taste_score:.1f}/10" if average_taste_score else "N/A"
        )
        completion_rate_text = f"{completion_rate:.1f}%"

        stats_text = f"""
### 📊 User Statistics for '{user_id}'

**Overall Performance:**
- Total Experiments: {total_experiments}
- Completed Experiments: {completed_experiments}
- Completion Rate: {completion_rate_text}
- Average Taste Score: {avg_score_text}
- Total Results Recorded: {total_results}

**Status Distribution:**
"""

        for status, count in status_distribution.items():
            percentage = (count / total_experiments) * 100
            stats_text += f"- {status.title()}: {count} ({percentage:.1f}%)\n"

        return stats_text

    except Exception as e:
        return f"Error retrieving user statistics: {str(e)}"


def create_history_view():
    """Create the experiment history view components."""
    with gr.Column() as history_column:
        gr.Markdown("### 📈 Experiment History & Statistics")

        with gr.Row():
            user_id_history = gr.Textbox(
                label="User ID",
                placeholder="Enter your user ID to view history",
                scale=2,
            )
            status_filter = gr.Dropdown(
                label="Filter by Status",
                choices=["All", "Pending", "In_progress", "Completed", "Cancelled"],
                value="All",
                scale=1,
            )
            load_history_btn = gr.Button("Load History", variant="primary")

        # Statistics section
        user_stats_output = gr.Markdown(
            value="Enter a User ID and click 'Load History' to view statistics."
        )

        # History table
        history_table = gr.Dataframe(
            headers=[
                "ID",
                "Status",
                "Method",
                "Coffee",
                "Water",
                "Temp",
                "Grind",
                "Time",
                "Taste Score",
                "Date",
            ],
            datatype=[
                "number",
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
            ],
            value=[],
            label="Experiment History",
            interactive=False,
            wrap=True,
        )

        history_status = gr.Markdown(value="")

        # Load history button logic
        def load_user_history(user_id, status_filter):
            history_data, status_msg = get_experiment_history(user_id, status_filter)
            stats_text = get_user_stats(user_id)
            return history_data, status_msg, stats_text

        load_history_btn.click(
            fn=load_user_history,
            inputs=[user_id_history, status_filter],
            outputs=[history_table, history_status, user_stats_output],
        )

    return history_column


def load_model():
    """Platform-aware model loading"""
    global model, tokenizer, platform_info, model_size

    if model is not None:
        return model, tokenizer, platform_info, model_size

    platform_info = detect_platform()
    get_platform_settings(platform_info)

    # For now, we'll use the trained QLora model
    # In a production setup, you'd have merged models for different sizes
    model_path = os.environ.get(
        "MODEL_PATH",
        str(project_root / "experiments" / "full-20250606-235528" / "model"),
    )
    model_size = "QLora"

    print(f"Loading coffee model from {model_path}")
    print(
        f"Platform: {platform_info['platform']} with {platform_info['device']} device"
    )

    try:
        # Apply platform-specific optimizations
        if platform_info["device"] == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", torch_dtype=torch.float16
            )
        elif platform_info["device"] == "mps":  # macOS Metal
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map={"": "mps"}, torch_dtype=torch.float16
            )
        else:  # CPU fallback with memory optimization
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Ensure tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Model loaded successfully on {platform_info['device']}")

    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to a simple response
        model = None
        tokenizer = None

    return model, tokenizer, platform_info, model_size


def parse_model_output(output_text):
    """Parse model output to extract structured recommendations"""
    try:
        # Look for JSON-like structure in the output
        start_idx = output_text.find("{")
        end_idx = output_text.rfind("}")

        if start_idx >= 0 and end_idx >= 0:
            json_str = output_text[start_idx : end_idx + 1]
            return json.loads(json_str)
        else:
            # If no JSON found, create a simple structure from the text
            return {
                "grind_change": "See reasoning below",
                "expected_time": "2:30-3:00",
                "extraction": "Needs adjustment",
                "confidence": 0.7,
                "reasoning": output_text.strip(),
            }
    except Exception:
        # Fallback to simple text parsing
        return {
            "grind_change": "See reasoning below",
            "expected_time": "2:30-3:00",
            "extraction": "Needs adjustment",
            "confidence": 0.7,
            "reasoning": output_text.strip(),
        }


def format_output(output_json):
    """Format the model output as beautiful but simple HTML"""
    if not output_json:
        return "❌ **Error:** Unable to process your request. Please try again."

    confidence = output_json.get("confidence", 0.5)
    recommendations = output_json.get("recommendations", {})
    reasoning = output_json.get("reasoning", "No reasoning provided")

    # Simple confidence indicator
    if confidence >= 0.8:
        confidence_badge = "🟢 **High Confidence**"
    elif confidence >= 0.6:
        confidence_badge = "🟡 **Medium Confidence**"
    else:
        confidence_badge = "🔴 **Low Confidence**"

    # Build recommendations section
    rec_text = ""
    if recommendations.get("grind_adjustment"):
        rec_text += f"⚙️ **Grind:** {recommendations['grind_adjustment']}\n\n"
    if recommendations.get("time_adjustment"):
        rec_text += f"⏱️ **Time:** {recommendations['time_adjustment']}\n\n"
    if recommendations.get("extraction_notes"):
        rec_text += f"🔬 **Extraction:** {recommendations['extraction_notes']}\n\n"

    # Format the complete response
    formatted = f"""
## {confidence_badge}

### 🎯 Recommendations
{rec_text if rec_text else "No specific recommendations available."}

### 🧠 AI Reasoning
*{reasoning}*

---
💡 **Tip:** Try one adjustment at a time to see which has the biggest impact on your brew!
    """

    return formatted


def predict_coffee(brewing_params):
    """Predict coffee brewing recommendations"""
    if not brewing_params.strip():
        return "📝 **Input Required:** Please enter your brewing parameters and taste notes to get personalized recommendations."

    model, tokenizer, platform_info, model_size = load_model()

    if model is None or tokenizer is None:
        return """
## 🎭 Demo Response
*Model not loaded - showing example output*

### 🎯 Recommendations
⚙️ **Grind:** Try going one step finer

⏱️ **Time:** Aim for 2:30-3:00 total brew time

🔬 **Extraction:** Your coffee appears under-extracted

### 🧠 AI Reasoning
*Based on your description, the sourness suggests under-extraction. A finer grind will increase extraction yield and should reduce the sour notes while bringing out more sweetness and body.*

---
💡 **Tip:** Try one adjustment at a time to see which has the biggest impact on your brew!
        """

    # Create a prompt for the coffee model
    prompt = f"""Analyze this V60 coffee brewing scenario and provide recommendations:

Brewing details: {brewing_params}

Please provide:
1. Grind adjustment recommendation
2. Expected brew time
3. Extraction assessment
4. Reasoning for the recommendations

Response:"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        # Move inputs to appropriate device
        if platform_info["device"] == "cuda":
            inputs = inputs.to("cuda")
        elif platform_info["device"] == "mps":
            inputs = inputs.to("mps")

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode the response
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the output
        response = output_text[len(prompt) :].strip()

        # Parse and format the output
        output_json = parse_model_output(response)
        formatted_output = format_output(output_json)

        return formatted_output

    except Exception as e:
        return f"""
<div style="
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(239, 68, 68, 0.2);
">
    <div style="font-size: 32px; margin-bottom: 12px;">⚠️</div>
    <h4 style="color: #dc2626; margin: 0 0 8px 0; font-weight: 600;">
        Error Generating Recommendation
    </h4>
    <p style="color: #b91c1c; margin: 0 0 12px 0; font-size: 14px;">
        {str(e)}
    </p>
    <p style="color: #b91c1c; margin: 0; font-size: 12px; opacity: 0.8;">
        Please try again or check your input format.
    </p>
</div>
        """


def create_interface():
    """Create the Gradio interface"""
    # Initialize experiment system
    initialize_experiment_system()

    # Load platform info for display
    platform_info = detect_platform()

    # Simplified but beautiful CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        text-align: center;
    }
    .feature-highlight {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #cbd5e0;
    }
    .beautiful-button {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
    }
    """

    with gr.Blocks(
        title="CoffeeRL-Lite V60 Assistant", theme=gr.themes.Soft(), css=custom_css
    ) as interface:
        # Simplified beautiful header
        gr.HTML(
            """
        <div class="main-header">
            <h1 style="margin: 0 0 10px 0; font-size: 28px; font-weight: 700;">
                ☕ CoffeeRL-Lite: V60 Assistant
            </h1>
            <p style="margin: 0; font-size: 16px; opacity: 0.9;">
                Your AI-powered coffee brewing companion with experiment laboratory
            </p>
        </div>
        """
        )

        # Platform info
        gr.Markdown(
            f"**🖥️ Running on:** {platform_info['platform']} with {platform_info['device']} acceleration"
        )

        with gr.Tabs():
            with gr.TabItem("☕ Coffee Assistant"):
                # Feature highlight section
                gr.HTML(
                    """
                <div class="feature-highlight">
                    <h3 style="margin: 0 0 10px 0; color: #2d3748; text-align: center;">
                        🤖 AI Brewing Assistant
                    </h3>
                    <p style="margin: 0; color: #4a5568; text-align: center;">
                        Get personalized brewing recommendations powered by machine learning
                    </p>
                </div>
                """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Brewing Parameters & Taste Notes")
                        gr.Markdown(
                            "*Describe your brewing setup and taste experience for personalized recommendations*"
                        )

                        input_text = gr.Textbox(
                            label="",
                            placeholder="Example: V60, 15g coffee, 250g water, medium grind, 2:10 brew time, tastes sour and weak",
                            lines=5,
                            max_lines=10,
                        )

                        submit_btn = gr.Button(
                            "🚀 Get AI Recommendation",
                            variant="primary",
                            elem_classes=["beautiful-button"],
                        )

                        with gr.Accordion("💡 Example Inputs", open=False):
                            gr.Markdown(
                                """
**😖 Under-extracted (sour):**
```
V60, 15g coffee, 250g water, medium grind, 2:00 brew time, tastes sour and lacks sweetness
```

**😤 Over-extracted (bitter):**
```
V60, 20g coffee, 320g water, medium-fine grind, 3:30 brew time, tastes bitter and harsh
```

**😐 Balanced but weak:**
```
V60, 18g coffee, 300g water, medium-coarse grind, 2:45 brew time, tastes balanced but weak
```

**😍 Good extraction:**
```
V60, 16g coffee, 260g water, medium-fine grind, 2:45 brew time, tastes sweet with good body
```
                            """
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 AI Recommendation")
                        gr.Markdown(
                            "*Your personalized brewing advice will appear here*"
                        )

                        output_text = gr.Markdown(
                            value="🤖 **Ready to Help!** Enter your brewing details and click 'Get AI Recommendation' to receive personalized advice."
                        )

                # Connect the button to the prediction function
                submit_btn.click(predict_coffee, inputs=input_text, outputs=output_text)

            with gr.TabItem("🧪 Experiment Lab"):
                gr.Markdown("## 🧪 Coffee Experiment Laboratory")
                gr.Markdown(
                    "Participate in coffee brewing experiments to help improve our AI models and discover new brewing techniques."
                )

                # Experiment cards section
                gr.Markdown("### Available Experiments")
                experiment_cards = gr.HTML(
                    value=create_experiment_grid(), label="Experiments"
                )

                with gr.Row():
                    refresh_btn = gr.Button(
                        "🔄 Refresh Experiments",
                        variant="secondary",
                        elem_classes=["beautiful-button"],
                    )
                    show_form_btn = gr.Button(
                        "📝 Submit Results",
                        variant="primary",
                        elem_classes=["beautiful-button"],
                    )
                    show_custom_btn = gr.Button(
                        "🎯 Create Custom Experiment",
                        variant="primary",
                        elem_classes=["beautiful-button"],
                    )

                refresh_btn.click(
                    fn=lambda: create_experiment_grid(), outputs=experiment_cards
                )

                # Custom experiment form
                custom_form_column, custom_result_output = (
                    create_custom_experiment_form()
                )

                # Result submission form
                form_column, form_result_output, experiment_id_input, user_id_input = (
                    create_result_submission_form()
                )

                # Show form button logic
                show_form_btn.click(
                    fn=lambda: gr.update(visible=True), outputs=form_column
                )

                # Show custom form button logic
                show_custom_btn.click(
                    fn=lambda: gr.update(visible=True), outputs=custom_form_column
                )

                # Experiment history and statistics section
                create_history_view()

        # Simplified footer
        gr.HTML(
            """
        <div style="
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            color: white;
            text-align: center;
        ">
            <h3 style="margin: 0 0 10px 0; font-size: 18px;">ℹ️ About CoffeeRL-Lite</h3>
            <p style="margin: 0 0 15px 0; opacity: 0.9;">
                This AI assistant helps optimize V60 pour-over coffee brewing by analyzing your brewing parameters and taste notes.
                The Experiment Lab allows you to participate in structured experiments to help improve our models.
            </p>
            <div style="font-size: 14px; opacity: 0.8;">
                💡 <strong>Tips:</strong> Include coffee amount, water amount, grind size, brew time, and taste notes for best results
            </div>
        </div>
        """
        )

    return interface


def main():
    """Main function with platform-specific launch settings"""
    interface = create_interface()
    platform_info = detect_platform()
    settings = get_platform_settings(platform_info)

    print("Starting CoffeeRL-Lite Gradio Interface...")
    print(f"Platform: {platform_info['platform']}")
    print(f"Device: {platform_info['device']} ({platform_info['device_name']})")

    # Check for environment variable override for port
    server_port = int(
        os.environ.get("GRADIO_SERVER_PORT", settings.get("server_port", 7860))
    )

    print(f"Starting server on port: {server_port}")

    # Apply platform-specific launch settings
    interface.launch(
        share=settings.get("share", False),
        server_name=settings.get("server_name", "127.0.0.1"),
        server_port=server_port,
        debug=settings.get("debug", False),
    )


if __name__ == "__main__":
    main()
