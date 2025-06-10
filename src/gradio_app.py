import json
import os
import sys
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup to avoid import issues
from config.platform_config import detect_platform, get_platform_settings  # noqa: E402
from src.experiment_tracker import ExperimentTracker  # noqa: E402

# Global variables to store loaded model
model = None
tokenizer = None
platform_info = None
model_size = None

# Sample experiment data for demonstration
SAMPLE_EXPERIMENTS = [
    {
        "id": 1,
        "title": "Optimal Grind Size for Ethiopian Beans",
        "description": "Investigate the relationship between grind size and extraction yield for Ethiopian single-origin beans.",
        "brew_method": "V60",
        "difficulty": "Medium",
        "estimated_time": "15 minutes",
        "reward_points": 50,
        "parameters": {
            "coffee_dose": 18.0,
            "water_amount": 300.0,
            "water_temperature": 92.0,
            "grind_size": "medium-fine",
            "brew_time": 240,
        },
        "scientific_rationale": "Ethiopian beans have unique density and cellular structure that may require specific grind sizes for optimal extraction. This experiment helps identify the sweet spot for balanced flavor extraction.",
        "status": "pending",
    },
    {
        "id": 2,
        "title": "Temperature Stability Impact",
        "description": "Study how water temperature consistency affects extraction quality and taste profile.",
        "brew_method": "V60",
        "difficulty": "Hard",
        "estimated_time": "25 minutes",
        "reward_points": 75,
        "parameters": {
            "coffee_dose": 16.0,
            "water_amount": 260.0,
            "water_temperature": 88.0,
            "grind_size": "medium",
            "brew_time": 210,
        },
        "scientific_rationale": "Temperature stability during brewing significantly impacts extraction kinetics. This experiment measures how temperature variations affect final cup quality and helps optimize brewing protocols.",
        "status": "pending",
    },
    {
        "id": 3,
        "title": "Bloom Time Optimization",
        "description": "Determine the optimal bloom duration for different coffee origins and roast levels.",
        "brew_method": "V60",
        "difficulty": "Easy",
        "estimated_time": "12 minutes",
        "reward_points": 30,
        "parameters": {
            "coffee_dose": 20.0,
            "water_amount": 320.0,
            "water_temperature": 94.0,
            "grind_size": "medium-coarse",
            "brew_time": 180,
            "bloom_time": 45,
        },
        "scientific_rationale": "Bloom time affects CO2 degassing and initial extraction. Finding the optimal bloom duration can improve overall extraction uniformity and flavor clarity.",
        "status": "completed",
    },
]


def get_difficulty_color(difficulty: str) -> str:
    """Get color code for difficulty level."""
    colors = {
        "Easy": "#28a745",  # Green
        "Medium": "#ffc107",  # Yellow
        "Hard": "#dc3545",  # Red
    }
    return colors.get(difficulty, "#6c757d")


def create_experiment_card(experiment: dict) -> str:
    """Create HTML for an experiment card."""
    difficulty_color = get_difficulty_color(experiment["difficulty"])
    status_badge = (
        "✅ Completed" if experiment["status"] == "completed" else "🔬 Available"
    )
    status_color = "#28a745" if experiment["status"] == "completed" else "#007bff"

    # Format parameters for display
    params = experiment["parameters"]
    param_list = []
    for key, value in params.items():
        formatted_key = key.replace("_", " ").title()
        if "temperature" in key:
            param_list.append(f"<li><strong>{formatted_key}:</strong> {value}°C</li>")
        elif "time" in key:
            param_list.append(f"<li><strong>{formatted_key}:</strong> {value}s</li>")
        elif "dose" in key or "amount" in key:
            param_list.append(f"<li><strong>{formatted_key}:</strong> {value}g</li>")
        else:
            param_list.append(f"<li><strong>{formatted_key}:</strong> {value}</li>")

    params_html = "".join(param_list)

    card_html = f"""
    <div class="experiment-card" style="
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        margin: 10px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
        max-width: 400px;
        width: 100%;
    " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 12px rgba(0, 0, 0, 0.15)'"
       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 6px rgba(0, 0, 0, 0.1)'">

        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="
                background-color: {status_color};
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
            ">{status_badge}</span>
            <span style="
                background-color: {difficulty_color};
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
            ">{experiment["difficulty"]}</span>
        </div>

        <h3 style="margin: 0 0 8px 0; color: #2c3e50; font-size: 18px;">{experiment["title"]}</h3>

        <p style="color: #6c757d; margin: 0 0 12px 0; font-size: 14px; line-height: 1.4;">
            {experiment["description"]}
        </p>

        <div style="margin-bottom: 12px;">
            <h4 style="margin: 0 0 6px 0; color: #495057; font-size: 14px;">Parameters:</h4>
            <ul style="margin: 0; padding-left: 16px; font-size: 13px; color: #6c757d;">
                {params_html}
            </ul>
        </div>

        <div style="margin-bottom: 12px; padding: 10px; background-color: #f8f9fa; border-radius: 6px;">
            <h4 style="margin: 0 0 6px 0; color: #495057; font-size: 14px;">Scientific Rationale:</h4>
            <p style="margin: 0; font-size: 13px; color: #6c757d; line-height: 1.4;">
                {experiment["scientific_rationale"]}
            </p>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 12px; color: #6c757d;">
                <span>⏱️ {experiment["estimated_time"]}</span>
            </div>
            <div style="font-size: 12px; color: #007bff; font-weight: bold;">
                <span>🏆 {experiment["reward_points"]} points</span>
            </div>
        </div>
    </div>
    """

    return card_html


def create_experiment_grid() -> str:
    """Create a grid of experiment cards."""
    cards_html = []

    for experiment in SAMPLE_EXPERIMENTS:
        cards_html.append(create_experiment_card(experiment))

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
    </style>
    """

    return grid_html


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

        # Initialize experiment tracker with SQLite
        from src.database import DatabaseManager

        sqlite_url = "sqlite:///cofferl.db"
        db_manager = DatabaseManager(sqlite_url)
        tracker = ExperimentTracker(db_manager)

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
        result = tracker.record_experiment_result(
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
            tracker.log_user_interaction(
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

    return form_column, result_output


def get_experiment_history(user_id: str, status_filter: str = "All") -> tuple:
    """Get experiment history for a user with optional status filtering."""
    try:
        if not user_id or user_id.strip() == "":
            return [], "Please enter a User ID to view experiment history."

        # Initialize experiment tracker with SQLite
        from src.database import DatabaseManager

        sqlite_url = "sqlite:///cofferl.db"
        db_manager = DatabaseManager(sqlite_url)
        tracker = ExperimentTracker(db_manager)

        # Get experiments for the user
        if status_filter == "All":
            experiments = tracker.get_experiments_by_user(user_id.strip())
        else:
            experiments = tracker.get_experiments_by_user(
                user_id.strip(), status=status_filter.lower()
            )

        if not experiments:
            return (
                [],
                f"No experiments found for user '{user_id}' with status '{status_filter}'.",
            )

        # Format data for display
        history_data = []
        for exp in experiments:
            # Get results for this experiment
            results = tracker.get_experiment_results(
                experiment_id=exp.id, user_id=user_id
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
                    f"{exp.coffee_dose}g",
                    f"{exp.water_amount}g",
                    f"{exp.water_temperature}°C",
                    exp.grind_size,
                    f"{exp.brew_time}s",
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
    try:
        if not user_id or user_id.strip() == "":
            return "Please enter a User ID to view statistics."

        # Initialize experiment tracker with SQLite
        from src.database import DatabaseManager

        sqlite_url = "sqlite:///cofferl.db"
        db_manager = DatabaseManager(sqlite_url)
        tracker = ExperimentTracker(db_manager)

        stats = tracker.get_user_experiment_stats(user_id.strip())

        if stats["total_experiments"] == 0:
            return f"No experiments found for user '{user_id}'."

        avg_score_text = (
            f"{stats['average_taste_score']:.1f}/10"
            if stats["average_taste_score"]
            else "N/A"
        )
        completion_rate_text = f"{stats['completion_rate']:.1f}%"

        stats_text = f"""
### 📊 User Statistics for '{user_id}'

**Overall Performance:**
- Total Experiments: {stats["total_experiments"]}
- Completed Experiments: {stats["completed_experiments"]}
- Completion Rate: {completion_rate_text}
- Average Taste Score: {avg_score_text}
- Total Results Recorded: {stats["total_results"]}

**Status Distribution:**
"""

        for status, count in stats["status_distribution"].items():
            percentage = (count / stats["total_experiments"]) * 100
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
    """Format the output for display"""
    if not output_json:
        return "Error: Could not parse model output"

    grind_change = output_json.get("grind_change", "unknown")
    expected_time = output_json.get("expected_time", "unknown")
    extraction = output_json.get("extraction", "unknown")
    confidence = output_json.get("confidence", 0)
    reasoning = output_json.get("reasoning", "")

    formatted = f"""### Recommendation

**Grind Adjustment:** {grind_change}

**Expected Brew Time:** {expected_time}

**Extraction Assessment:** {extraction}

**Confidence:** {confidence:.2f}

### Reasoning
{reasoning}"""

    return formatted


def predict_coffee(brewing_params):
    """Predict coffee brewing recommendations"""
    if not brewing_params.strip():
        return "Please enter your brewing parameters and taste notes."

    model, tokenizer, platform_info, model_size = load_model()

    if model is None or tokenizer is None:
        return """### Demo Response (Model not loaded)

**Grind Adjustment:** Try going one step finer

**Expected Brew Time:** 2:30-3:00

**Extraction Assessment:** Likely under-extracted based on sour notes

### Reasoning
Based on your description of sour taste, this typically indicates under-extraction. Try grinding finer to increase extraction yield. The 2:10 brew time is reasonable for V60, but with a finer grind you might see it extend to 2:30-3:00 which should improve sweetness and reduce sourness."""

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
        return f"Error generating recommendation: {str(e)}\n\nPlease try again or check your input format."


def create_interface():
    """Create the Gradio interface"""
    # Load platform info for display
    platform_info = detect_platform()

    with gr.Blocks(
        title="CoffeeRL-Lite V60 Assistant", theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown("# ☕ CoffeeRL-Lite: V60 Pour-Over Assistant")
        gr.Markdown(
            "Your AI-powered coffee brewing companion with experiment laboratory."
        )

        # Display platform information
        gr.Markdown(
            f"*Running on: {platform_info['platform']} with {platform_info['device']} acceleration*"
        )

        with gr.Tabs():
            with gr.TabItem("Coffee Assistant"):
                gr.Markdown("## Get personalized brewing recommendations")
                gr.Markdown(
                    "Enter your V60 brewing parameters and taste notes to get personalized grind adjustment recommendations."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        input_text = gr.Textbox(
                            label="Brewing Parameters & Taste Notes",
                            placeholder="Example: V60, 15g coffee, 250g water, medium grind, 2:10 brew time, tastes sour and weak",
                            lines=5,
                            max_lines=10,
                        )

                        submit_btn = gr.Button("Get Recommendation", variant="primary")

                        gr.Markdown("### Example Inputs:")
                        with gr.Accordion("Click to see examples", open=False):
                            gr.Markdown(
                                """
**Under-extracted (sour):**
V60, 15g coffee, 250g water, medium grind, 2:00 brew time, tastes sour and lacks sweetness

**Over-extracted (bitter):**
V60, 20g coffee, 320g water, medium-fine grind, 3:30 brew time, tastes bitter and harsh

**Balanced but weak:**
V60, 18g coffee, 300g water, medium-coarse grind, 2:45 brew time, tastes balanced but weak

**Good extraction:**
V60, 16g coffee, 260g water, medium-fine grind, 2:45 brew time, tastes sweet with good body
                            """
                            )

                    with gr.Column(scale=1):
                        output_text = gr.Markdown(
                            label="Recommendation",
                            value="Enter your brewing details and click 'Get Recommendation' to receive personalized advice.",
                        )

                # Connect the button to the prediction function
                submit_btn.click(predict_coffee, inputs=input_text, outputs=output_text)

            with gr.TabItem("Experiment Lab"):
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
                        "🔄 Refresh Experiments", variant="secondary"
                    )
                    show_form_btn = gr.Button("📝 Submit Results", variant="primary")

                refresh_btn.click(
                    fn=lambda: create_experiment_grid(), outputs=experiment_cards
                )

                # Result submission form
                form_column, form_result_output = create_result_submission_form()

                # Show form button logic
                show_form_btn.click(
                    fn=lambda: gr.update(visible=True), outputs=form_column
                )

                # Experiment history and statistics section
                create_history_view()

        # Add footer information
        gr.Markdown(
            """
---
### About CoffeeRL-Lite
This AI assistant helps optimize V60 pour-over coffee brewing by analyzing your brewing parameters and taste notes.
The Experiment Lab allows you to participate in structured experiments to help improve our models.

**Tips for best results:**
- Include coffee amount, water amount, grind size, brew time, and taste notes
- Be specific about taste issues (sour, bitter, weak, harsh, etc.)
- Mention your brewing method (V60, Chemex, etc.)
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
