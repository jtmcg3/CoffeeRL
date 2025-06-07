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

# Global variables to store loaded model
model = None
tokenizer = None
platform_info = None
model_size = None


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
            "Enter your V60 brewing parameters and taste notes to get personalized grind adjustment recommendations."
        )

        # Display platform information
        gr.Markdown(
            f"*Running on: {platform_info['platform']} with {platform_info['device']} acceleration*"
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

        # Add some footer information
        gr.Markdown(
            """
---
### About CoffeeRL-Lite
This AI assistant helps optimize V60 pour-over coffee brewing by analyzing your brewing parameters and taste notes.
It provides recommendations for grind adjustments, expected brew times, and extraction assessments.

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
