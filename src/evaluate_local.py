#!/usr/bin/env python3
"""Local evaluation script for CoffeeRL-Lite trained models.

This script evaluates locally trained Qwen2-0.5B models on coffee brewing
recommendation tasks with comprehensive metrics and analysis.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from qlora_config import get_device_map  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="CoffeeRL-Lite Local Model Evaluation Script"
    )

    # Model and data paths
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="data/processed/coffee_validation_dataset",
        help="Path to evaluation dataset directory",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation_results.json",
        help="Output file for evaluation results",
    )

    # Evaluation parameters
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for generation"
    )
    parser.add_argument(
        "--quick-eval",
        action="store_true",
        help="Run quick evaluation with subset of data",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save individual predictions to file",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU evaluation")

    return parser.parse_args()


def load_model_and_tokenizer(
    model_path: str, force_cpu: bool = False
) -> Tuple[Any, Any]:
    """Load the trained model and tokenizer."""
    print(f"🔄 Loading model from {model_path}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device_map = "cpu" if force_cpu else get_device_map()
        torch_dtype = torch.float32 if device_map == "cpu" else torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        print(f"✅ Model loaded successfully on {device_map}")
        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def load_evaluation_data(
    eval_data_path: str, max_samples: Optional[int] = None
) -> List[Dict]:
    """Load evaluation dataset."""
    print(f"📂 Loading evaluation data from {eval_data_path}...")

    try:
        if os.path.isdir(eval_data_path):
            dataset = load_from_disk(eval_data_path)
            data = [{"text": item["text"]} for item in dataset]
        else:
            with open(eval_data_path, "r") as f:
                data = json.load(f)

        if max_samples:
            data = data[:max_samples]

        print(f"✅ Loaded {len(data)} evaluation examples")
        return data

    except Exception as e:
        print(f"❌ Error loading evaluation data: {e}")
        raise


def extract_input_and_expected_output(text: str) -> Tuple[str, Dict]:
    """Extract input prompt and expected output from evaluation text."""
    parts = text.split(
        "Provide grind adjustment, expected brew time, extraction assessment, and reasoning:"
    )
    if len(parts) != 2:
        raise ValueError(f"Invalid text format: {text[:100]}...")

    input_text = parts[0].strip()
    expected_json_str = parts[1].strip()

    try:
        expected_output = json.loads(expected_json_str)
        return input_text, expected_output
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in expected output: {e}")


def generate_prediction(
    model: Any, tokenizer: Any, input_text: str, temperature: float = 0.1
) -> str:
    """Generate model prediction for input text."""
    prompt = f"{input_text}\n\nProvide grind adjustment, expected brew time, extraction assessment, and reasoning:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt) :].strip()

    return response


def parse_model_output(output: str) -> Dict:
    """Parse model output to extract structured prediction."""
    json_match = re.search(r"\{[^}]*\}", output)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    result = {}

    # Extract grind_change
    grind_patterns = [
        r'grind_change["\']?\s*:\s*["\']?([^"\'}\s,]+)',
        r"(finer_\d+|coarser_\d+|none)",
    ]
    for pattern in grind_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            result["grind_change"] = match.group(1).strip().strip("\"'")
            break

    # Extract extraction
    extraction_patterns = [
        r'extraction["\']?\s*:\s*["\']?([^"\'}\s,]+)',
        r"(under|over|good)",
    ]
    for pattern in extraction_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            result["extraction"] = match.group(1).strip().strip("\"'")
            break

    return result


def calculate_metrics(
    predictions: List[Dict], expected: List[Dict]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    grind_correct = 0
    extraction_correct = 0
    total = 0

    for pred, exp in zip(predictions, expected):
        total += 1

        if "grind_change" in pred and "grind_change" in exp:
            if pred["grind_change"] == exp["grind_change"]:
                grind_correct += 1

        if "extraction" in pred and "extraction" in exp:
            if pred["extraction"] == exp["extraction"]:
                extraction_correct += 1

    return {
        "grind_accuracy": grind_correct / total if total > 0 else 0.0,
        "extraction_accuracy": extraction_correct / total if total > 0 else 0.0,
        "average_accuracy": (
            (grind_correct + extraction_correct) / (2 * total) if total > 0 else 0.0
        ),
    }


def evaluate_model(
    model: Any, tokenizer: Any, eval_data: List[Dict], args: argparse.Namespace
) -> Dict[str, Any]:
    """Run model evaluation."""
    print(f"🔍 Starting evaluation on {len(eval_data)} samples...")

    predictions = []
    expected_outputs = []
    raw_predictions = []

    start_time = time.time()

    for i, item in enumerate(tqdm(eval_data, desc="Evaluating")):
        try:
            input_text, expected_output = extract_input_and_expected_output(
                item["text"]
            )
            raw_prediction = generate_prediction(
                model, tokenizer, input_text, args.temperature
            )
            parsed_prediction = parse_model_output(raw_prediction)

            predictions.append(parsed_prediction)
            expected_outputs.append(expected_output)

            if args.save_predictions:
                raw_predictions.append(
                    {
                        "input": input_text,
                        "expected": expected_output,
                        "raw_output": raw_prediction,
                        "parsed_output": parsed_prediction,
                    }
                )

        except Exception as e:
            print(f"⚠️  Error processing sample {i}: {e}")
            predictions.append({})
            expected_outputs.append({})

    evaluation_time = time.time() - start_time

    # Calculate metrics
    metrics = calculate_metrics(predictions, expected_outputs)

    results = {
        "evaluation_info": {
            "model_path": args.model_path,
            "num_samples": len(eval_data),
            "evaluation_time_seconds": evaluation_time,
        },
        "metrics": metrics,
    }

    if args.save_predictions:
        results["predictions"] = raw_predictions

    return results


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 50)
    print("📋 EVALUATION SUMMARY")
    print("=" * 50)

    info = results["evaluation_info"]
    metrics = results["metrics"]

    print(f"📁 Model: {info['model_path']}")
    print(f"📊 Samples: {info['num_samples']}")
    print(f"⏱️  Time: {info['evaluation_time_seconds']:.2f}s")
    print()
    print("🎯 METRICS:")
    print(
        f"  Grind Change Accuracy:  {metrics['grind_accuracy']:.3f} ({metrics['grind_accuracy']*100:.1f}%)"
    )
    print(
        f"  Extraction Accuracy:    {metrics['extraction_accuracy']:.3f} ({metrics['extraction_accuracy']*100:.1f}%)"
    )
    print(
        f"  Average Accuracy:       {metrics['average_accuracy']:.3f} ({metrics['average_accuracy']*100:.1f}%)"
    )
    print("=" * 50)


def main():
    """Main evaluation function."""
    args = parse_arguments()

    print("☕ CoffeeRL-Lite Model Evaluation")
    print("=" * 40)

    if args.quick_eval:
        args.max_samples = 50
        print("🚀 Quick evaluation mode: using 50 samples")

    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.force_cpu)
        eval_data = load_evaluation_data(args.eval_data, args.max_samples)
        results = evaluate_model(model, tokenizer, eval_data, args)

        print_evaluation_summary(results)

        print(f"💾 Saving results to {args.output_file}...")
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✅ Evaluation complete! Results saved to {args.output_file}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
