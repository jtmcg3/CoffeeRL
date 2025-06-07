#!/bin/bash
# Local evaluation script for CoffeeRL-Lite
# Usage: ./scripts/evaluate_local.sh [quick|full|custom] <model_path>

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "src/evaluate_local.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [mode] <model_path>"
    echo ""
    echo "Modes:"
    echo "  quick   - Quick evaluation with 50 samples"
    echo "  full    - Full evaluation with all samples"
    echo "  custom  - Custom evaluation (use environment variables)"
    echo "  help    - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 quick ./models/coffee-qwen2-qlora"
    echo "  $0 full ./models/coffee-qwen2-qlora"
    echo "  SAMPLES=100 TEMP=0.2 $0 custom ./models/coffee-qwen2-qlora"
    echo ""
    echo "Environment variables for custom mode:"
    echo "  SAMPLES     - Maximum number of samples to evaluate"
    echo "  TEMP        - Temperature for generation (default: 0.1)"
    echo "  EVAL_DATA   - Path to evaluation dataset"
    echo "  OUTPUT_FILE - Output file for results"
    echo "  FORCE_CPU   - Set to 'true' to force CPU evaluation"
    echo "  SAVE_PREDS  - Set to 'true' to save individual predictions"
}

# Parse arguments
MODE=${1:-help}
MODEL_PATH=${2:-}

if [ "$MODE" = "help" ] || [ -z "$MODEL_PATH" ]; then
    show_usage
    exit 0
fi

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    print_error "Model path does not exist: $MODEL_PATH"
    exit 1
fi

print_info "CoffeeRL-Lite Model Evaluation"
print_info "Model: $MODEL_PATH"
print_info "Mode: $MODE"

# Set default values
EVAL_DATA=${EVAL_DATA:-"data/processed/coffee_validation_qwen2_0.5B.json"}
OUTPUT_FILE=${OUTPUT_FILE:-"evaluation_results_$(date +%Y%m%d_%H%M%S).json"}
TEMP=${TEMP:-0.1}

# Check if evaluation data exists
if [ ! -f "$EVAL_DATA" ] && [ ! -d "$EVAL_DATA" ]; then
    print_warning "Evaluation data not found at $EVAL_DATA"
    print_info "Looking for alternative evaluation data..."

    # Try to find alternative evaluation data
    if [ -f "data/processed/coffee_validation_dataset.json" ]; then
        EVAL_DATA="data/processed/coffee_validation_dataset.json"
        print_info "Using: $EVAL_DATA"
    elif [ -d "data/processed/coffee_validation_dataset" ]; then
        EVAL_DATA="data/processed/coffee_validation_dataset"
        print_info "Using: $EVAL_DATA"
    else
        print_error "No evaluation data found. Please ensure evaluation data exists."
        exit 1
    fi
fi

# Build command based on mode
case $MODE in
    "quick")
        print_info "Running quick evaluation (50 samples)..."
        CMD="uv run python src/evaluate_local.py --model-path \"$MODEL_PATH\" --eval-data \"$EVAL_DATA\" --output-file \"$OUTPUT_FILE\" --quick-eval --temperature $TEMP"
        ;;
    "full")
        print_info "Running full evaluation (all samples)..."
        CMD="uv run python src/evaluate_local.py --model-path \"$MODEL_PATH\" --eval-data \"$EVAL_DATA\" --output-file \"$OUTPUT_FILE\" --temperature $TEMP"
        ;;
    "custom")
        print_info "Running custom evaluation..."
        CMD="uv run python src/evaluate_local.py --model-path \"$MODEL_PATH\" --eval-data \"$EVAL_DATA\" --output-file \"$OUTPUT_FILE\" --temperature $TEMP"

        # Add custom parameters
        if [ -n "$SAMPLES" ]; then
            CMD="$CMD --max-samples $SAMPLES"
            print_info "Max samples: $SAMPLES"
        fi

        if [ "$FORCE_CPU" = "true" ]; then
            CMD="$CMD --force-cpu"
            print_info "Forcing CPU evaluation"
        fi

        if [ "$SAVE_PREDS" = "true" ]; then
            CMD="$CMD --save-predictions"
            print_info "Saving individual predictions"
        fi
        ;;
    *)
        print_error "Invalid mode: $MODE"
        show_usage
        exit 1
        ;;
esac

print_info "Output file: $OUTPUT_FILE"
print_info "Temperature: $TEMP"

# Show platform summary
print_info "Platform Summary:"
uv run python config/platform_config.py

echo ""
print_info "Starting evaluation..."

# Run the evaluation
eval $CMD

if [ $? -eq 0 ]; then
    print_success "Evaluation completed successfully!"
    print_info "Results saved to: $OUTPUT_FILE"

    # Show quick summary if jq is available
    if command -v jq &> /dev/null && [ -f "$OUTPUT_FILE" ]; then
        echo ""
        print_info "Quick Summary:"
        echo "Samples: $(jq -r '.evaluation_info.num_samples' "$OUTPUT_FILE")"
        echo "Grind Accuracy: $(jq -r '.metrics.grind_accuracy' "$OUTPUT_FILE" | awk '{printf "%.1f%%", $1*100}')"
        echo "Extraction Accuracy: $(jq -r '.metrics.extraction_accuracy' "$OUTPUT_FILE" | awk '{printf "%.1f%%", $1*100}')"
        echo "Average Accuracy: $(jq -r '.metrics.average_accuracy' "$OUTPUT_FILE" | awk '{printf "%.1f%%", $1*100}')"
    fi
else
    print_error "Evaluation failed!"
    exit 1
fi
