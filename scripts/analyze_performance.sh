#!/bin/bash

# CoffeeRL Model Performance Analysis Script
# Analyzes trained models against success criteria and compares performance

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
MODE="auto"
OUTPUT_DIR="analysis_results"
RESULTS_DIR="evaluation_results"
MODEL_NAMES=""
RESULTS_FILES=""

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

show_usage() {
    cat << EOF
CoffeeRL Model Performance Analysis

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --mode MODE              Analysis mode: auto, single, compare (default: auto)
    --results-dir DIR        Directory containing evaluation results (default: evaluation_results)
    --results-files FILES    Comma-separated list of specific result files
    --model-names NAMES      Comma-separated list of model names
    --output-dir DIR         Output directory for analysis (default: analysis_results)
    --help                   Show this help message

MODES:
    auto        Automatically find and analyze all evaluation results
    single      Analyze a single model (requires --results-files)
    compare     Compare multiple models (requires --results-files and --model-names)

EXAMPLES:
    # Auto-analyze all results in evaluation_results/
    $0 --mode auto

    # Analyze specific result file
    $0 --mode single --results-files evaluation_results.json

    # Compare two models
    $0 --mode compare \\
        --results-files "model1_results.json,model2_results.json" \\
        --model-names "Qwen2-0.5B,Qwen2-1.5B"

    # Custom output directory
    $0 --mode auto --output-dir custom_analysis

EOF
}

check_dependencies() {
    print_status "Checking dependencies..."

    # Check if uv is available
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install uv first."
        exit 1
    fi

    # Check if Python script exists
    if [[ ! -f "src/analyze_performance.py" ]]; then
        print_error "Analysis script not found: src/analyze_performance.py"
        exit 1
    fi

    print_success "Dependencies check passed"
}

find_evaluation_results() {
    print_status "Searching for evaluation results in ${RESULTS_DIR}..."

    if [[ ! -d "$RESULTS_DIR" ]]; then
        print_warning "Results directory not found: $RESULTS_DIR"
        return 1
    fi

    # Find JSON files that look like evaluation results
    local found_files=()
    while IFS= read -r -d '' file; do
        # Check if file contains evaluation metrics
        if grep -q '"metrics"' "$file" 2>/dev/null; then
            found_files+=("$file")
        fi
    done < <(find "$RESULTS_DIR" -name "*.json" -print0)

    if [[ ${#found_files[@]} -eq 0 ]]; then
        print_warning "No evaluation result files found in $RESULTS_DIR"
        return 1
    fi

    print_success "Found ${#found_files[@]} evaluation result file(s):"
    for file in "${found_files[@]}"; do
        echo "  - $file"
    done

    # Set results files for auto mode
    RESULTS_FILES=$(IFS=','; echo "${found_files[*]}")
    return 0
}

extract_model_names() {
    print_status "Extracting model names from result files..."

    local names=()
    IFS=',' read -ra files <<< "$RESULTS_FILES"

    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            # Try to extract model name from file content or filename
            local model_name=""

            # Check if model path is in the JSON
            if command -v jq &> /dev/null; then
                model_name=$(jq -r '.evaluation_info.model_path // empty' "$file" 2>/dev/null || echo "")
                if [[ -n "$model_name" ]]; then
                    # Extract model size from path
                    if [[ "$model_name" =~ 0\.5[bB] ]]; then
                        model_name="Qwen2-0.5B"
                    elif [[ "$model_name" =~ 1\.5[bB] ]]; then
                        model_name="Qwen2-1.5B"
                    else
                        model_name="Unknown"
                    fi
                fi
            fi

            # Fallback to filename analysis
            if [[ -z "$model_name" ]]; then
                local basename=$(basename "$file" .json)
                if [[ "$basename" =~ 0\.5[bB] ]]; then
                    model_name="Qwen2-0.5B"
                elif [[ "$basename" =~ 1\.5[bB] ]]; then
                    model_name="Qwen2-1.5B"
                else
                    model_name="$basename"
                fi
            fi

            names+=("$model_name")
        fi
    done

    MODEL_NAMES=$(IFS=','; echo "${names[*]}")
    print_success "Extracted model names: $MODEL_NAMES"
}

run_analysis() {
    print_header "Running Model Performance Analysis"

    # Prepare arguments
    local args=("--output-dir" "$OUTPUT_DIR")

    # Add results files
    IFS=',' read -ra files <<< "$RESULTS_FILES"
    args+=("--results")
    args+=("${files[@]}")

    # Add model names if available
    if [[ -n "$MODEL_NAMES" ]]; then
        IFS=',' read -ra names <<< "$MODEL_NAMES"
        args+=("--model-names")
        args+=("${names[@]}")
    fi

    print_status "Running analysis with arguments: ${args[*]}"

    # Run the analysis
    if uv run python src/analyze_performance.py "${args[@]}"; then
        print_success "Analysis completed successfully!"

        # Show results
        echo ""
        print_header "Analysis Results"

        if [[ -f "$OUTPUT_DIR/performance_analysis_report.md" ]]; then
            print_success "📄 Report: $OUTPUT_DIR/performance_analysis_report.md"
        fi

        if [[ -f "$OUTPUT_DIR/model_performance_comparison.png" ]]; then
            print_success "📊 Plots: $OUTPUT_DIR/model_performance_comparison.png"
        fi

        # Show quick summary if report exists
        if [[ -f "$OUTPUT_DIR/performance_analysis_report.md" ]]; then
            echo ""
            print_status "Quick Summary:"
            grep -E "^- \*\*" "$OUTPUT_DIR/performance_analysis_report.md" | head -5 || true
        fi

    else
        print_error "Analysis failed!"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --results-files)
            RESULTS_FILES="$2"
            shift 2
            ;;
        --model-names)
            MODEL_NAMES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_header "CoffeeRL Model Performance Analysis"
print_status "Mode: $MODE"
print_status "Output directory: $OUTPUT_DIR"

check_dependencies

case $MODE in
    auto)
        print_status "Auto mode: Finding evaluation results..."
        if find_evaluation_results; then
            extract_model_names
            run_analysis
        else
            print_error "No evaluation results found for auto analysis"
            print_status "Try running evaluation first with: make eval-quick"
            exit 1
        fi
        ;;
    single)
        if [[ -z "$RESULTS_FILES" ]]; then
            print_error "Single mode requires --results-files"
            show_usage
            exit 1
        fi
        print_status "Single mode: Analyzing $RESULTS_FILES"
        extract_model_names
        run_analysis
        ;;
    compare)
        if [[ -z "$RESULTS_FILES" ]]; then
            print_error "Compare mode requires --results-files"
            show_usage
            exit 1
        fi
        print_status "Compare mode: Analyzing multiple models"
        if [[ -z "$MODEL_NAMES" ]]; then
            extract_model_names
        fi
        run_analysis
        ;;
    *)
        print_error "Invalid mode: $MODE"
        show_usage
        exit 1
        ;;
esac

print_success "Model performance analysis complete!"
