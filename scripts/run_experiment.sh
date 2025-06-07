#!/bin/bash

# CoffeeRL Automated Training and Evaluation Pipeline
# This script orchestrates the complete training and evaluation workflow

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
EXPERIMENT_NAME="coffee-qwen2-$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="experiments/${EXPERIMENT_NAME}"
MODEL_SIZE="0.5B"
TRAINING_MODE="dev"
EVALUATION_MODE="quick"
SKIP_TRAINING=false
SKIP_EVALUATION=false
CLEANUP_CHECKPOINTS=false
SAVE_PREDICTIONS=false
HYPERPARAMETER_SWEEP=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

CoffeeRL Automated Training and Evaluation Pipeline

OPTIONS:
    --experiment-name NAME    Set experiment name (default: coffee-qwen2-TIMESTAMP)
    --model-size SIZE         Model size: 0.5B or 1.5B (default: 0.5B)
    --training-mode MODE      Training mode: dev, full, custom (default: dev)
    --evaluation-mode MODE    Evaluation mode: quick, full, custom (default: quick)
    --output-dir DIR          Output directory (default: experiments/EXPERIMENT_NAME)
    --skip-training           Skip training phase
    --skip-evaluation         Skip evaluation phase
    --cleanup-checkpoints     Remove intermediate checkpoints after training
    --save-predictions        Save individual predictions during evaluation
    --hyperparameter-sweep    Run hyperparameter sweep (multiple training runs)
    --help                    Show this help message

EXAMPLES:
    # Quick development run
    $0 --training-mode dev --evaluation-mode quick

    # Full training and evaluation
    $0 --training-mode full --evaluation-mode full --model-size 1.5B

    # Evaluation only (using existing model)
    $0 --skip-training --evaluation-mode full --output-dir experiments/existing-model

    # Hyperparameter sweep
    $0 --hyperparameter-sweep --training-mode full

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment-name)
            EXPERIMENT_NAME="$2"
            OUTPUT_DIR="experiments/${EXPERIMENT_NAME}"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --training-mode)
            TRAINING_MODE="$2"
            shift 2
            ;;
        --evaluation-mode)
            EVALUATION_MODE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --cleanup-checkpoints)
            CLEANUP_CHECKPOINTS=true
            shift
            ;;
        --save-predictions)
            SAVE_PREDICTIONS=true
            shift
            ;;
        --hyperparameter-sweep)
            HYPERPARAMETER_SWEEP=true
            shift
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

# Function to check prerequisites
check_prerequisites() {
    print_header "CHECKING PREREQUISITES"

    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install uv first."
        exit 1
    fi
    print_success "uv is installed"

    # Check if required data directories exist
    if [[ ! -d "data/processed/coffee_training_dataset" ]]; then
        print_error "Training dataset not found. Please run data preparation first."
        exit 1
    fi
    print_success "Training dataset found"

    if [[ ! -d "data/processed/coffee_validation_dataset" ]]; then
        print_error "Validation dataset not found. Please run data preparation first."
        exit 1
    fi
    print_success "Validation dataset found"

    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        print_warning "No NVIDIA GPU detected. Training will use CPU/MPS."
    fi

    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    print_success "Output directory created: ${OUTPUT_DIR}"
}

# Function to run training
run_training() {
    print_header "TRAINING PHASE"

    local model_output_dir="${OUTPUT_DIR}/model"
    local training_args=""

    # Build training arguments
    case $TRAINING_MODE in
        "dev")
            training_args="--dev-mode --max-train-samples 100"
            ;;
        "full")
            training_args=""
            ;;
        "custom")
            # Add custom arguments here if needed
            training_args=""
            ;;
        *)
            print_error "Invalid training mode: $TRAINING_MODE"
            exit 1
            ;;
    esac

    # Add model size
    if [[ "$MODEL_SIZE" == "1.5B" ]]; then
        training_args="$training_args --model-name Qwen/Qwen2-1.5B"
    else
        training_args="$training_args --model-name Qwen/Qwen2-0.5B"
    fi

    # Add output directory
    training_args="$training_args --output-dir $model_output_dir"

    print_status "Starting training with arguments: $training_args"
    print_status "Training mode: $TRAINING_MODE"
    print_status "Model size: $MODEL_SIZE"
    print_status "Output directory: $model_output_dir"

    # Run training
    if uv run python src/train_local.py $training_args 2>&1 | tee "${OUTPUT_DIR}/training.log"; then
        print_success "Training completed successfully"
    else
        print_error "Training failed. Check ${OUTPUT_DIR}/training.log for details."
        exit 1
    fi

    # Cleanup checkpoints if requested
    if [[ "$CLEANUP_CHECKPOINTS" == "true" ]]; then
        print_status "Cleaning up intermediate checkpoints..."
        find "$model_output_dir" -name "checkpoint-*" -type d | head -n -1 | xargs rm -rf
        print_success "Checkpoint cleanup completed"
    fi
}

# Function to run evaluation
run_evaluation() {
    print_header "EVALUATION PHASE"

    local model_path="${OUTPUT_DIR}/model"
    local eval_output="${OUTPUT_DIR}/evaluation_results.json"
    local eval_args=""

    # Check if model exists
    if [[ ! -d "$model_path" ]]; then
        print_error "Model not found at $model_path"
        exit 1
    fi

    # Build evaluation arguments
    case $EVALUATION_MODE in
        "quick")
            eval_args="--quick-eval --max-samples 50"
            ;;
        "full")
            eval_args=""
            ;;
        "custom")
            # Add custom arguments here if needed
            eval_args=""
            ;;
        *)
            print_error "Invalid evaluation mode: $EVALUATION_MODE"
            exit 1
            ;;
    esac

    # Add save predictions if requested
    if [[ "$SAVE_PREDICTIONS" == "true" ]]; then
        eval_args="$eval_args --save-predictions"
    fi

    # Add paths
    eval_args="$eval_args --model-path $model_path --output-file $eval_output"

    print_status "Starting evaluation with arguments: $eval_args"
    print_status "Evaluation mode: $EVALUATION_MODE"
    print_status "Model path: $model_path"
    print_status "Output file: $eval_output"

    # Run evaluation
    if uv run python src/evaluate_local.py $eval_args 2>&1 | tee "${OUTPUT_DIR}/evaluation.log"; then
        print_success "Evaluation completed successfully"

        # Display summary if jq is available
        if command -v jq &> /dev/null && [[ -f "$eval_output" ]]; then
            print_header "EVALUATION SUMMARY"
            echo "📊 Metrics:"
            jq -r '.metrics | to_entries[] | "  \(.key): \(.value | . * 100 | round / 100)"' "$eval_output"
            echo ""
            echo "ℹ️  Evaluation Info:"
            jq -r '.evaluation_info | to_entries[] | "  \(.key): \(.value)"' "$eval_output"
        fi
    else
        print_error "Evaluation failed. Check ${OUTPUT_DIR}/evaluation.log for details."
        exit 1
    fi
}

# Function to run hyperparameter sweep
run_hyperparameter_sweep() {
    print_header "HYPERPARAMETER SWEEP"

    local sweep_configs=(
        "lr=1e-4,epochs=2,batch=2"
        "lr=2e-4,epochs=3,batch=2"
        "lr=5e-4,epochs=2,batch=4"
    )

    local best_score=0
    local best_config=""
    local sweep_results="${OUTPUT_DIR}/sweep_results.json"

    echo "[]" > "$sweep_results"

    for i in "${!sweep_configs[@]}"; do
        local config="${sweep_configs[$i]}"
        local run_name="sweep_run_$((i+1))"
        local run_dir="${OUTPUT_DIR}/${run_name}"

        print_status "Running sweep configuration $((i+1))/${#sweep_configs[@]}: $config"

        # Parse configuration
        IFS=',' read -ra PARAMS <<< "$config"
        local lr_arg=""
        local epochs_arg=""
        local batch_arg=""

        for param in "${PARAMS[@]}"; do
            IFS='=' read -ra PAIR <<< "$param"
            case "${PAIR[0]}" in
                "lr") lr_arg="--learning-rate ${PAIR[1]}" ;;
                "epochs") epochs_arg="--num-epochs ${PAIR[1]}" ;;
                "batch") batch_arg="--batch-size ${PAIR[1]}" ;;
            esac
        done

        # Run training for this configuration
        mkdir -p "$run_dir"
        local model_output_dir="${run_dir}/model"
        local training_args="--output-dir $model_output_dir $lr_arg $epochs_arg $batch_arg"

        if [[ "$MODEL_SIZE" == "1.5B" ]]; then
            training_args="$training_args --model-name Qwen/Qwen2-1.5B"
        else
            training_args="$training_args --model-name Qwen/Qwen2-0.5B"
        fi

        if uv run python src/train_local.py $training_args > "${run_dir}/training.log" 2>&1; then
            # Run evaluation
            local eval_output="${run_dir}/evaluation_results.json"
            if uv run python src/evaluate_local.py --model-path "$model_output_dir" --output-file "$eval_output" --quick-eval > "${run_dir}/evaluation.log" 2>&1; then
                # Extract score and update best
                if command -v jq &> /dev/null; then
                    local score=$(jq -r '.metrics.average_accuracy' "$eval_output")
                    if (( $(echo "$score > $best_score" | bc -l) )); then
                        best_score=$score
                        best_config=$config
                    fi

                    # Add to sweep results
                    local result=$(jq -n --arg config "$config" --arg score "$score" --arg run "$run_name" '{config: $config, score: ($score | tonumber), run: $run}')
                    jq ". += [$result]" "$sweep_results" > "${sweep_results}.tmp" && mv "${sweep_results}.tmp" "$sweep_results"
                fi

                print_success "Sweep run $((i+1)) completed with score: $score"
            else
                print_warning "Evaluation failed for sweep run $((i+1))"
            fi
        else
            print_warning "Training failed for sweep run $((i+1))"
        fi
    done

    print_header "SWEEP RESULTS"
    print_success "Best configuration: $best_config (score: $best_score)"

    if command -v jq &> /dev/null; then
        echo "📊 All results:"
        jq -r '.[] | "  \(.config): \(.score)"' "$sweep_results"
    fi
}

# Function to generate experiment report
generate_report() {
    print_header "GENERATING EXPERIMENT REPORT"

    local report_file="${OUTPUT_DIR}/experiment_report.md"

    cat > "$report_file" << EOF
# CoffeeRL Experiment Report

**Experiment Name:** ${EXPERIMENT_NAME}
**Date:** $(date)
**Model Size:** ${MODEL_SIZE}

## Configuration

- **Training Mode:** ${TRAINING_MODE}
- **Evaluation Mode:** ${EVALUATION_MODE}
- **Output Directory:** ${OUTPUT_DIR}
- **Skip Training:** ${SKIP_TRAINING}
- **Skip Evaluation:** ${SKIP_EVALUATION}
- **Hyperparameter Sweep:** ${HYPERPARAMETER_SWEEP}

## System Information

- **OS:** $(uname -s)
- **Architecture:** $(uname -m)
- **Python Version:** $(uv run python --version)

EOF

    # Add GPU information if available
    if command -v nvidia-smi &> /dev/null; then
        echo "- **GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)" >> "$report_file"
        echo "- **GPU Memory:** $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB" >> "$report_file"
    else
        echo "- **GPU:** Not available" >> "$report_file"
    fi

    echo "" >> "$report_file"

    # Add evaluation results if available
    local eval_results="${OUTPUT_DIR}/evaluation_results.json"
    if [[ -f "$eval_results" ]] && command -v jq &> /dev/null; then
        cat >> "$report_file" << EOF
## Evaluation Results

$(jq -r '.metrics | to_entries[] | "- **\(.key | gsub("_"; " ") | ascii_upcase):** \(.value | . * 100 | round / 100)%"' "$eval_results")

### Evaluation Details

$(jq -r '.evaluation_info | to_entries[] | "- **\(.key | gsub("_"; " ") | ascii_upcase):** \(.value)"' "$eval_results")

EOF
    fi

    # Add file listing
    cat >> "$report_file" << EOF
## Generated Files

$(find "$OUTPUT_DIR" -type f -name "*.json" -o -name "*.log" -o -name "*.md" | sed 's|^|- |')

## Notes

This experiment was run using the CoffeeRL automated pipeline.
For more details, check the individual log files in the output directory.

EOF

    print_success "Experiment report generated: $report_file"
}

# Main execution
main() {
    print_header "COFFEERL AUTOMATED TRAINING AND EVALUATION PIPELINE"

    print_status "Experiment: $EXPERIMENT_NAME"
    print_status "Output directory: $OUTPUT_DIR"
    print_status "Model size: $MODEL_SIZE"
    print_status "Training mode: $TRAINING_MODE"
    print_status "Evaluation mode: $EVALUATION_MODE"

    # Check prerequisites
    check_prerequisites

    # Run hyperparameter sweep if requested
    if [[ "$HYPERPARAMETER_SWEEP" == "true" ]]; then
        run_hyperparameter_sweep
    else
        # Run training if not skipped
        if [[ "$SKIP_TRAINING" == "false" ]]; then
            run_training
        else
            print_warning "Skipping training phase"
        fi

        # Run evaluation if not skipped
        if [[ "$SKIP_EVALUATION" == "false" ]]; then
            run_evaluation
        else
            print_warning "Skipping evaluation phase"
        fi
    fi

    # Generate experiment report
    generate_report

    print_header "PIPELINE COMPLETED SUCCESSFULLY"
    print_success "All results saved to: $OUTPUT_DIR"

    # Show quick summary
    if [[ -f "${OUTPUT_DIR}/evaluation_results.json" ]] && command -v jq &> /dev/null; then
        local avg_accuracy=$(jq -r '.metrics.average_accuracy' "${OUTPUT_DIR}/evaluation_results.json")
        print_success "Final average accuracy: $(echo "$avg_accuracy * 100" | bc -l | cut -d. -f1)%"
    fi
}

# Run main function
main "$@"
