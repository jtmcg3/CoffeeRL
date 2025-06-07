#!/bin/bash
# Local training script for CoffeeRL-Lite
# Usage: ./scripts/train_local.sh [dev|full|custom]

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
if [ ! -f "src/train_local.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    elif python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        print_success "Apple Silicon GPU (MPS) detected"
    else
        print_warning "No GPU detected - training will be slower"
    fi
}

# Function to run development training
run_dev_training() {
    print_info "Starting development training (small dataset)..."
    uv run python src/train_local.py \
        --dev-mode \
        --max-train-samples 50 \
        --max-eval-samples 10 \
        --epochs 1 \
        --output-dir "./models/coffee-qwen2-dev" \
        --logging-steps 5
}

# Function to run full training
run_full_training() {
    print_info "Starting full training..."
    uv run python src/train_local.py \
        --epochs 3 \
        --output-dir "./models/coffee-qwen2-full" \
        --logging-steps 10
}

# Function to run custom training with user parameters
run_custom_training() {
    print_info "Starting custom training..."
    print_info "You can modify the parameters below or pass them as arguments"

    # Default parameters
    EPOCHS=${EPOCHS:-3}
    BATCH_SIZE=${BATCH_SIZE:-}
    LEARNING_RATE=${LEARNING_RATE:-2e-4}
    OUTPUT_DIR=${OUTPUT_DIR:-"./models/coffee-qwen2-custom"}

    cmd="uv run python src/train_local.py --epochs $EPOCHS --learning-rate $LEARNING_RATE --output-dir $OUTPUT_DIR"

    if [ -n "$BATCH_SIZE" ]; then
        cmd="$cmd --batch-size $BATCH_SIZE"
    fi

    if [ "$DEV_MODE" = "true" ]; then
        cmd="$cmd --dev-mode"
    fi

    if [ "$NO_EVAL" = "true" ]; then
        cmd="$cmd --no-eval"
    fi

    if [ "$FORCE_CPU" = "true" ]; then
        cmd="$cmd --force-cpu"
    fi

    print_info "Running: $cmd"
    eval $cmd
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [mode] [options]"
    echo ""
    echo "Modes:"
    echo "  dev     - Quick development training (1 epoch, small dataset)"
    echo "  full    - Full training (3 epochs, complete dataset)"
    echo "  custom  - Custom training (use environment variables)"
    echo ""
    echo "Environment variables for custom mode:"
    echo "  EPOCHS=3              - Number of training epochs"
    echo "  BATCH_SIZE=2          - Training batch size (auto-detected if not set)"
    echo "  LEARNING_RATE=2e-4    - Learning rate"
    echo "  OUTPUT_DIR=./models/  - Output directory"
    echo "  DEV_MODE=true         - Enable development mode"
    echo "  NO_EVAL=true          - Skip evaluation"
    echo "  FORCE_CPU=true        - Force CPU training"
    echo ""
    echo "Examples:"
    echo "  $0 dev                                    # Quick development run"
    echo "  $0 full                                   # Full training"
    echo "  EPOCHS=5 $0 custom                       # Custom with 5 epochs"
    echo "  DEV_MODE=true EPOCHS=2 $0 custom         # Custom dev mode"
}

# Main script
main() {
    echo "🚀 CoffeeRL-Lite Local Training Script"
    echo "======================================"

    # Check GPU
    check_gpu
    echo ""

    # Parse mode
    MODE=${1:-""}

    case $MODE in
        "dev")
            run_dev_training
            ;;
        "full")
            run_full_training
            ;;
        "custom")
            run_custom_training
            ;;
        "help"|"-h"|"--help")
            show_usage
            exit 0
            ;;
        "")
            print_warning "No mode specified. Use 'dev', 'full', 'custom', or 'help'"
            echo ""
            show_usage
            exit 1
            ;;
        *)
            print_error "Unknown mode: $MODE"
            echo ""
            show_usage
            exit 1
            ;;
    esac

    if [ $? -eq 0 ]; then
        print_success "Training completed successfully!"
        echo ""
        print_info "Next steps:"
        echo "  1. Check training logs: tensorboard --logdir ./models/"
        echo "  2. Test the model: python src/evaluate_local.py --model-path ./models/[model-dir]"
        echo "  3. Run inference: python src/inference.py --model-path ./models/[model-dir]"
    else
        print_error "Training failed!"
        exit 1
    fi
}

# Run main function
main "$@"
