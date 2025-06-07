# CoffeeRL Automated Training and Evaluation Pipeline

This document describes the automated training and evaluation pipeline for the CoffeeRL project, which provides a streamlined way to train and evaluate Qwen2 models for coffee brewing recommendations.

## Overview

The pipeline consists of three main components:

1. **Training Script** (`src/train_local.py`) - Handles model training with QLoRA
2. **Evaluation Script** (`src/evaluate_local.py`) - Evaluates trained models
3. **Pipeline Orchestrator** (`scripts/run_experiment.sh`) - Automates the complete workflow
4. **Makefile** - Provides convenient shortcuts for common operations

## Quick Start

### Prerequisites

Ensure you have the required datasets:

```bash
make check-data
```

This should show:
```
✅ Training dataset found
✅ Validation dataset found
```

### Basic Usage

#### 1. Quick Development Run

For rapid iteration and testing:

```bash
# Using Makefile (recommended)
make pipeline-dev

# Or using the script directly
./scripts/run_experiment.sh --training-mode dev --evaluation-mode quick
```

This will:
- Train Qwen2-0.5B on 100 samples (dev mode)
- Evaluate on 50 samples (quick mode)
- Complete in ~5-10 minutes

#### 2. Full Training and Evaluation

For production-quality results:

```bash
# Using Makefile
make pipeline-full

# Or using the script directly
./scripts/run_experiment.sh --training-mode full --evaluation-mode full
```

This will:
- Train Qwen2-0.5B on the complete dataset
- Evaluate on the full validation set
- Take 30-60 minutes depending on hardware

#### 3. Hyperparameter Optimization

To find the best hyperparameters:

```bash
# Using Makefile
make hyperparameter-sweep

# Or using the script directly
./scripts/run_experiment.sh --hyperparameter-sweep
```

## Pipeline Components

### Training Script (`src/train_local.py`)

Handles model training with the following features:

- **Platform-aware configuration**: Automatically detects hardware and optimizes settings
- **Development mode**: Quick training with data subsets for testing
- **Comprehensive logging**: Detailed progress tracking and metadata saving
- **Error handling**: Robust error recovery and user-friendly messages

Key arguments:
- `--model-name`: Qwen2 model to use (Qwen/Qwen2-0.5B or Qwen/Qwen2-1.5B)
- `--dev-mode`: Enable development mode with reduced data
- `--max-samples`: Limit training samples
- `--output-dir`: Where to save the trained model

### Evaluation Script (`src/evaluate_local.py`)

Evaluates trained models with:

- **Multiple evaluation modes**: Quick testing or comprehensive evaluation
- **Robust output parsing**: Handles various model output formats
- **Detailed metrics**: Grind change and extraction accuracy
- **Prediction saving**: Optional detailed prediction logging

Key arguments:
- `--model-path`: Path to trained model
- `--quick-eval`: Fast evaluation with subset of data
- `--max-samples`: Limit evaluation samples
- `--save-predictions`: Save individual predictions

### Pipeline Orchestrator (`scripts/run_experiment.sh`)

Automates the complete workflow:

- **Prerequisite checking**: Verifies datasets and dependencies
- **Experiment management**: Creates organized output directories
- **Progress monitoring**: Real-time status updates with colored output
- **Report generation**: Automatic experiment documentation
- **Error handling**: Graceful failure recovery

## Makefile Targets

The Makefile provides convenient shortcuts:

### Development Workflow
```bash
make install          # Install dependencies
make test             # Run all tests
make lint             # Check code quality
make format           # Format code
make clean            # Clean up generated files
```

### Training & Evaluation
```bash
make train-dev        # Quick development training
make train-full       # Full training
make eval-quick       # Quick evaluation
make eval-full        # Full evaluation
```

### Automated Pipelines
```bash
make pipeline-dev     # Complete dev pipeline
make pipeline-full    # Complete full pipeline
make hyperparameter-sweep  # Hyperparameter optimization
```

### Data Operations
```bash
make check-data       # Verify datasets exist
make data-stats       # Show dataset statistics
```

### Docker Operations
```bash
make docker-build     # Build development image
make docker-test      # Run tests in Docker
make docker-train     # Run training in Docker
```

## Advanced Usage

### Custom Experiments

You can customize experiments with various options:

```bash
# Custom experiment name
./scripts/run_experiment.sh \
    --experiment-name my-custom-experiment \
    --training-mode full \
    --evaluation-mode full

# Use larger model (requires more memory)
./scripts/run_experiment.sh \
    --model-size 1.5B \
    --training-mode dev

# Evaluation only (skip training)
./scripts/run_experiment.sh \
    --skip-training \
    --evaluation-mode full \
    --output-dir experiments/existing-model

# Save detailed predictions
./scripts/run_experiment.sh \
    --training-mode dev \
    --evaluation-mode quick \
    --save-predictions
```

### Experiment Management

```bash
# List all experiments
make list-experiments

# Clean old experiments (>7 days)
make clean-experiments

# View experiment results
cat experiments/my-experiment/experiment_report.md
```

### Performance Optimization

For different hardware configurations:

#### CPU-only Training
```bash
# Smaller batch sizes for CPU
./scripts/run_experiment.sh \
    --training-mode dev \
    --model-size 0.5B
```

#### GPU Training
```bash
# Larger model for GPU
./scripts/run_experiment.sh \
    --training-mode full \
    --model-size 1.5B
```

#### Docker Training (Linux/CUDA)
```bash
make docker-train
```

## Output Structure

Each experiment creates a structured output directory:

```
experiments/
└── coffee-qwen2-20241207-143022/
    ├── model/                    # Trained model files
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── training_args.bin
    ├── training.log              # Training progress log
    ├── evaluation.log            # Evaluation progress log
    ├── evaluation_results.json   # Detailed metrics
    └── experiment_report.md      # Summary report
```

### Evaluation Results Format

The `evaluation_results.json` contains:

```json
{
  "metrics": {
    "grind_accuracy": 0.85,
    "extraction_accuracy": 0.78,
    "average_accuracy": 0.815
  },
  "evaluation_info": {
    "total_samples": 100,
    "evaluation_time_seconds": 45.2,
    "model_path": "/path/to/model"
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Missing Dependencies
```bash
# Reinstall dependencies
make install
```

#### 2. Missing Datasets
```bash
# Check dataset status
make check-data

# If missing, run data preparation scripts
python scripts/process_community_data.py
```

#### 3. Out of Memory Errors
```bash
# Use smaller model or dev mode
./scripts/run_experiment.sh --model-size 0.5B --training-mode dev
```

#### 4. Permission Errors
```bash
# Make script executable
chmod +x scripts/run_experiment.sh
```

### Debug Mode

For detailed debugging:

```bash
# Enable verbose logging
COFFEERL_LOG_LEVEL=DEBUG ./scripts/run_experiment.sh --training-mode dev
```

### Performance Monitoring

Monitor system resources during training:

```bash
# In another terminal
watch -n 1 'nvidia-smi; echo ""; ps aux | grep python'
```

## Integration with CI/CD

The pipeline supports automated testing:

```bash
# CI checks
make ci-check

# Full development cycle
make dev-cycle
```

## Best Practices

1. **Start with dev mode** for quick iteration
2. **Use meaningful experiment names** for organization
3. **Monitor resource usage** during training
4. **Save experiment reports** for reproducibility
5. **Clean old experiments** regularly to save disk space
6. **Use Docker** for consistent environments
7. **Run hyperparameter sweeps** for optimal performance

## Performance Expectations

### Development Mode (--training-mode dev)
- **Training time**: 2-5 minutes
- **Evaluation time**: 30-60 seconds
- **Memory usage**: 2-4 GB
- **Accuracy**: 70-80% (limited data)

### Full Mode (--training-mode full)
- **Training time**: 20-45 minutes
- **Evaluation time**: 2-5 minutes
- **Memory usage**: 4-8 GB
- **Accuracy**: 80-90% (full data)

### Hardware Recommendations

- **Minimum**: 8 GB RAM, CPU-only
- **Recommended**: 16 GB RAM, GPU with 6+ GB VRAM
- **Optimal**: 32 GB RAM, GPU with 12+ GB VRAM

## Contributing

When adding new features to the pipeline:

1. Update the relevant script (training, evaluation, or orchestrator)
2. Add corresponding tests in `tests/test_pipeline.py`
3. Update Makefile targets if needed
4. Update this documentation
5. Test with both dev and full modes

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review experiment logs in the output directory
3. Run tests to verify setup: `make test`
4. Check system requirements and dependencies
