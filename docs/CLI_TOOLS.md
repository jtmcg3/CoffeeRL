# CLI Tools Reference

This document provides comprehensive documentation for all command-line tools available in the CoffeRL project.

## Table of Contents

- [Model Comparison Tool](#model-comparison-tool)
- [Batch Training Management](#batch-training-management)
- [Reinforcement Learning Training](#reinforcement-learning-training)
- [Testing and Validation](#testing-and-validation)
- [Configuration and Setup](#configuration-and-setup)

## Model Comparison Tool

The model comparison tool (`src/model_comparator.py`) provides comprehensive performance comparison between different model versions or checkpoints.

### Basic Usage

```bash
# Quick comparison with 10 samples (recommended for development)
uv run python src/model_comparator.py \
  --model1 checkpoints/batch_training/batch_1 \
  --model2 checkpoints/batch_training/batch_2 \
  --max-samples 10

# Full comparison with 50 samples (default)
uv run python src/model_comparator.py \
  --model1 checkpoints/batch_training/batch_1 \
  --model2 checkpoints/batch_training/batch_2 \
  --quick
```

### Advanced Options

#### Hugging Face Hub Integration

```bash
# Compare models from Hugging Face Hub
uv run python src/model_comparator.py \
  --model1 batch-1 --model1-hf \
  --model2 batch-2 --model2-hf \
  --dataset data/processed/coffee_validation_dataset

# Mix local and HF Hub models
uv run python src/model_comparator.py \
  --model1 checkpoints/batch_training/batch_1 \
  --model2 batch-2 --model2-hf
```

#### Custom Datasets and Output

```bash
# Use custom validation dataset
uv run python src/model_comparator.py \
  --model1 path/to/model1 \
  --model2 path/to/model2 \
  --dataset data/processed/custom_validation_dataset

# Save comparison results to file
uv run python src/model_comparator.py \
  --model1 path/to/model1 \
  --model2 path/to/model2 \
  --output comparison_results.json

# Combine custom dataset and output
uv run python src/model_comparator.py \
  --model1 path/to/model1 \
  --model2 path/to/model2 \
  --dataset data/processed/coffee_validation_dataset \
  --output detailed_comparison.json \
  --max-samples 100
```

### Command-Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--model1` | Path to first model | Required | `checkpoints/batch_training/batch_1` |
| `--model2` | Path to second model | Required | `checkpoints/batch_training/batch_2` |
| `--model1-hf` | Load model1 from HF Hub | `False` | `--model1-hf` |
| `--model2-hf` | Load model2 from HF Hub | `False` | `--model2-hf` |
| `--dataset` | Path to validation dataset | Auto-detected | `data/processed/coffee_validation_dataset` |
| `--max-samples` | Maximum samples to evaluate | `None` | `--max-samples 10` |
| `--quick` | Use 50 samples for quick comparison | `False` | `--quick` |
| `--output` | Save results to JSON file | `None` | `--output results.json` |

### Output Format

The tool provides detailed comparison metrics:

```
Model Comparison Results
========================

Model 1: checkpoints/batch_training/batch_1
Model 2: checkpoints/batch_training/batch_2

Performance Metrics:
Model 1 - Average Reward: 0.1833, Completion Rate: 80.00%, Std Dev: 0.2357
Model 2 - Average Reward: 0.1067, Completion Rate: 100.00%, Std Dev: 0.1789

Comparison:
Model 1 wins with 0.0767 higher average reward
Model 1 has 20.00% lower completion rate
Model 1 has 0.0568 higher reward standard deviation

Improvement Analysis:
Average Reward: +71.85% improvement for Model 1
Completion Rate: -20.00% change for Model 1
```

### Performance Considerations

- **Development Testing**: Use `--max-samples 10` for fast iteration (~4 minutes)
- **Thorough Evaluation**: Use `--quick` for 50 samples (~10-15 minutes)
- **Full Evaluation**: Omit sample limits for complete dataset evaluation (30+ minutes)

## Batch Training Management

The batch training tool (`src/batch_trainer.py`) manages automated training workflows and data accumulation.

### Status and Information

```bash
# Check current training status
uv run python src/batch_trainer.py status

# View training history
uv run python src/batch_trainer.py history

# Check Hugging Face Hub status
uv run python src/batch_trainer.py hf-status
```

### Training Operations

```bash
# Run batch training with default settings
uv run python src/batch_trainer.py train

# Run batch training with custom episode count
uv run python src/batch_trainer.py train --episodes 500

# Run training with specific configuration
uv run python src/batch_trainer.py train \
  --episodes 1000 \
  --save-freq 100 \
  --eval-freq 50
```

### Data Management

```bash
# Add dummy data for testing (100 samples)
uv run python src/batch_trainer.py add-dummy-data

# Add custom amount of dummy data
uv run python src/batch_trainer.py add-dummy-data --dummy-data-size 250

# Check data accumulation status
uv run python src/batch_trainer.py data-status
```

### Command-Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--episodes` | Number of training episodes | `500` | `--episodes 1000` |
| `--save-freq` | Checkpoint save frequency | `50` | `--save-freq 100` |
| `--eval-freq` | Evaluation frequency | `25` | `--eval-freq 50` |
| `--dummy-data-size` | Size of dummy data to generate | `100` | `--dummy-data-size 250` |

## Reinforcement Learning Training

The RL training tool (`src/train_rl.py`) provides direct access to the reinforcement learning training loop.

### Basic Training

```bash
# Train with default settings (100 episodes)
uv run python src/train_rl.py

# Train with custom episode count
uv run python src/train_rl.py --episodes 200

# Quick validation run
uv run python src/train_rl.py --episodes 10
```

### Advanced Training Options

```bash
# Train with custom dataset
uv run python src/train_rl.py \
  --dataset data/processed/coffee_training_dataset \
  --episodes 200 \
  --save-freq 50

# Evaluation-only mode (no training)
uv run python src/train_rl.py --episodes 10 --eval-only

# Training with custom model path
uv run python src/train_rl.py \
  --model-path checkpoints/custom_model \
  --episodes 150 \
  --learning-rate 1e-5
```

### Command-Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--episodes` | Number of training episodes | `100` | `--episodes 200` |
| `--dataset` | Path to training dataset | Auto-detected | `data/processed/coffee_training_dataset` |
| `--save-freq` | Checkpoint save frequency | `25` | `--save-freq 50` |
| `--eval-only` | Run evaluation without training | `False` | `--eval-only` |
| `--model-path` | Path to model checkpoint | Auto-detected | `checkpoints/custom_model` |
| `--learning-rate` | Learning rate for training | `5e-6` | `--learning-rate 1e-5` |

## Testing and Validation

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/test_model_comparator.py
uv run pytest tests/test_reward_calculator.py
uv run pytest tests/test_batch_trainer.py

# Run tests with verbose output
uv run pytest -v

# Run tests with coverage reporting
uv run pytest --cov=src --cov-report=html

# Run tests with coverage and show missing lines
uv run pytest --cov=src --cov-report=term-missing
```

### Test Categories

#### Model Comparator Tests
```bash
# Run all model comparator tests (21 test cases)
uv run pytest tests/test_model_comparator.py -v

# Run specific test categories
uv run pytest tests/test_model_comparator.py::TestModelComparator::test_load_model_local -v
uv run pytest tests/test_model_comparator.py::TestModelComparator::test_compare_models -v
```

#### Reward Calculator Tests
```bash
# Run reward calculation tests
uv run pytest tests/test_reward_calculator.py -v

# Test specific reward functions
uv run pytest tests/test_reward_calculator.py::test_calculate_reward -v
```

#### Integration Tests
```bash
# Run end-to-end integration tests
uv run pytest tests/test_integration.py -v

# Test full training pipeline
uv run pytest tests/test_training_pipeline.py -v
```

### Test Coverage Analysis

```bash
# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# Generate XML coverage report (for CI/CD)
uv run pytest --cov=src --cov-report=xml

# Show coverage summary in terminal
uv run pytest --cov=src --cov-report=term
```

### Performance Testing

```bash
# Run performance benchmarks
uv run pytest tests/test_performance.py -v

# Test model comparison speed
uv run python src/model_comparator.py \
  --model1 checkpoints/batch_training/batch_1 \
  --model2 checkpoints/batch_training/batch_2 \
  --max-samples 5  # Minimal test for speed

# Benchmark training performance
uv run python src/train_rl.py --episodes 5 --benchmark
```

## Configuration and Setup

### Platform Configuration

```bash
# Check platform compatibility
uv run python config/platform_config.py

# Test quantization support
uv run python config/platform_config.py --test-quantization

# Show platform-specific settings
uv run python config/platform_config.py --show-config
```

### Environment Setup

```bash
# Install dependencies
uv sync

# Install development dependencies
uv sync --dev

# Update dependencies
uv sync --upgrade

# Check dependency status
uv pip list
```

### Model Management

```bash
# List available models
uv run python src/hf_model_manager.py list

# Download model from HF Hub
uv run python src/hf_model_manager.py download --model-name batch-1

# Upload model to HF Hub
uv run python src/hf_model_manager.py upload \
  --model-path checkpoints/batch_training/batch_1 \
  --model-name batch-1-updated

# Check model status
uv run python src/hf_model_manager.py status --model-name batch-1
```

## Common Workflows

### Development Workflow

```bash
# 1. Quick model comparison for development
uv run python src/model_comparator.py \
  --model1 checkpoints/batch_training/batch_1 \
  --model2 checkpoints/batch_training/batch_2 \
  --max-samples 10

# 2. Run tests to ensure functionality
uv run pytest tests/test_model_comparator.py -v

# 3. Quick training validation
uv run python src/train_rl.py --episodes 5

# 4. Check training status
uv run python src/batch_trainer.py status
```

### Production Validation Workflow

```bash
# 1. Full model comparison
uv run python src/model_comparator.py \
  --model1 checkpoints/batch_training/batch_1 \
  --model2 checkpoints/batch_training/batch_2 \
  --output production_comparison.json

# 2. Run comprehensive tests
uv run pytest --cov=src --cov-report=html

# 3. Validate training pipeline
uv run python src/train_rl.py --episodes 50 --eval-only

# 4. Check HF Hub integration
uv run python src/batch_trainer.py hf-status
```

### Debugging Workflow

```bash
# 1. Run tests with verbose output
uv run pytest -v -s

# 2. Test specific functionality
uv run python src/model_comparator.py \
  --model1 checkpoints/batch_training/batch_1 \
  --model2 checkpoints/batch_training/batch_2 \
  --max-samples 1  # Minimal test

# 3. Check platform configuration
uv run python config/platform_config.py

# 4. Validate data integrity
uv run python src/batch_trainer.py data-status
```

## Error Handling and Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Verify model paths exist
   ls -la checkpoints/batch_training/

   # Test model loading
   uv run python -c "from src.model_comparator import ModelComparator; mc = ModelComparator(); mc.load_model('checkpoints/batch_training/batch_1')"
   ```

2. **Dataset Issues**
   ```bash
   # Check dataset availability
   ls -la data/processed/

   # Validate dataset format
   uv run python -c "from datasets import load_from_disk; ds = load_from_disk('data/processed/coffee_validation_dataset'); print(ds)"
   ```

3. **Memory Issues**
   ```bash
   # Use smaller sample sizes
   uv run python src/model_comparator.py \
     --model1 path/to/model1 \
     --model2 path/to/model2 \
     --max-samples 5

   # Check platform configuration
   uv run python config/platform_config.py
   ```

4. **HF Hub Authentication**
   ```bash
   # Login to Hugging Face
   huggingface-cli login

   # Test HF Hub access
   uv run python src/hf_model_manager.py status --model-name batch-1
   ```

### Performance Optimization

1. **Fast Development Testing**
   - Use `--max-samples 10` for model comparisons
   - Use `--episodes 5` for training validation
   - Run specific test modules instead of full test suite

2. **Memory Optimization**
   - Use quantization on supported platforms
   - Reduce batch sizes for training
   - Use gradient checkpointing for large models

3. **Speed Optimization**
   - Use MPS acceleration on Apple Silicon
   - Enable CUDA on supported GPUs
   - Use appropriate torch data types per platform

## Best Practices

1. **Development**
   - Always run tests before committing changes
   - Use quick comparison modes during development
   - Validate changes with minimal sample sizes first

2. **Production**
   - Run full comparisons before deploying models
   - Maintain comprehensive test coverage
   - Use version control for model checkpoints

3. **Debugging**
   - Start with minimal test cases
   - Check platform compatibility first
   - Validate data integrity before training

4. **Performance**
   - Use appropriate sample sizes for your use case
   - Monitor memory usage during training
   - Leverage platform-specific optimizations
