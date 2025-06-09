# CoffeeRL-Lite

A lightweight reinforcement learning framework for coffee optimization using transformer models and PEFT (Parameter-Efficient Fine-Tuning). Features include supervised fine-tuning, reinforcement learning with PPO, and comprehensive model comparison tools.

## Interface Preview

<img src="docs/static/gradio_interface.png" alt="Gradio Interface" width="600"/>

*Preview of the Gradio web interface for fine-tuning coffee parameters*

## Key Features

- 🤖 **Reinforcement Learning**: PPO-based training with TRL library
- 📊 **Model Comparison**: Comprehensive performance comparison between models
- 🔄 **Batch Training**: Automated weekly batch training with data accumulation
- 🤗 **Hugging Face Integration**: Model versioning and sharing via HF Hub
- 📈 **Performance Tracking**: Detailed metrics and reward calculation
- 🧪 **Comprehensive Testing**: Full test suite with 100% coverage for core components

## Project Structure

```
CoffeRL/
├── data/           # Datasets and data processing scripts
├── models/         # Saved model checkpoints and configurations
├── src/            # Source code for the CoffeeRL-Lite implementation
│   ├── model_comparator.py    # Model performance comparison system
│   ├── batch_trainer.py       # Batch training management
│   ├── hf_model_manager.py    # Hugging Face Hub integration
│   ├── train_rl.py           # Reinforcement learning training
│   ├── reward_calculator.py   # Reward function implementation
│   └── ...
├── notebooks/      # Jupyter notebooks for exploration and analysis
├── config/         # Configuration files for training and deployment
├── scripts/        # Utility scripts and automation
├── tests/          # Test suite with comprehensive coverage
├── docs/           # Documentation
├── checkpoints/    # Training checkpoints and batch results
├── .venv/          # Virtual environment (managed by UV)
├── pyproject.toml  # Project configuration and dependencies
├── uv.lock         # Dependency lock file
├── Dockerfile      # Docker configuration for Linux deployment
├── docker-compose.yml # Docker Compose for easy deployment
└── README.md       # This file
```

## CLI Tools

### Model Comparison Tool

Compare performance between different model versions or checkpoints:

```bash
# Quick comparison (10 samples for fast testing)
uv run python src/model_comparator.py \
  --model1 checkpoints/batch_training/batch_1 \
  --model2 checkpoints/batch_training/batch_2 \
  --max-samples 10

# Full comparison with 50 samples
uv run python src/model_comparator.py \
  --model1 checkpoints/batch_training/batch_1 \
  --model2 checkpoints/batch_training/batch_2 \
  --quick

# Compare HF Hub models
uv run python src/model_comparator.py \
  --model1 batch-1 --model1-hf \
  --model2 batch-2 --model2-hf \
  --dataset data/processed/coffee_validation_dataset

# Save comparison results
uv run python src/model_comparator.py \
  --model1 path/to/model1 \
  --model2 path/to/model2 \
  --output comparison_results.json
```

### Batch Training Management

Manage batch training workflows:

```bash
# Check training status
uv run python src/batch_trainer.py status

# Run batch training
uv run python src/batch_trainer.py train --episodes 500

# Add dummy data for testing
uv run python src/batch_trainer.py add-dummy-data --dummy-data-size 100

# View training history
uv run python src/batch_trainer.py history

# Check HF Hub status
uv run python src/batch_trainer.py hf-status
```

### Reinforcement Learning Training

Direct RL training with PPO:

```bash
# Train with default settings
uv run python src/train_rl.py --episodes 100

# Train with custom dataset
uv run python src/train_rl.py \
  --dataset data/processed/coffee_training_dataset \
  --episodes 200 \
  --save-freq 50

# Quick validation run
uv run python src/train_rl.py --episodes 10 --eval-only
```

## Platform Compatibility

### macOS (Current Development Platform)
- ✅ **Supported**: All core ML libraries (transformers, peft, datasets, accelerate, gradio, pandas, torch)
- ✅ **Apple Silicon**: Optimized for M1/M2/M3 chips with MPS acceleration
- ❌ **bitsandbytes**: Not supported on macOS - quantization features disabled
- 🔄 **Workaround**: Use Docker for Linux environment when quantization is needed

### Linux/Windows
- ✅ **Full Support**: All libraries including bitsandbytes for quantization
- ✅ **CUDA**: GPU acceleration supported
- ✅ **Quantization**: 4-bit and 8-bit model quantization available

## Setup

This project uses [UV](https://github.com/astral-sh/uv) for fast Python package management.

### Local Development (macOS/Linux/Windows)

```bash
# Clone the repository
git clone <repository-url>
cd CoffeRL

# Install dependencies with UV
uv sync

# Check platform compatibility
uv run python config/platform_config.py

# Run tests
uv run pytest

# Start development
uv run python main.py
```

### Docker Deployment (Full Linux Support)

For full feature support including bitsandbytes quantization:

```bash
# Development with hot reload
docker-compose --profile dev up

# Production deployment
docker-compose --profile prod up

# GPU-enabled deployment (requires nvidia-docker)
docker-compose --profile gpu up
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/test_model_comparator.py
uv run pytest tests/test_reward_calculator.py

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run tests with verbose output
uv run pytest -v
```

### Test Coverage

- **Model Comparator**: 21 unit tests covering all comparison functionality
- **Reward Calculator**: Comprehensive testing of reward calculation logic
- **RL Environment**: Training loop and environment setup testing
- **Batch Training**: End-to-end batch training workflow testing
- **HF Integration**: Model versioning and Hub interaction testing

## Performance Metrics

The system tracks comprehensive performance metrics:

- **Average Reward**: Overall model performance score
- **Task Completion Rate**: Percentage of successful predictions
- **Reward Standard Deviation**: Consistency of model performance
- **Training Statistics**: Loss curves, convergence metrics
- **Comparison Reports**: Detailed model-vs-model analysis

## Dependencies

### Core ML Libraries
- **transformers**: Hugging Face transformers library
- **peft**: Parameter-Efficient Fine-Tuning
- **datasets**: Hugging Face datasets library
- **accelerate**: Distributed training support
- **torch**: PyTorch deep learning framework
- **trl**: Transformer Reinforcement Learning
- **gradio**: Web UI for model interaction
- **pandas**: Data manipulation

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **isort**: Import sorting
- **mypy**: Type checking
- **pre-commit**: Git hooks for code quality

### Platform-Specific
- **bitsandbytes**: Quantization support (Linux/Windows only)
  - Automatically handled by `config/platform_config.py`
  - Graceful fallback on unsupported platforms
  - Docker deployment available for full support

## Quantization Strategy

### macOS Development
- Quantization features are automatically disabled
- Models run in full precision (float32/bfloat16)
- Development and testing can proceed normally
- Use `config/platform_config.py` for platform-aware code

### Production Deployment
- Use Docker for Linux environment with full bitsandbytes support
- 4-bit and 8-bit quantization available
- GPU acceleration with CUDA
- Automatic platform detection and configuration

### Example Usage

```python
from config.platform_config import get_quantization_config, get_torch_dtype

# Get platform-appropriate quantization config
quant_config = get_quantization_config(use_4bit=True)

# Get optimal torch dtype for platform
torch_dtype = get_torch_dtype()

# Load model with platform-specific settings
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,  # None on macOS
    torch_dtype=torch_dtype,
    device_map="auto"
)
```

## Development Workflow

1. **Local Development**: Use macOS/local environment for development and testing
2. **Model Training**: Use RL training scripts for model improvement
3. **Performance Evaluation**: Use model comparison tools to validate improvements
4. **Batch Processing**: Accumulate data and run batch training weekly
5. **Version Management**: Automatic model versioning via Hugging Face Hub
6. **Code Quality**: Pre-commit hooks ensure consistent code style
7. **Testing**: Comprehensive test suite with platform-aware tests
8. **Deployment**: Docker for production with full quantization support

## Getting Started

1. **Setup Environment**: `uv sync` to install dependencies
2. **Check Compatibility**: `uv run python config/platform_config.py`
3. **Run Tests**: `uv run pytest` to verify installation
4. **Quick Model Comparison**:
   ```bash
   uv run python src/model_comparator.py \
     --model1 checkpoints/batch_training/batch_1 \
     --model2 checkpoints/batch_training/batch_2 \
     --max-samples 10
   ```
5. **Start Training**: `uv run python src/train_rl.py --episodes 10`
6. **Launch Interface**: `uv run python main.py`

## Documentation

- [CLI Tools Reference](docs/CLI_TOOLS.md) - Comprehensive CLI documentation
- [Pipeline Documentation](docs/PIPELINE.md) - Training and evaluation pipelines
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [QLoRA Setup](docs/QLORA_SETUP.md) - Quantization and fine-tuning setup

## Notes

- **bitsandbytes compatibility**: Automatically handled - no manual intervention needed
- **Apple Silicon**: Optimized for M1/M2/M3 with MPS acceleration
- **Memory efficiency**: Platform-appropriate data types selected automatically
- **Cloud deployment**: Docker images work on any cloud provider with GPU support
- **Model Comparison**: Use `--max-samples 10` for fast testing during development

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) document for details.
