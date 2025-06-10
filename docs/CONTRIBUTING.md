# Contributing to CoffeeRL-Lite

Thank you for your interest in contributing to CoffeeRL-Lite! This guide will help you get started with development, testing, and contributing to the project.

## Development Setup

This project uses [UV](https://github.com/astral-sh/uv) for fast Python package management.

### Local Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd CoffeRL

# Install dependencies with UV
uv sync

# Check platform compatibility
uv run python config/platform_config.py

# Run tests to verify setup
uv run pytest
```

### Development Workflow

1. **Local Development**: Use macOS/local environment for development and testing
2. **Model Training**: Use RL training scripts for model improvement
3. **Performance Evaluation**: Use model comparison tools to validate improvements
4. **Batch Processing**: Accumulate data and run batch training weekly
5. **Version Management**: Automatic model versioning via Hugging Face Hub
6. **Code Quality**: Pre-commit hooks ensure consistent code style
7. **Testing**: Comprehensive test suite with platform-aware tests
8. **Deployment**: Docker for production with full quantization support

## Project Structure (Developer View)

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
└── README.md       # User-facing documentation
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

### Writing Tests

- Follow existing test patterns in the `tests/` directory
- Ensure platform-aware tests for macOS/Linux compatibility
- Mock external dependencies (HuggingFace Hub, file I/O)
- Test both success and error cases
- Maintain high test coverage for core components

## Code Quality

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

### Code Style

- **black**: Code formatting
- **flake8**: Linting
- **isort**: Import sorting
- **mypy**: Type checking

### Development Tools

```bash
# Format code
uv run black src/ tests/

# Check linting
uv run flake8 src/ tests/

# Sort imports
uv run isort src/ tests/

# Type checking
uv run mypy src/
```

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

## Platform Development

### macOS Development
- Quantization features are automatically disabled
- Models run in full precision (float32/bfloat16)
- Development and testing can proceed normally
- Use `config/platform_config.py` for platform-aware code

### Linux/Windows Development
- Full support including bitsandbytes for quantization
- GPU acceleration with CUDA
- 4-bit and 8-bit model quantization available

### Docker Development

For full feature support including bitsandbytes quantization:

```bash
# Development with hot reload
docker-compose --profile dev up

# Production deployment
docker-compose --profile prod up

# GPU-enabled deployment (requires nvidia-docker)
docker-compose --profile gpu up
```

## Performance Metrics

The system tracks comprehensive performance metrics:

- **Average Reward**: Overall model performance score
- **Task Completion Rate**: Percentage of successful predictions
- **Reward Standard Deviation**: Consistency of model performance
- **Training Statistics**: Loss curves, convergence metrics
- **Comparison Reports**: Detailed model-vs-model analysis

## Quantization Strategy

### Development Environment
- Quantization features are automatically disabled on macOS
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

## Contributing Guidelines

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Ensure all tests pass** with `uv run pytest`
4. **Follow code style** guidelines (pre-commit hooks will help)
5. **Update documentation** as needed
6. **Submit a pull request** with a clear description of changes

## Getting Help

- Check existing issues and discussions
- Review the documentation in the `docs/` directory
- Run tests to verify your setup
- Use platform-appropriate development practices

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) document for details.
