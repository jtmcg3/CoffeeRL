# QLoRA Setup Guide for CoffeeRL-Lite

This guide covers setting up QLoRA (Quantized Low-Rank Adaptation) for fine-tuning Qwen2 models with platform awareness and automatic model selection.

## Overview

CoffeeRL-Lite supports platform-aware QLoRA training with automatic selection between:
- **Qwen2-0.5B**: For local development (especially macOS with Apple Silicon)
- **Qwen2-1.5B**: For cloud deployment and Linux environments with GPU support

## Platform Support

### macOS (Apple Silicon)
- **Model**: Qwen2-0.5B (smaller, more memory efficient)
- **Quantization**: Disabled (bitsandbytes not supported)
- **Precision**: bfloat16 (native Apple Silicon support)
- **Device**: CPU (MPS has limitations with some operations)

### Linux/Cloud (CUDA)
- **Model**: Qwen2-1.5B (larger, better performance)
- **Quantization**: 4-bit quantization with bitsandbytes
- **Precision**: float16
- **Device**: GPU with automatic device mapping

### Docker
- **Model**: Qwen2-1.5B (full quantization support)
- **Quantization**: 4-bit quantization enabled
- **Environment**: Linux container with bitsandbytes

## Quick Start

### 1. Install Dependencies

```bash
# Install with UV (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Check Platform Configuration

```bash
# Check your platform capabilities
uv run python config/platform_config.py
```

This will show:
- System information
- GPU availability
- Quantization support
- Optimal model selection
- Training parameters

### 3. Local Training (macOS/CPU)

```bash
# Train with automatic platform detection
uv run python src/train_qlora.py

# Or force specific model size
QWEN_MODEL_SIZE=0.5B uv run python src/train_qlora.py
```

### 4. Docker Training (Linux/GPU)

```bash
# CPU training
docker-compose --profile qlora-cpu up

# GPU training (requires nvidia-docker)
docker-compose --profile qlora-gpu up
```

### 5. Cloud Training (Google Colab, etc.)

```bash
# The system will automatically detect cloud environment
# and use Qwen2-1.5B with full quantization
python src/train_qlora.py
```

## Configuration Details

### Automatic Model Selection

The system automatically selects the optimal model based on:

1. **Environment Variable Override**:
   ```bash
   export QWEN_MODEL_SIZE=1.5B  # Force specific model
   ```

2. **Cloud Environment Detection**:
   - Google Colab (`COLAB_GPU` environment variable)
   - AWS (`AWS_EXECUTION_ENV` environment variable)
   - Docker environment (`/.dockerenv` file exists)
   - High memory systems (>12GB RAM)

3. **Default Fallback**:
   - Cloud environments → Qwen2-1.5B
   - Local environments → Qwen2-0.5B

### QLoRA Configuration

The LoRA configuration is optimized for Qwen2 architecture:

```python
LoraConfig(
    r=16,                    # Low rank for efficiency
    lora_alpha=32,           # Scaling factor
    target_modules=[         # Qwen2-specific modules
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training Parameters

Parameters are automatically adjusted based on platform:

| Platform | Batch Size | Gradient Accumulation | Precision | Quantization |
|----------|------------|----------------------|-----------|--------------|
| macOS    | 2          | 8                    | bfloat16  | Disabled     |
| Linux/GPU| 4          | 4                    | float16   | 4-bit        |
| Cloud    | 4          | 4                    | float16   | 4-bit        |

## Memory Requirements

### Qwen2-0.5B (Local Development)
- **Without Quantization**: ~2-3GB RAM
- **Training**: ~4-6GB RAM
- **Recommended**: 8GB+ system RAM

### Qwen2-1.5B (Cloud/GPU)
- **With 4-bit Quantization**: ~3-4GB VRAM
- **Training**: ~6-8GB VRAM
- **Recommended**: 12GB+ VRAM (T4, V100, A10G)

## Troubleshooting

### Common Issues

1. **bitsandbytes Import Error on macOS**:
   ```
   Warning: bitsandbytes is not supported on this platform (macOS).
   Quantization features will be disabled.
   ```
   This is expected. The system will automatically use full precision.

2. **CUDA Out of Memory**:
   - Reduce batch size: `export BATCH_SIZE=1`
   - Use gradient checkpointing
   - Switch to Qwen2-0.5B: `export QWEN_MODEL_SIZE=0.5B`

3. **MPS Backend Issues on Apple Silicon**:
   The system automatically uses CPU for training to avoid MPS limitations.

### Performance Optimization

1. **For Apple Silicon (M1/M2/M3)**:
   ```bash
   # Use optimized PyTorch for Apple Silicon
   pip install torch torchvision torchaudio
   ```

2. **For CUDA Systems**:
   ```bash
   # Ensure CUDA-compatible PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **For Docker**:
   ```bash
   # Use GPU-optimized image
   docker-compose --profile qlora-gpu up
   ```

## Advanced Configuration

### Custom Model Selection

```python
from config.platform_config import get_optimal_qwen_model

# Override model selection
import os
os.environ["QWEN_MODEL_SIZE"] = "1.5B"
model_name = get_optimal_qwen_model()
```

### Custom Training Parameters

```python
from qlora_config import setup_training_arguments

# Create custom training arguments
training_args = setup_training_arguments(
    output_dir="./custom_output"
)

# Modify specific parameters
training_args.num_train_epochs = 5
training_args.learning_rate = 1e-4
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QWEN_MODEL_SIZE` | Force model size (0.5B or 1.5B) | Auto-detect |
| `BATCH_SIZE` | Override batch size | Platform-dependent |
| `CUDA_VISIBLE_DEVICES` | GPU selection | All available |

## Integration with Existing Workflows

### With Jupyter Notebooks

```python
# In notebook cell
from src.qlora_config import setup_training_environment

model, tokenizer, training_args = setup_training_environment()
```

### With Existing Training Scripts

```python
# Import platform-aware components
from config.platform_config import get_optimal_qwen_model
from qlora_config import load_model_and_tokenizer

# Use in existing code
model_name = get_optimal_qwen_model()
model, tokenizer = load_model_and_tokenizer()
```

## Next Steps

1. **Prepare Training Data**: Ensure your datasets are in the correct format
2. **Run Training**: Use the provided training script or integrate with your workflow
3. **Monitor Training**: Use TensorBoard to track progress
4. **Evaluate Results**: Test the fine-tuned model on validation data

For more information, see:
- [Training Guide](TRAINING.md)
- [Data Preparation](DATA_PREPARATION.md)
- [Deployment Guide](DEPLOYMENT.md)
