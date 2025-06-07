# CoffeeRL-Lite Deployment Guide

This guide covers various deployment options for training and running CoffeeRL-Lite models across different environments.

## Quick Start

### Local Development (macOS/Linux)

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest -v

# Train with development subset
export COFFEE_DEV_MODE=true
uv run python src/train_qlora.py
```

### Docker Development

```bash
# Run tests
docker-compose --profile test up --build

# Development environment
docker-compose --profile dev up --build

# Training with GPU
docker-compose --profile train up --build
```

## Environment Variables

### Training Configuration

- `COFFEE_DEV_MODE`: Enable development mode with data subset (default: false)
- `COFFEE_MAX_TRAIN_SAMPLES`: Max training samples in dev mode (default: 100)
- `COFFEE_MAX_EVAL_SAMPLES`: Max evaluation samples in dev mode (default: 20)
- `COFFEE_OUTPUT_DIR`: Output directory for trained models (default: ./models/coffee-qwen2-qlora)

### Platform Override

- `PLATFORM_OVERRIDE`: Force platform detection (LINUX_GPU, MACOS, LINUX_CPU)

### Caching

- `TRANSFORMERS_CACHE`: Transformers model cache directory
- `HF_HOME`: Hugging Face cache directory

## Deployment Options

### 1. Local Development (macOS)

**Recommended for**: Initial development, testing, small experiments

```bash
# Setup
uv sync

# Quick test with subset
export COFFEE_DEV_MODE=true
export COFFEE_MAX_TRAIN_SAMPLES=10
uv run python src/train_qlora.py

# Full local training (uses Qwen2-0.5B)
unset COFFEE_DEV_MODE
uv run python src/train_qlora.py
```

**Features**:
- Automatic Qwen2-0.5B model selection
- MPS acceleration on Apple Silicon
- Reduced batch sizes for memory efficiency
- Fast iteration cycles

### 2. Docker Local Training

**Recommended for**: Consistent environment, Linux simulation

```bash
# Build and run training
docker-compose --profile train up --build

# With custom settings
COFFEE_OUTPUT_DIR=./custom_output docker-compose --profile train up
```

**Features**:
- Consistent Linux environment
- Automatic dependency management
- Volume mounting for data persistence
- GPU support when available

### 3. Google Colab

**Recommended for**: Free GPU access, experimentation

```python
# In Colab notebook
!git clone https://github.com/your-repo/coffeerl.git
%cd coffeerl

# Install dependencies
!pip install uv
!uv sync

# Set environment for Colab
import os
os.environ['PLATFORM_OVERRIDE'] = 'LINUX_GPU'
os.environ['COFFEE_OUTPUT_DIR'] = '/content/drive/MyDrive/coffee_models'

# Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

# Run training
!uv run python src/train_qlora.py
```

**Setup Steps**:
1. Upload your dataset to Google Drive
2. Create a new Colab notebook
3. Enable GPU runtime (Runtime → Change runtime type → GPU)
4. Run the setup code above

### 4. AWS SageMaker

**Recommended for**: Production training, large datasets

```python
# sagemaker_training.py
import sagemaker
from sagemaker.pytorch import PyTorch

# Define training job
estimator = PyTorch(
    entry_point='train_qlora.py',
    source_dir='src',
    role=sagemaker.get_execution_role(),
    instance_type='ml.g4dn.xlarge',  # GPU instance
    instance_count=1,
    framework_version='2.0.1',
    py_version='py310',
    environment={
        'PLATFORM_OVERRIDE': 'LINUX_GPU',
        'COFFEE_OUTPUT_DIR': '/opt/ml/model'
    }
)

# Start training
estimator.fit({
    'training': 's3://your-bucket/coffee_training_dataset',
    'validation': 's3://your-bucket/coffee_validation_dataset'
})
```

### 5. Azure Machine Learning

**Recommended for**: Enterprise environments, Azure ecosystem

```python
# azure_training.py
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment

# Create environment
env = Environment(
    image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
    conda_file="environment.yml"
)

# Define training job
job = command(
    code="./src",
    command="python train_qlora.py",
    environment=env,
    compute="gpu-cluster",
    environment_variables={
        "PLATFORM_OVERRIDE": "LINUX_GPU",
        "COFFEE_OUTPUT_DIR": "/tmp/outputs"
    }
)

# Submit job
ml_client.jobs.create_or_update(job)
```

### 6. Kubernetes Deployment

**Recommended for**: Production inference, scalability

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coffeerl-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: coffeerl
  template:
    metadata:
      labels:
        app: coffeerl
    spec:
      containers:
      - name: coffeerl
        image: coffeerl:production
        ports:
        - containerPort: 7860
        env:
        - name: GRADIO_SERVER_NAME
          value: "0.0.0.0"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## Performance Optimization

### Memory Management

```bash
# For limited memory environments
export COFFEE_DEV_MODE=true
export COFFEE_MAX_TRAIN_SAMPLES=50

# For high-memory environments
unset COFFEE_DEV_MODE
export COFFEE_OUTPUT_DIR=/path/to/fast/storage
```

### GPU Optimization

```bash
# Single GPU training
export CUDA_VISIBLE_DEVICES=0

# Multi-GPU training (if supported)
export CUDA_VISIBLE_DEVICES=0,1
```

### Storage Optimization

```bash
# Use fast local storage for cache
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/hf_cache

# Use network storage for models
export COFFEE_OUTPUT_DIR=/shared/models
```

## Monitoring and Logging

### TensorBoard

```bash
# Start TensorBoard
uv run tensorboard --logdir=./logs

# In Docker
docker run -p 6006:6006 -v $(pwd)/logs:/logs tensorflow/tensorflow:latest tensorboard --logdir=/logs --host=0.0.0.0
```

### Training Metrics

Training automatically saves:
- `training_metadata.json`: Training configuration and timing
- `training_metrics.json`: Loss curves and metrics
- `eval_results.json`: Evaluation results

### Log Files

```bash
# View training logs
tail -f logs/training.log

# In Docker
docker-compose logs -f train
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   export COFFEE_DEV_MODE=true
   export COFFEE_MAX_TRAIN_SAMPLES=10
   ```

2. **CUDA Not Available**
   ```bash
   export PLATFORM_OVERRIDE=LINUX_CPU
   ```

3. **Model Download Issues**
   ```bash
   export HF_HOME=/path/with/space
   ```

4. **Permission Issues in Docker**
   ```bash
   docker-compose run --user $(id -u):$(id -g) train
   ```

### Performance Tuning

1. **Batch Size Optimization**
   - Start with small batch sizes (1-2)
   - Increase gradually until memory limit
   - Use gradient accumulation for effective larger batches

2. **Sequence Length**
   - Reduce max_seq_length for memory savings
   - Monitor truncation in logs

3. **Model Size Selection**
   - Use Qwen2-0.5B for development/testing
   - Use Qwen2-1.5B for production training

## Security Considerations

### Docker Security

```bash
# Run as non-root user
docker-compose run --user 1000:1000 train

# Limit resources
docker run --memory=4g --cpus=2 coffeerl:train
```

### Cloud Security

- Use IAM roles instead of API keys
- Enable encryption at rest for model storage
- Use private networks for training clusters
- Regularly update base images

## Cost Optimization

### Cloud Training Costs

1. **Use Spot Instances**: 60-90% cost savings
2. **Right-size Instances**: Start small, scale up
3. **Preemptible Training**: Save checkpoints frequently
4. **Data Transfer**: Keep data in same region

### Storage Costs

1. **Model Compression**: Use quantized models
2. **Cache Management**: Clean up old cache files
3. **Lifecycle Policies**: Auto-delete old training runs

## Next Steps

1. **Model Serving**: Deploy trained models with FastAPI/Gradio
2. **Monitoring**: Set up model performance monitoring
3. **CI/CD**: Automate training pipelines
4. **Scaling**: Implement distributed training for larger models
