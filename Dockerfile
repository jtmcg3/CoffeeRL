# Multi-stage build for CoffeeRL-Lite with QLoRA and bitsandbytes support
FROM python:3.11.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    PLATFORM_OVERRIDE=LINUX_GPU

# Install system dependencies for QLoRA training
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    cmake \
    ninja-build \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install UV for faster dependency management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files and README first for better layer caching
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
RUN uv sync --frozen

# Development stage with additional tools
FROM base as development
RUN uv sync --frozen --group dev

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p models data/processed logs

# Default command for development
CMD ["uv", "run", "python", "-m", "pytest", "-v"]

# QLoRA training stage optimized for GPU training
FROM base as qlora

# Install bitsandbytes for quantization support on Linux
RUN uv add bitsandbytes

# Copy source code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p models data/processed logs output && \
    chmod -R 755 models data logs output

# Set environment variables for training
ENV COFFEE_OUTPUT_DIR=/app/output \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    HF_HOME=/app/.cache/huggingface

# Create cache directories
RUN mkdir -p /app/.cache/transformers /app/.cache/huggingface

# Default command for training
CMD ["uv", "run", "python", "src/train_qlora.py"]

# Production inference stage
FROM base as production

# Install only production dependencies
RUN uv sync --frozen --no-dev

# Add bitsandbytes for quantization support on Linux
RUN uv add bitsandbytes

# Copy source code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 app && \
    mkdir -p models data logs && \
    chown -R app:app /app

USER app

# Expose port for Gradio interface
EXPOSE 7860

# Default command for inference
CMD ["uv", "run", "python", "main.py"]

# Testing stage for CI/CD
FROM development as testing

# Copy test data if available
COPY tests/ tests/

# Set environment for testing
ENV COFFEE_DEV_MODE=true \
    COFFEE_MAX_TRAIN_SAMPLES=10 \
    COFFEE_MAX_EVAL_SAMPLES=5

# Run tests by default
CMD ["uv", "run", "python", "-m", "pytest", "-v", "--tb=short"]
