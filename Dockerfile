# Multi-stage build for CoffeeRL-Lite with QLoRA and bitsandbytes support
FROM python:3.11.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

# Install system dependencies for QLoRA training
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy source code
COPY . .

# Development stage
FROM base as development
RUN uv sync --frozen --group dev
CMD ["uv", "run", "python", "-m", "pytest"]

# QLoRA training stage
FROM base as qlora
# Install bitsandbytes for quantization support on Linux
RUN uv add bitsandbytes

# Copy source code
COPY . .

# Create models directory
RUN mkdir -p models

# Default command for training
CMD ["uv", "run", "python", "src/train_qlora.py"]

# Production stage
FROM base as production
# Install only production dependencies
RUN uv sync --frozen --no-dev

# Add bitsandbytes for quantization support on Linux
RUN uv add bitsandbytes

# Copy source code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port for Gradio
EXPOSE 7860

# Default command
CMD ["uv", "run", "python", "main.py"]
