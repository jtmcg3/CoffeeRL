# Multi-stage build for CoffeeRL-Lite with bitsandbytes support
FROM python:3.11.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (including bitsandbytes for Linux)
RUN uv sync --frozen

# Copy source code
COPY . .

# Development stage
FROM base as development
RUN uv sync --frozen --group dev
CMD ["uv", "run", "python", "-m", "pytest"]

# Production stage
FROM base as production
# Install only production dependencies
RUN uv sync --frozen --no-dev

# Add bitsandbytes for quantization support on Linux
RUN uv add bitsandbytes

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port for Gradio
EXPOSE 7860

# Default command
CMD ["uv", "run", "python", "main.py"]
