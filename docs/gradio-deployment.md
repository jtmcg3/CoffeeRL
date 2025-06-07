# Gradio Web Interface Deployment Guide

This guide covers deploying the CoffeeRL-Lite Gradio web interface across different platforms.

## Quick Start (Local Development)

1. **Install dependencies:**
   ```bash
   pip install -r requirements-gradio.txt
   ```

2. **Run the application:**
   ```bash
   uv run python run_gradio.py
   ```

3. **Access the interface:**
   Open your browser to `http://127.0.0.1:7860`

## Platform-Specific Deployment

### macOS (Local Development)

The app automatically detects Apple Silicon and Intel Macs and optimizes accordingly:

- **Apple Silicon (M1/M2/M3):** Uses MPS acceleration with float16 precision
- **Intel Macs:** Falls back to CPU with memory optimization

```bash
# Clone and setup
git clone <your-repo>
cd CoffeRL
uv sync
uv run python run_gradio.py
```

### Linux with GPU

For CUDA-enabled systems:

```bash
# Ensure CUDA is available
nvidia-smi

# Install dependencies
pip install -r requirements-gradio.txt

# Run with GPU acceleration
python app.py
```

### Docker Deployment

1. **Build the container:**
   ```bash
   docker build -f Dockerfile.gradio -t coffeerl-gradio .
   ```

2. **Run the container:**
   ```bash
   # CPU-only
   docker run -p 7860:7860 coffeerl-gradio

   # With GPU support
   docker run --gpus all -p 7860:7860 coffeerl-gradio
   ```

3. **Access the interface:**
   Open `http://localhost:7860`

### Docker Compose (Recommended)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  coffeerl-gradio:
    build:
      context: .
      dockerfile: Dockerfile.gradio
    ports:
      - "7860:7860"
    environment:
      - MODEL_PATH=./models/coffee-qwen2-qlora
      - DEPLOYMENT_ENV=production
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    # Uncomment for GPU support
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

Run with:
```bash
docker-compose up -d
```

## Cloud Deployment

### Hugging Face Spaces

1. **Create a new Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Gradio" as the SDK

2. **Upload files:**
   ```bash
   # Clone your space
   git clone https://huggingface.co/spaces/yourusername/coffeerl-lite
   cd coffeerl-lite

   # Copy necessary files
   cp ../app.py .
   cp ../requirements-gradio.txt requirements.txt
   cp -r ../config .
   cp -r ../models/coffee-qwen2-qlora ./model

   # Create README.md for Space configuration
   cat > README.md << EOF
   ---
   title: CoffeeRL-Lite V60 Assistant
   emoji: ☕
   colorFrom: brown
   colorTo: green
   sdk: gradio
   sdk_version: 4.8.0
   app_file: app.py
   pinned: false
   ---

   # CoffeeRL-Lite: V60 Pour-Over Assistant

   A practical coffee brewing AI assistant focused on V60 pour-over optimization.
   EOF

   # Commit and push
   git add .
   git commit -m "Initial deployment"
   git push
   ```

### AWS/GCP/Azure

Use the Docker container for cloud deployment:

1. **Build and push to container registry:**
   ```bash
   # AWS ECR example
   aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-west-2.amazonaws.com
   docker build -f Dockerfile.gradio -t coffeerl-gradio .
   docker tag coffeerl-gradio:latest <account>.dkr.ecr.us-west-2.amazonaws.com/coffeerl-gradio:latest
   docker push <account>.dkr.ecr.us-west-2.amazonaws.com/coffeerl-gradio:latest
   ```

2. **Deploy to container service:**
   - **AWS:** Use ECS or EKS
   - **GCP:** Use Cloud Run or GKE
   - **Azure:** Use Container Instances or AKS

## Performance Optimization

### Hardware Requirements

- **Minimum (CPU only):** 4GB RAM, 2 CPU cores
- **Recommended (GPU):** 8GB VRAM, 16GB RAM
- **Storage:** ~2GB for model files

### Memory Optimization

The app includes several optimizations:

1. **Lazy loading:** Model loads only when first prediction is made
2. **Platform detection:** Automatically selects optimal settings
3. **Memory management:** Uses appropriate precision based on hardware

### Scaling Considerations

For production deployments:

1. **Load balancing:** Use multiple container instances
2. **Caching:** Consider caching frequent predictions
3. **Monitoring:** Monitor memory usage and response times
4. **Auto-scaling:** Configure based on CPU/memory usage

## Troubleshooting

### Common Issues

1. **Model loading errors:**
   ```bash
   # Check model path
   ls -la models/coffee-qwen2-qlora/

   # Verify dependencies
   pip list | grep -E "(torch|transformers|peft)"
   ```

2. **CUDA out of memory:**
   - Reduce batch size in platform settings
   - Use CPU fallback: `CUDA_VISIBLE_DEVICES="" python app.py`

3. **Port conflicts:**
   ```bash
   # Use different port
   export GRADIO_SERVER_PORT=7861
   python app.py
   ```

### Logs and Debugging

Enable debug mode:
```bash
export GRADIO_DEBUG=1
python app.py
```

Check container logs:
```bash
docker logs <container-id>
```

## Security Considerations

For production deployments:

1. **Network security:** Use HTTPS and proper firewall rules
2. **Authentication:** Consider adding authentication for sensitive deployments
3. **Rate limiting:** Implement rate limiting to prevent abuse
4. **Input validation:** The app includes basic input validation

## Monitoring

Monitor these metrics:

- **Response time:** Model inference latency
- **Memory usage:** Peak and average memory consumption
- **Error rate:** Failed predictions
- **Concurrent users:** Active sessions

Example monitoring with Docker:
```bash
# Monitor resource usage
docker stats <container-id>

# Check health
curl http://localhost:7860/health
```
