# CoffeeRL-Lite

A lightweight reinforcement learning framework for coffee optimization using transformer models and PEFT (Parameter-Efficient Fine-Tuning).

## Project Structure

```
CoffeRL/
├── data/           # Datasets and data processing scripts
├── models/         # Saved model checkpoints and configurations
├── src/            # Source code for the CoffeeRL-Lite implementation
├── notebooks/      # Jupyter notebooks for exploration and analysis
├── config/         # Configuration files for training and deployment
├── scripts/        # Utility scripts and automation
├── .venv/          # Virtual environment (managed by UV)
├── pyproject.toml  # Project configuration and dependencies
├── uv.lock         # Dependency lock file
└── README.md       # This file
```

## Setup

This project uses [UV](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites

- Python 3.11.11
- UV package manager

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd CoffeRL
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

## Dependencies

- **transformers**: Hugging Face transformers library
- **peft**: Parameter-Efficient Fine-Tuning
- **datasets**: Hugging Face datasets library
- **accelerate**: Distributed training support
- **gradio**: Web interface for model interaction
- **pandas**: Data manipulation and analysis
- **torch**: PyTorch deep learning framework

*Note: bitsandbytes is not available on macOS and has been excluded from this setup.*

## Development

Run the main application:
```bash
uv run python main.py
```

## License

[Add your license information here]
