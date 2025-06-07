# CoffeeRL Makefile
# Provides convenient targets for training, evaluation, and pipeline operations

.PHONY: help install test lint format clean train-dev train-full eval-quick eval-full pipeline-dev pipeline-full docker-build docker-test

# Default target
help:
	@echo "CoffeeRL Development Commands"
	@echo "============================="
	@echo ""
	@echo "Setup & Development:"
	@echo "  install          Install dependencies with uv"
	@echo "  test             Run all tests"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  clean            Clean up generated files"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  train-dev        Quick development training (Qwen2-0.5B, 100 samples)"
	@echo "  train-full       Full training (Qwen2-0.5B, all data)"
	@echo "  eval-quick       Quick evaluation (50 samples)"
	@echo "  eval-full        Full evaluation (all validation data)"
	@echo ""
	@echo "Automated Pipelines:"
	@echo "  pipeline-dev     Run complete dev pipeline (train + eval)"
	@echo "  pipeline-full    Run complete full pipeline (train + eval)"
	@echo "  hyperparameter-sweep  Run hyperparameter optimization"
	@echo ""
	@echo "Performance Analysis:"
	@echo "  analyze-auto     Auto-analyze all evaluation results"
	@echo "  analyze-compare  Compare multiple model results"
	@echo ""
	@echo "Docker Operations:"
	@echo "  docker-build     Build Docker development image"
	@echo "  docker-test      Run tests in Docker container"
	@echo "  docker-train     Run training in Docker container"
	@echo ""
	@echo "Data Operations:"
	@echo "  check-data       Verify training and validation datasets exist"
	@echo "  data-stats       Show dataset statistics"

# Setup and development targets
install:
	uv sync

test:
	uv run pytest tests/ -v

lint:
	uv run flake8 src/ tests/ scripts/
	uv run black --check src/ tests/ scripts/
	uv run isort --check-only src/ tests/ scripts/

format:
	uv run black src/ tests/ scripts/
	uv run isort src/ tests/ scripts/

clean:
	rm -rf experiments/
	rm -rf models/coffee-qwen2-qlora/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" -delete

# Training targets
train-dev:
	uv run python src/train_local.py \
		--model-name Qwen/Qwen2-0.5B \
		--dev-mode \
		--max-samples 100 \
		--output-dir models/coffee-qwen2-dev

train-full:
	uv run python src/train_local.py \
		--model-name Qwen/Qwen2-0.5B \
		--output-dir models/coffee-qwen2-full

train-1.5b:
	uv run python src/train_local.py \
		--model-name Qwen/Qwen2-1.5B \
		--output-dir models/coffee-qwen2-1.5b

# Evaluation targets
eval-quick:
	@if [ ! -d "models/coffee-qwen2-dev" ] && [ ! -d "models/coffee-qwen2-full" ]; then \
		echo "No trained model found. Run 'make train-dev' or 'make train-full' first."; \
		exit 1; \
	fi
	@MODEL_DIR=$$(ls -d models/coffee-qwen2-* | head -1); \
	uv run python src/evaluate_local.py \
		--model-path $$MODEL_DIR \
		--quick-eval \
		--max-samples 50 \
		--output-file evaluation_quick.json

eval-full:
	@if [ ! -d "models/coffee-qwen2-dev" ] && [ ! -d "models/coffee-qwen2-full" ]; then \
		echo "No trained model found. Run 'make train-dev' or 'make train-full' first."; \
		exit 1; \
	fi
	@MODEL_DIR=$$(ls -d models/coffee-qwen2-* | head -1); \
	uv run python src/evaluate_local.py \
		--model-path $$MODEL_DIR \
		--output-file evaluation_full.json \
		--save-predictions

# Pipeline targets
pipeline-dev:
	./scripts/run_experiment.sh \
		--training-mode dev \
		--evaluation-mode quick \
		--experiment-name dev-$(shell date +%Y%m%d-%H%M%S)

pipeline-full:
	./scripts/run_experiment.sh \
		--training-mode full \
		--evaluation-mode full \
		--experiment-name full-$(shell date +%Y%m%d-%H%M%S)

hyperparameter-sweep:
	./scripts/run_experiment.sh \
		--hyperparameter-sweep \
		--experiment-name sweep-$(shell date +%Y%m%d-%H%M%S)

# Docker targets
docker-build:
	docker build -t coffeerl:dev --target development .

docker-test:
	docker compose -f docker-compose.yml run --rm test

docker-train:
	docker compose -f docker-compose.yml run --rm qlora-training

# Data verification targets
check-data:
	@echo "Checking dataset availability..."
	@if [ -d "data/processed/coffee_training_dataset" ]; then \
		echo "✅ Training dataset found"; \
		echo "   Files: $$(find data/processed/coffee_training_dataset -name "*.json" | wc -l)"; \
	else \
		echo "❌ Training dataset not found"; \
	fi
	@if [ -d "data/processed/coffee_validation_dataset" ]; then \
		echo "✅ Validation dataset found"; \
		echo "   Files: $$(find data/processed/coffee_validation_dataset -name "*.json" | wc -l)"; \
	else \
		echo "❌ Validation dataset not found"; \
	fi

data-stats:
	@echo "Dataset Statistics"
	@echo "=================="
	@if [ -d "data/processed/coffee_training_dataset" ]; then \
		echo "Training dataset:"; \
		find data/processed/coffee_training_dataset -name "*.json" -exec wc -l {} + | tail -1 | awk '{print "  Total lines: " $$1}'; \
	fi
	@if [ -d "data/processed/coffee_validation_dataset" ]; then \
		echo "Validation dataset:"; \
		find data/processed/coffee_validation_dataset -name "*.json" -exec wc -l {} + | tail -1 | awk '{print "  Total lines: " $$1}'; \
	fi

# Advanced targets
benchmark:
	@echo "Running performance benchmark..."
	./scripts/run_experiment.sh \
		--training-mode dev \
		--evaluation-mode quick \
		--experiment-name benchmark-$(shell date +%Y%m%d-%H%M%S) \
		--save-predictions

compare-models:
	@echo "Comparing model sizes..."
	./scripts/run_experiment.sh \
		--training-mode dev \
		--evaluation-mode quick \
		--model-size 0.5B \
		--experiment-name compare-0.5b-$(shell date +%Y%m%d-%H%M%S)
	./scripts/run_experiment.sh \
		--training-mode dev \
		--evaluation-mode quick \
		--model-size 1.5B \
		--experiment-name compare-1.5b-$(shell date +%Y%m%d-%H%M%S)

# Development workflow targets
dev-setup: install check-data
	@echo "Development environment ready!"

dev-cycle: format lint test train-dev eval-quick
	@echo "Development cycle completed!"

ci-check: lint test
	@echo "CI checks passed!"

# Experiment management
list-experiments:
	@echo "Available experiments:"
	@if [ -d "experiments" ]; then \
		ls -la experiments/ | grep ^d | awk '{print "  " $$9}' | grep -v "^  \."; \
	else \
		echo "  No experiments found"; \
	fi

clean-experiments:
	@echo "Cleaning old experiments..."
	@if [ -d "experiments" ]; then \
		find experiments/ -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true; \
		echo "Removed experiments older than 7 days"; \
	fi

# Performance analysis targets
analyze-auto:
	./scripts/analyze_performance.sh --mode auto

analyze-compare:
	@echo "Comparing models from evaluation results..."
	@if [ ! -d "evaluation_results" ]; then \
		echo "No evaluation results found. Run evaluations first."; \
		exit 1; \
	fi
	./scripts/analyze_performance.sh --mode compare \
		--results-dir evaluation_results

analyze-custom:
	@echo "Usage: make analyze-custom RESULTS='file1.json,file2.json' NAMES='Model1,Model2'"
	@if [ -z "$(RESULTS)" ]; then \
		echo "Error: RESULTS parameter required"; \
		exit 1; \
	fi
	./scripts/analyze_performance.sh --mode compare \
		--results-files "$(RESULTS)" \
		$(if $(NAMES),--model-names "$(NAMES)")
