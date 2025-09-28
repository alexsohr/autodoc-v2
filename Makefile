.PHONY: help install dev-install lint format type-check test test-cov clean clean-cache dev-run run docker-build docker-run

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -e .

dev-install: ## Install development dependencies
	pip install -e ".[dev]"

lint: ## Run linting (flake8)
	flake8 src tests

format: ## Format code with black and isort
	black src tests
	isort src tests

format-check: ## Check code formatting without making changes
	black --check src tests
	isort --check-only src tests

type-check: ## Run type checking with mypy
	mypy src

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term

test-fast: ## Run tests excluding slow ones
	pytest -m "not slow"

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-cache: ## Clean cache files using Python script (cross-platform)
	python scripts/clean_cache.py

dev-run: clean-cache ## Clean cache and run the development server
	@echo "Starting AutoDoc v2 development server (with cache cleaning)..."
	python -m src.api.main

run: ## Run the development server (without cache cleaning)
	python -m src.api.main

run-prod: ## Run with gunicorn (production)
	gunicorn src.api.main:app --bind 0.0.0.0:8000 --workers 4 --worker-class uvicorn.workers.UvicornWorker

docker-build: ## Build Docker image
	docker build -t autodoc-v2:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env autodoc-v2:latest

check: format-check lint type-check ## Run all code quality checks

ci: check test ## Run all CI checks
