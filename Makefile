.PHONY: help setup install clean test train format lint docs

help:
	@echo "=== Geothermal Piles PINN Project ==="
	@echo ""
	@echo "Available commands:"
	@echo "  make setup              - Setup project (venv + install)"
	@echo "  make install            - Install dependencies"
	@echo "  make clean              - Remove cache files and build artifacts"
	@echo "  make test               - Run all tests"
	@echo "  make test-cov           - Run tests with coverage"
	@echo "  make train              - Run training"
	@echo "  make evaluate           - Evaluate model"
	@echo "  make format             - Format code with Black"
	@echo "  make lint               - Lint code with Flake8 + Pylint"
	@echo "  make notebook           - Start Jupyter Lab"
	@echo "  make docs               - Generate documentation"
	@echo "  make check              - Run all checks (lint + test)"
	@echo ""

setup:
	@echo "Setting up project..."
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@echo "✓ Project setup complete!"
	@echo "Activate venv with: source venv/bin/activate"

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed!"

clean:
	@echo "Cleaning project..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf htmlcov/ .coverage
	@echo "✓ Cleaned!"

test:
	@echo "Running tests..."
	pytest -v

test-cov:
	@echo "Running tests with coverage..."
	pytest --cov=src --cov-report=html --cov-report=term-missing
	@echo "✓ Coverage report generated in htmlcov/"

train:
	@echo "Starting training..."
	python scripts/train.py

train-custom:
	@echo "Starting training with custom params..."
	python scripts/train.py --config configs/training_params.yaml

evaluate:
	@echo "Evaluating model..."
	python scripts/evaluate.py

format:
	@echo "Formatting code with Black..."
	black src/ scripts/ tests/
	@echo "✓ Code formatted!"

lint:
	@echo "Linting code..."
	flake8 src/ scripts/ tests/ --max-line-length=100
	pylint src/ --disable=all --enable=E,F
	@echo "✓ Linting complete!"

check: lint test
	@echo "✓ All checks passed!"

notebook:
	@echo "Starting Jupyter Lab..."
	jupyter lab

notebook-simple:
	@echo "Starting Jupyter Notebook..."
	jupyter notebook

docs:
	@echo "Building documentation..."
	cd docs && make html
	@echo "✓ Docs built in docs/_build/html/"

freeze:
	@echo "Updating requirements.txt..."
	pip freeze > requirements.txt
	@echo "✓ requirements.txt updated!"

status:
	@echo "=== Project Status ==="
	@echo "Git status:"
	@git status
	@echo ""
	@echo "Installed packages:"
	@pip list | head -20

.DEFAULT_GOAL := help
