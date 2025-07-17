.PHONY: install lint format check test test-preprocessing clean help setup setup-cuda setup-vortex setup-evo2

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  lint        - Run Ruff linter"
	@echo "  format      - Format code with Ruff"
	@echo "  check       - Run linting and formatting checks"
	@echo "  test        - Run tests"
	@echo "  test-preprocessing - Test preprocessing module"
	@echo "  clean       - Clean up cache files"
	@echo "  setup-dev   - Setup development environment"

# Install dependencies
install:
	poetry install

# Run Ruff linter
lint:
	poetry run ruff check .

# Format code with Ruff
format:
	poetry run ruff format .
	poetry run ruff check --fix .

# Run all checks
check: lint format

# Run tests
test:
	poetry run python -m pytest

# Test preprocessing module
test-preprocessing:
	poetry run python tests/test_preprocessing.py

# Clean up cache files
clean:
	poetry run ruff clean
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

# Setup development environment
setup-dev: install
	poetry run pre-commit install

# Verify installation
verify:
	@echo "🔍 Verifying installation..."
	poetry run python -c "import torch; import helical; print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ Helical: OK'); print(f'✅ CUDA: {torch.cuda.is_available()}')"
	@echo "🔍 Checking evo2..."
	poetry run python -c "import evo2; print(f'✅ Evo2: OK')" 2>/dev/null || echo "⚠️  Evo2 not available" 