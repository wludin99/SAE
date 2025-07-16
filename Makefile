.PHONY: install lint format check test clean help setup setup-cuda setup-vortex

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  setup       - Complete environment setup (CUDA + dependencies + vortex)"
	@echo "  setup-cuda  - Install CUDA toolkit with conda"
	@echo "  setup-vortex- Setup vortex repository"
	@echo "  lint        - Run Ruff linter"
	@echo "  format      - Format code with Ruff"
	@echo "  check       - Run linting and formatting checks"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean up cache files"
	@echo "  setup-dev   - Setup development environment"

# Install dependencies
install:
	poetry install

# Complete environment setup
setup:
	@echo "🚀 Setting up complete SAE environment..."
	@chmod +x setup_env.sh
	./setup_env.sh

# Setup with Python script (alternative)
setup-python:
	@echo "🚀 Setting up SAE environment with Python script..."
	python setup_env.py

# Install CUDA toolkit
setup-cuda:
	@echo "📦 Installing CUDA toolkit..."
	conda install cuda-toolkit=12.4 -c nvidia -y
	@echo "✅ CUDA toolkit installed"

# Setup vortex repository
setup-vortex:
	@echo "🔧 Setting up vortex..."
	@if [ ! -d "../vortex" ]; then \
		echo "📥 Cloning vortex repository to parent directory..."; \
		cd .. && git clone https://github.com/Zymrael/vortex.git; \
	fi
	cd ../vortex && git checkout f243e8e
	@echo "📦 Using torch version: 2.6.0"
	@if [[ "$OSTYPE" == "darwin"* ]]; then \
		sed -i '' "s/torch==2.5.1/torch==2.6.0/g" pyproject.toml; \
	else \
		sed -i "s/torch==2.5.1/torch==2.6.0/g" pyproject.toml; \
	fi
	cd ../vortex && make setup-full
	@echo "✅ Vortex setup complete in ../vortex"

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