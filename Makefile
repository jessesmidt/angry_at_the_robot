VENV = .venv
PYTHON := uv run python -m student

.PHONY: install run run-default debug clean lint lint-strict help test

.DEFAULT_GOAL := help

venv:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Virtual environment created"

install: venv
	@echo "Installing RAG against the machine dependencies..."
	uv sync --all-groups
	@echo "Dependencies installed!"

index:
	@echo "Chunking repo and creating index..."
	$(PYTHON) index

run:
	@echo "Running CLI..."
	$(PYTHON)

debug:
	@echo "Running in debug mode..."
	uv run python -m pdb -m student index
	uv run python -m pdb -m student search "Is this debugger working?"

clean:
	@echo "Cleaning temporary files and caches..."
	@if [ -d "$(VENV)" ]; then rm -rf $(VENV) && echo "Removed $(VENV)/"; fi
	rm -rf dist/ build/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	@echo "Cleanup complete!"

lint:
	@echo "Running flake8..."
	uv run flake8
	@echo "Running mypy..."
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@echo "Running flake8..."
	uv run flake8
	@echo "Running mypy (strict)..."
	uv run mypy . --strict

help:
	@echo "RAG against the macine - Available Make targets:"
	@echo ""
	@echo "  make install       - Install dependencies using uv"
	@echo "  make index         - chunk the vLLM repo, build index"
	@echo "  make run      	 	- run with standard prompt"
	@echo "  make debug         - Run in debug mode"
	@echo "  make clean         - Remove temporary files and caches"
	@echo "  make lint          - Run flake8 and mypy"
	@echo "  make lint-strict   - Run strict linting"
	@echo "  make help          - Show this help message"
	@echo ""