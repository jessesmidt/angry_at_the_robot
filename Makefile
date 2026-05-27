VENV = .venv
PYTHON := uv run python -m student

DOCS_UNANSWERED := data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json
CODE_UNANSWERED := data/datasets_public/public/UnansweredQuestions/dataset_code_public.json
DOCS_ANSWERED := data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json
CODE_ANSWERED := data/datasets_public/public/AnsweredQuestions/dataset_code_public.json
MOULINETTE := data/moulinette/moulinette_pkg/moulinette-ubuntu
OUTPUT_DIR := data/output/search_results

.PHONY: install run debug clean lint lint-strict help
.PHONY: index search search_docs search_code evaluate evaluate_docs evaluate_code

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
	@echo "RAG against the machine - jsmidt"
	@$(PYTHON) index

search:
	@echo "Usage: make search Q='your question here'"
	@$(PYTHON) search "$(Q)"

search_docs:
	@echo "Searching docs dataset..."
	@$(PYTHON) search_dataset \
		--dataset_path $(DOCS_UNANSWERED) \
		--k 5 \
		--save_dir $(OUTPUT_DIR)

search_code:
	@echo "Searching code dataset..."
	@$(PYTHON) search_dataset \
		--dataset_path $(CODE_UNANSWERED) \
		--k 5 \
		--save_dir $(OUTPUT_DIR)

search_all: search_docs search_code

evaluate_docs:
	@echo "Evaluating docs recall..."
	@chmod +x $(MOULINETTE)
	@$(MOULINETTE) evaluate_student_search_results \
		$(OUTPUT_DIR)/dataset_docs_public.json \
		$(DOCS_ANSWERED) \
		--k 5 \
		--max_context_length 2000 \
		--threshold 0.80

evaluate_code:
	@echo "Evaluating code recall..."
	@chmod +x $(MOULINETTE)
	@$(MOULINETTE) evaluate_student_search_results \
		$(OUTPUT_DIR)/dataset_code_public.json \
		$(CODE_ANSWERED) \
		--k 5 \
		--max_context_length 2000 \
		--threshold 0.50

evaluate: 
	evaluate_docs evaluate_code

answer_docs: 
		$(PYTHON) answer_dataset \
        --student_search_results_path data/output/search_results/dataset_docs_public.json \
        --save_directory data/output/search_results_and_answer

answer_code: 
		$(PYTHON) answer_dataset \
        --student_search_results_path data/output/search_results/dataset_code_public.json \
        --save_directory data/output/search_results_and_answer

answer_all: answer_docs answer_code


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
	@echo "RAG against the machine - Available Make targets:"
	@echo ""
	@echo "  make install         - Install dependencies using uv"
	@echo "  make index           - Chunk the vLLM repo and build index"
	@echo "  make search Q='...'  - Search a single query"
	@echo "  make search_docs     - Search docs dataset"
	@echo "  make search_code     - Search code dataset"
	@echo "  make search_all      - Search both datasets"
	@echo "  make evaluate_docs   - Evaluate docs recall@5"
	@echo "  make evaluate_code   - Evaluate code recall@5"
	@echo "  make evaluate        - Evaluate both datasets"
	@echo "  make run             - Run CLI"
	@echo "  make debug           - Run in debug mode"
	@echo "  make clean           - Remove temporary files and caches"
	@echo "  make lint            - Run flake8 and mypy"
	@echo "  make lint-strict     - Run strict linting"
	@echo "  make help            - Show this help message"
	@echo ""