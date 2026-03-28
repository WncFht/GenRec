.PHONY: build commit license quality style test clean-junk

check_dirs := scripts src tests

RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")
BUILD := $(shell command -v uv >/dev/null 2>&1 && echo "uv build" || echo "python -m build")
TOOL := $(shell command -v uv >/dev/null 2>&1 && echo "uvx" || echo "")

build:
	$(BUILD)

commit:
	$(TOOL) pre-commit install
	$(TOOL) pre-commit run --all-files

license:
	$(RUN) python3 tests/check_license.py $(check_dirs)

quality:
	$(TOOL) ruff check $(check_dirs)
	$(TOOL) ruff format --check $(check_dirs)

style:
	$(TOOL) ruff check $(check_dirs) --fix
	$(TOOL) ruff format $(check_dirs)

test:
	WANDB_DISABLED=true $(RUN) pytest -vv --import-mode=importlib tests/

clean-junk:
	find . -name '.DS_Store' -delete
	find . -name '*.pid' -delete
	find . -name '*.py[co]' -delete
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	find log -type f -empty -delete
