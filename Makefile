.PHONY: install lint test cov all

install:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy --strict

test:
	uv run pytest

cov:
	uv run pytest --cov=fitsq --cov-report=term-missing --cov-fail-under=80

all: lint cov
