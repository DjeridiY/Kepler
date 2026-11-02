.PHONY: check
check:
	uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q -p no:cacheprovider
