PC = python3

SET_UVCACHE =	UV_CACHE_DIR=/tmp/$(USER)/.cache/uv \
				UV_PROJECT_ENVIRONMENT=/tmp/$(USER)/.venv \
				HF_HOME=/tmp/$(USER)/.cache/huggingface \

run:
	$(SET_UVCACHE) uv run python -m src data/functions_definition.json data/function_calling_tests.json

book:
	$(SET_UVCACHE) uv run book.py

install:
	$(SET_UVCACHE) uv venv
	ln -sfn /tmp/$(USER)/.venv .venv

sync:
	$(SET_UVCACHE) uv sync

# Add debug command

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +

fclean: clean
	UV_CACHE_DIR=/tmp/$(USER)/.cache/uv uv cache clean
	rm -rf /tmp/$(USER)/.cache/huggingface
	rm -rf /tmp/$(USER)/.venv
	rm -rf .venv
	

lint:
	mypy . --strict
	flake8
	