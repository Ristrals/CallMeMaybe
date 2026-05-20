PC = python3

SET_UVCACHE =	UV_CACHE_DIR=/tmp/$(USER)/.cache/uv \
				UV_PROJECT_ENVIRONMENT=/tmp/$(USER)/.venv \
				HF_HOME=/tmp/$(USER)/.cache/huggingface \
				UV_IGNORE_ENV=1

run:
	$(SET_UVCACHE) uv run main.py

install:
	uv venv

sync:
	$(SET_UVCACHE) uv sync

uninstall:
	rm -rf .venv

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +

fclean: clean
	UV_CACHE_DIR=/tmp/$(USER)/.cache/uv uv cache clean
	rm -rf /tmp/$(USER)/.cache/huggingface
	rm -rf /tmp/$(USER)/.venv