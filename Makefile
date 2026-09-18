.PHONY: precommit-hook
precommit-hook:
	@common_dir="$$(git rev-parse --git-common-dir 2>/dev/null)"; \
	hooks_path="$$(git config --get core.hooksPath 2>/dev/null)"; \
	if [ -n "$$common_dir" ] && [ -z "$$hooks_path" ]; then \
	mkdir -p "$$common_dir/hooks" && printf '#!/bin/sh\nmake precommit\n' > "$$common_dir/hooks/pre-push" && chmod +x "$$common_dir/hooks/pre-push"; \
	fi

.PHONY: fmt
fmt:
	uvx ruff check --fix --line-length 5000 --target-version py314 --extend-select I --ignore F403,F405,F821,E731,E402,B008 qwen27b_jev.py tests
	uvx ruff format --line-length 5000 --target-version py314 qwen27b_jev.py tests

.PHONY: lint
lint:
	uv run --with vulture vulture --min-confidence 80 qwen27b_jev.py tests
	uv run --with pyright pyright qwen27b_jev.py

.PHONY: tests
tests:
	uv run pytest -W ignore tests/

.PHONY: precommit
precommit:
	uv sync
	$(MAKE) precommit-hook
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) tests
