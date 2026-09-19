.PHONY: fmt
fmt:
	uvx ruff check --fix --line-length 5000 --extend-select I jev_v1.py jev_v2.py plain.py bench.py
	uvx ruff format --line-length 5000 jev_v1.py jev_v2.py plain.py bench.py

.PHONY: lint
lint:
	uv run --with pyright pyright jev_v1.py jev_v2.py plain.py

.PHONY: bench
bench:
	uv run bench.py

.PHONY: precommit
precommit:
	$(MAKE) fmt
	$(MAKE) lint
