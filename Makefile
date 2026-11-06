.PHONY: help install test bench clean build

help:
	@echo "CUDA Attention Kernel Commands:"
	@echo "  make install    - Setup venv and install dependencies"
	@echo "  make test       - Run all unit tests with pytest"
	@echo "  make bench      - Run latency and memory scaling benchmark"
	@echo "  make clean      - Clean build artifacts and cache"

install:
	python3 -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt

test:
	PYTHONPATH=. ./.venv/bin/pytest tests/ -v

bench:
	PYTHONPATH=. ./.venv/bin/python benchmarks/benchmark_attention.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ __pycache__ flash_attn/__pycache__

