.PHONY: install, lint-format, lint-style, test-py, test-types, test, lint, all, mnist
.PHONY: mnist-profile

install:
	poetry install

lint-format:
	poetry run black tests

lint-style:
	poetry run pycodestyle tests --show-source

test-py:
	poetry run pytest

test-types:
	poetry run mypy tests

lint: lint-format lint-style

test: test-py test-types

all: lint test

mnist:
	python src/mnist.py

mnist-profile:
	poetry run pprofile \
		--statistic .01 \
		--format callgrind \
		--out mnist.callgrind \
		src/mnist.py
