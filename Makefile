.PHONY: install, lint-format, lint-style, test-py, test-types, test, lint, all, mnist
.PHONY: mnist-logs

install:
	poetry install

lint-format:
	poetry run black tests

lint-style:
	poetry run pycodestyle tests --show-source

test-py:
	poetry run python -W ignore::DeprecationWarning -m pytest -v -s

test-types:
	poetry run mypy tests

lint: lint-format lint-style

test: test-py test-types

all: lint test

mnist:
	poetry run python src/mnist.py

mnist-logs:
	poetry run tensorboard --logdir data/mnist-model/logs/

mnist-cnn:
	poetry run python src/mnist-cnn.py

mnist-cnn-logs:
	poetry run tensorboard --logdir data/mnist-cnn/logs/

sentiment:
	poetry run python src/sentiment.py

sentiment-logs:
	poetry run tensorboard --logdir data/sentiment/logs/
