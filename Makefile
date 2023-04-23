package:
	python -m pip install --upgrade pip
	pip install build
	python -m build

doc:
	pip install -r docs/requirements.txt
	cd docs && sphinx-apidoc -o ./source -f ../src/detectors -d 4 && make clean && make html

test:
	pytest -vv tests/models.py tests/methods.py tests/docstrings.py -s --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered

format:
	black --config pyproject.toml .
	isort --settings pyproject.toml .

lint:
	black . --check
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --statistics

help:
	@echo 'package      - build the package'
	@echo 'doc		      - build the documentation'
	@echo 'test         - run unit tests and generate coverage report'
	@echo 'format       - run code formatters'
	@echo 'lint         - run linters'