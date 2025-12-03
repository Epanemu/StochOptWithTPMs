# StochOpt Testing

This directory contains the test suite for the StochOpt experiments.

## Running Tests

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Files
```bash
# Unit tests
pytest tests/test_newsvendor.py -v

# Integration tests
pytest tests/test_methods.py -v

# End-to-end tests
pytest tests/test_runner.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

View HTML coverage report at `htmlcov/index.html`.

### Run Tests in Parallel
```bash
pytest tests/ -n auto
```

## Test Structure

- `conftest.py`: Shared fixtures and test configuration
- `test_newsvendor.py`: Unit tests for newsvendor problem
- `test_methods.py`: Integration tests for optimization methods
- `test_runner.py`: End-to-end tests for experiment runner

## CI/CD

Tests run automatically on GitHub Actions for:
- Python 3.9, 3.10, 3.11
- All pushes and pull requests
- Coverage reports uploaded to Codecov

See `.github/workflows/test.yml` for CI configuration.
