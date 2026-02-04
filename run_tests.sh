#!/bin/bash
# Quick test runner for StochOpt with virtual environment

VENV_DIR=".venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv $VENV_DIR
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "Running tests..."
python -m pytest tests/ -v --tb=short

echo ""
echo "================================"
echo "Other useful commands:"
echo "  Coverage report:    python -m pytest tests/ --cov=src --cov-report=term --cov-report=html"
echo "  Specific test:      python -m pytest tests/test_newsvendor.py -v"
echo "  Parallel tests:     python -m pytest tests/ -n auto"
echo "  Deactivate venv:    deactivate"
echo "================================"
