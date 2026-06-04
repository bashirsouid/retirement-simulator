#!/bin/bash

# Monte Carlo Portfolio Simulator - Setup and Run Script
# This script creates a virtual environment, installs dependencies, and runs the simulation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
PYTHON_SCRIPT="monte_carlo_portfolio.py"
CONFIG_FILE="${SCRIPT_DIR}/config.json"
CONFIG_EXAMPLE_FILE="${SCRIPT_DIR}/config.example.json"

echo "=== Monte Carlo Portfolio Simulator Setup ==="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

echo "Found Python: $(python3 --version)"
echo ""

# Copy config.example.json to config.json if config.json doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    if [ -f "$CONFIG_EXAMPLE_FILE" ]; then
        echo "Creating config.json from config.example.json..."
        cp "$CONFIG_EXAMPLE_FILE" "$CONFIG_FILE"
        echo "Config file created."
    else
        echo "Error: Neither config.json nor config.example.json found in ${SCRIPT_DIR}"
        exit 1
    fi
else
    echo "Using existing config.json"
fi
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi
echo ""

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

# Install/upgrade pip using the venv's pip binary directly (avoids ensurepip permission issues)
if [ ! -f "$VENV_PIP" ]; then
    echo "pip not found in venv; attempting bootstrap via get-pip.py..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV_PYTHON" || {
        echo "Error: Could not install pip. Please run: $VENV_PYTHON -m pip install numpy pandas matplotlib"
        exit 1
    }
fi

echo "Upgrading pip..."
"$VENV_PIP" install --upgrade pip --quiet

echo "Installing required packages (numpy, pandas, matplotlib)..."
"$VENV_PIP" install numpy pandas matplotlib --quiet

echo ""
echo "=== Dependencies handling complete ==="
echo ""

# Check if Python script exists
if [ ! -f "${SCRIPT_DIR}/${PYTHON_SCRIPT}" ]; then
    echo "Error: ${PYTHON_SCRIPT} not found in ${SCRIPT_DIR}"
    echo "Please ensure the Python script is in the same directory as this bash script."
    exit 1
fi

# Run the simulation
echo "=== Running Monte Carlo Simulation ==="
echo ""
"$VENV_PYTHON" "${SCRIPT_DIR}/${PYTHON_SCRIPT}"

echo ""
echo "=== Simulation Complete ==="
echo "Output saved to: ${SCRIPT_DIR}/portfolio_simulation.csv"
echo "Chart saved to:  ${SCRIPT_DIR}/portfolio_simulation.png"
echo ""

echo "To run again, simply execute: ./$(basename "${BASH_SOURCE[0]}")"
