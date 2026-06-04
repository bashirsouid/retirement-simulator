#!/usr/bin/env bash
# Monte Carlo Portfolio Simulator — Runner

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/monte_carlo_portfolio.py"
CONFIG="${SCRIPT_DIR}/config.toml"
CONFIG_EXAMPLE="${SCRIPT_DIR}/config.example.toml"
VENV_DIR="${SCRIPT_DIR}/venv"
REQUIREMENTS="numpy pandas matplotlib"

echo "=== Monte Carlo Portfolio Simulator ==="
echo ""

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Please install Python 3.11+."
    exit 1
fi
echo "Found Python: $(python3 --version)"
echo ""

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: monte_carlo_portfolio.py not found in ${SCRIPT_DIR}"
    exit 1
fi

# ── Config setup ──────────────────────────────────────────────────────────────
if [ ! -f "$CONFIG" ]; then
    if [ -f "$CONFIG_EXAMPLE" ]; then
        echo "No config.toml found — copying from config.example.toml..."
        cp "$CONFIG_EXAMPLE" "$CONFIG"
        echo "Created config.toml. Edit it with your values before running."
        echo ""
    else
        echo "Error: No config.toml or config.example.toml found in ${SCRIPT_DIR}"
        exit 1
    fi
else
    echo "Using existing config.toml"
fi
echo ""

# ── Dependency setup ──────────────────────────────────────────────────────────
# Use existing venv if present, otherwise create one cleanly with python3 -m venv
# (no ensurepip bootstrapping — Python 3.11+ ships pip inside venv by default)

PYTHON="${VENV_DIR}/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
    echo ""
fi

# Check if all required packages are already installed
NEED_INSTALL=false
for pkg in $REQUIREMENTS; do
    if ! "$PYTHON" -c "import ${pkg}" &>/dev/null; then
        NEED_INSTALL=true
        break
    fi
done

if [ "$NEED_INSTALL" = true ]; then
    echo "Installing dependencies (numpy, pandas, matplotlib)..."
    "$VENV_DIR/bin/pip" install --quiet $REQUIREMENTS
    echo "Dependencies installed."
    echo ""
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== Running Monte Carlo Simulation ==="
echo ""
"$PYTHON" "$PYTHON_SCRIPT"

echo ""
echo "=== Done ==="
echo "To run again: ./$(basename "${BASH_SOURCE[0]}")"
