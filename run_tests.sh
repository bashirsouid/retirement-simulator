#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python3}"

echo "=== Monte Carlo Simulator Test Suite ==="
echo "Using Python: $($PYTHON_BIN --version)"
echo

echo "[1/2] Running unit tests..."
$PYTHON_BIN -m unittest -v test_monte_carlo_portfolio.py

echo
echo "[2/2] Running CLI smoke test (single default run)..."
# This ensures the refactored CLI path works with config.toml
if ! $PYTHON_BIN monte_carlo_portfolio.py > /tmp/mc_sim_smoke.log 2>&1; then
    echo "CLI smoke test FAILED. See /tmp/mc_sim_smoke.log for details."
    exit 1
fi

echo
echo "All tests passed."