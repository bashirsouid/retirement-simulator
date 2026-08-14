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
echo "[2/2] Running CLI smoke test..."
if [[ ! -f config.toml ]]; then
    echo "No config.toml; copying config.example.toml"
    cp config.example.toml config.toml
fi

# Keep the smoke run small so CI does not simulate 10k + stress paths.
SMOKE_DIR="$(mktemp -d)"
cp monte_carlo_portfolio.py "$SMOKE_DIR/"
python3 - <<PY
from pathlib import Path
text = Path("config.toml").read_text()
text = text.replace("simulations = 10000", "simulations = 20")
text = text.replace("stress_first_n_years = 10", "stress_first_n_years = 0")
Path("$SMOKE_DIR/config.toml").write_text(text)
PY

if ! (cd "$SMOKE_DIR" && $PYTHON_BIN monte_carlo_portfolio.py > /tmp/mc_sim_smoke.log 2>&1); then
    echo "CLI smoke test FAILED. See /tmp/mc_sim_smoke.log for details."
    rm -rf "$SMOKE_DIR"
    exit 1
fi
rm -rf "$SMOKE_DIR"

echo
echo "All tests passed."
