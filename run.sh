#!/bin/bash
set -euo pipefail

# CAGNNF Code Ocean reproducibility demo
# This script is intentionally lightweight and uses synthetic data so that
# reviewers can verify the computational workflow without restricted EEG data.

echo "Running CAGNNF reproducibility demo..."

# Code Ocean usually provides a /results folder. For local GitHub runs, create one.
mkdir -p /results 2>/dev/null || mkdir -p results

python scripts/reproduce_demo_results.py --steps 5 --output-dir /results || \
python scripts/reproduce_demo_results.py --steps 5 --output-dir results

echo "Finished. Demo outputs have been written to /results or ./results."
