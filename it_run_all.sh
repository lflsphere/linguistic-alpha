#!/bin/bash
# IT Sector Analysis — run everything on the VM
# Usage: bash it_run_all.sh
set -e

echo "=== IT Sector Analysis Pipeline ==="
echo "$(date)"

cd ~/linguistic-alpha
git pull origin it-sector-analysis

# Step 1: Extract IT sector data (needs raw CRSP data in ~/earnings_sp500)
echo ""
echo "=== Step 1: Extract IT sector datasets ==="
python3 -u it_sector_extract.py --folder ~/earnings_sp500

# Step 2: Run Models A & B on both options
echo ""
echo "=== Step 2: Run Models A & B ==="
python3 -u it_run_models.py --folder ~/earnings_it

echo ""
echo "=== Done ==="
echo "Results in ~/earnings_it/"
ls -la ~/earnings_it/*.txt ~/earnings_it/*.json ~/earnings_it/*.parquet 2>/dev/null
