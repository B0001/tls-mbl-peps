#!/usr/bin/env bash
set -e; cd "$(dirname "$0")"
python3 golden_3x3.py
python3 sdrg_3site.py
python3 bench_kernel.py timing 2 3 4
echo "tier-1 prototype verification complete"
