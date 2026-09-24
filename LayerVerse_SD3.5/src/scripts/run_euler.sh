#!/usr/bin/env bash
# Euler + LayerVerse on PIE-Bench with SD3.5-medium. Extra arguments are passed through, e.g. --devices 0 1 2 3.
cd "$(dirname "$0")/.."
python edit_run_benchmark.py --sampler euler --num_steps 28 --guidance 7.5 "$@"
