#!/usr/bin/env bash
# Euler + LayerVerse on PIE-Bench with FLUX.1-dev. Extra arguments are passed through, e.g. --devices 0 1 2 3.
cd "$(dirname "$0")/.."
python edit_run_benchmark.py --sampler euler --num_steps 15 --guidance 2 "$@"
