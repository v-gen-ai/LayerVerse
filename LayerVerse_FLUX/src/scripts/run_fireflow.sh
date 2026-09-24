#!/usr/bin/env bash
# FireFlow + LayerVerse on PIE-Bench with FLUX.1-dev. Extra arguments are passed through, e.g. --devices 0 1 2 3.
cd "$(dirname "$0")/.."
python edit_run_benchmark.py --sampler fireflow --num_steps 15 --guidance 2 "$@"
