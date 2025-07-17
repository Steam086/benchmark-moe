#!/usr/bin/env bash
# Usage: ./run_single_node.sh --parallel_config <config.yaml> [other args]
export USE_MIX_MOE=1
torchrun --nproc_per_node=8 main.py
