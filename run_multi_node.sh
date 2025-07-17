#!/usr/bin/env bash
# Usage: ./run_multi_node.sh --nnodes <num_nodes> --node_rank <rank> --master_addr <ip> --master_port <port> --parallel_config <config.yaml> [other args]
export USE_MIX_MOE=1
torchrun --nproc_per_node=8 "$@" main.py
