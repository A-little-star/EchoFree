#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")/../steps"

CUDA_VISIBLE_DEVICES='1'  torchrun \
 --nnodes=1 \
 --nproc_per_node=1 \
 --master_port=23111 \
 inference_stream.py -conf ../configs/test_echofree_stream.yml
 # inference.py -conf ../configs/test_echofree.yml