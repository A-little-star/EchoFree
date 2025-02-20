#!/bin/bash
NUM_NODES=1
NUM_TRAINERS=2
MASTER_PORT=23150

cd "$(dirname "${BASH_SOURCE[0]}")/../steps"

# single node multi-gpu training
CUDA_VISIBLE_DEVICES='0,1'  torchrun \
 --nnodes=${NUM_NODES} \
 --nproc_per_node=${NUM_TRAINERS} \
 --master_port=${MASTER_PORT} \
 train.py \
 -conf ../configs/train_echofree.yml
