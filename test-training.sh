#!/bin/bash
# test that the training works with a very small training set
name=${1-training-run}
time python -m calm.training --name=$name --num-steps=100 --num-layers=3 \
        --embed-dim=120 --batch-size=46 \
        --warmup-steps=10 --max-positions=50 --ffn-embed-dim=48 \
        training_data.fasta training_data2.fasta
