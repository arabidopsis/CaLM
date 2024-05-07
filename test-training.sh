#!/bin/bash
# test that the training works with a very small training set
# currently takes ~ 4m30 on my laptop
name=training-run
if [ $# -gt 0 ]; then
        name=$1
        shift
fi
time python -m calm.training --name=$name --num-steps=100 --num-layers=3 \
        --embed-dim=120 --batch-size=46 \
        --warmup-steps=10 --max-positions=50 --ffn-embed-dim=48 \
        --ntasks-per-node=4 \
        "$@" \
        training_data.fasta training_data2.fasta
