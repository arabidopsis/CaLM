#!/bin/bash
# test that the training works with a very small training set
python training.py --name=training-run3 --num-steps=100 --num-layers=3 --embed-dim=120 --batch-size=46 \
        --warmup-steps=10 --max-positions=50 --ffn-embed-dim=48 \
        --fasta-file=training_data.fasta
