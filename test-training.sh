#!/bin/bash
# test that the training works with a very small training set
python training.py --num_steps=500 --num_layers=3 --embed_dim=120 --batch_size=46 \
        --warmup_steps=300 --max_positions=100 --ffn_embed_dim=48 \
        --data=training_data.fasta
