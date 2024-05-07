#!/bin/bash
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -o calm_%j.out
#SBATCH --export=NONE
##SBATCH --gres=gpu:1
module load calm
cd $MYSCRATCH
echo "start $(date)"
# test that the training works with a very small training set
srun $(which python) -m calm.training --name=training-run --num-steps=100 --num-layers=3 \
        --embed-dim=120 --batch-size=46 \
        --warmup-steps=10 --max-positions=50 --ffn-embed-dim=48 \
        --no-progress-bar --num-data-workers=1 \
        $HOME/CaLM/training_data.fasta

echo "done $(date)"

