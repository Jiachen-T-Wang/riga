#!/bin/bash

#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-freq=high
#SBATCH -t 0-20:00

for i in {1..500}
do
        python main.py $i
done