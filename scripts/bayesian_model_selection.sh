#!/bin/bash
#SBATCH --array=0-2%3
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=preempt
#SBATCH --time=168:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Define an array of commands
experiments=(
    'python ../src/diag.py --num_epochs=1000 --experiment_path="/cluster/home/eharve06/overfitting-ELBO/experiments/sin_dataset/diag_epochs=1000_N=20_num_samples=10_seed=2.pth" --lrs 0.1 --N=20 --num_samples=10 --ranks 25 40 63 100 158 251 398 631 1000 1585 2512 3981 6310 10000 --seed=2'
    'python ../src/rank1.py --num_epochs=1000 --experiment_path="/cluster/home/eharve06/overfitting-ELBO/experiments/sin_dataset/rank1_epochs=1000_N=20_num_samples=10_seed=2.pth" --lrs 0.1 --N=20 --num_samples=10 --ranks 25 40 63 100 158 251 398 631 1000 1585 2512 3981 6310 10000 --seed=2'
    'python ../src/fullrank.py --num_epochs=1000 --experiment_path="/cluster/home/eharve06/overfitting-ELBO/experiments/sin_dataset/fullrank_epochs=1000_N=20_num_samples=10_seed=2.pth" --lrs 0.1 --N=20 --num_samples=10 --ranks 25 40 63 100 158 251 398 631 1000 1585 2512 3981 6310 10000 --seed=2'
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate