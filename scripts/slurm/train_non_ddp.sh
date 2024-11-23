#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --signal=SIGUSR1@90
#SBATCH --open-mode=append

EXPERIMENT_PATH="./saved/"${SLURM_JOB_NAME}
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/log.txt --error=${EXPERIMENT_PATH}/err.txt --label python -u ./src/train.py \
--config "$1" \
--save_path ${EXPERIMENT_PATH} \
--use_slurm
