#!/usr/bin/env bash
#SBATCH --nodes=16
#SBATCH --gres=gpu:32g:6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=0
#SBATCH --signal=SIGUSR1@90
#SBATCH --open-mode=append
#SBATCH --partition=el8,el8-rpi,dcs-2024

EXPERIMENT_PATH="./saved/"${SLURM_JOB_NAME}
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/log.txt --error=${EXPERIMENT_PATH}/err.txt --label python -u ./src/train.py \
--config "$1" \
--save_path ${EXPERIMENT_PATH} \
--use_slurm
