#!/usr/bin/env bash
CONFIG_PATH="$1"
PROJ_NAME="$2"
TASK_NAME="$3"
EXPERIMENT_PATH="./saved/train-""$3"

jbsub \
-q nonstandard \
-mem 100G \
-cores 16+1 \
-require a100_40gb \
-proj ${PROJ_NAME} \
-name ${TASK_NAME} \
python -u ./src/train.py \
--config ${CONFIG_PATH} \
--save_path ${EXPERIMENT_PATH} \
--clearml_task_name ${TASK_NAME}
