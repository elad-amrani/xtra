#!/usr/bin/env bash
PROJ_NAME="train-""$2"
EXPERIMENT_PATH="./mmlm_saved/${PROJ_NAME}"

jbsub \
-q nonstandard \
-mem 700G \
-cores 1x112+8 \
-require a100 \
-proj ${PROJ_NAME} \
blaunch.sh python -u ./src/train.py \
--config "$1" \
--save_path ${EXPERIMENT_PATH} \
--use_lsf_ccc \
--clearml_task_name "$2"
