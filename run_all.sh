#!/bin/bash

# ==============================================================================
# Script: run_all.sh
# Description: Full pipeline from training to visualization.
# Usage: ./run_all.sh <dataset> <model> <optimizer> <lr> <wd> <epochs> <patience> <device>
# ==============================================================================

# Asignación de variables
DATASET=$1
MODEL=$2
OPTIMIZER=$3
LR=$4
WD=$5
EPOCHS=$6
PATIENCE=$7
DEVICE=$8

echo "Running training ..."
# Asegúrate de que no haya espacios después de cada \
python3 ./src/run_training.py \
    --dataset_name "$DATASET" \
    --network_name "$MODEL" \
    --optimizer "$OPTIMIZER" \
    --base_lr "$LR" \
    --weight_decay "$WD" \
    --num_epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --device "$DEVICE"

echo "Generating figure ..."
python3 ./src/generate_curves.py \
    --dataset_name "$DATASET" \
    --network_name "$MODEL" \
    --optimizer_name "$OPTIMIZER"

echo "¡Resultados listos!"