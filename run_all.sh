#!/bin/bash
DATASET=$1
MODEL=$2
OPTIMIZER=$3
LR=$4
WD=$5
EPOCHS=$6
PATIENCE=$7

echo "Running training ..."
python3 ./src/run_training.py \
    --dataset_name "$DATASET"\
    --network_name "$MODEL"\
    --optimizer "$OPTIMIZER"\
    --base_lr "$LR"\
    --weight_decay "$WD"\
    --num_epochs "$EPOCHS"\
    --patience "$PATIENCE"

echo "Generating figure ..."
python3 ./src/generate_curves.py \
    --dataset_name "$DATASET"\
    --network_name "$MODEL"\
    --optimizer_name "$OPTIMIZER"
echo "¡Resultados listos! :)"