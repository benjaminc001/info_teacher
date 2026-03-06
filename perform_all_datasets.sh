#!/bin/bash
MODEL=$1
DEVICE=$2
for dataset in housing unfavorable_housing_full ccpp noisy_ccpp sarcos unfavorable_sarcos
do
    echo "Running experiments for dataset: $dataset"
    bash run_all.sh "$dataset" "$MODEL" "adamw" 1e-3 1e-3 100 10 "$DEVICE"
done
echo "All experiments completed!"