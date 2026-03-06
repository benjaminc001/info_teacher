#!/usr/bin/venv bash
set -e

for dataset in reduced_housing unfavorable_housing tanh_easy sin_easy linear_easy
do
    bash run_all.sh "$dataset" "transformer" "adamw" 1e-3 1e-3 100 10 "cuda"
    done
echo "All experiments completed!"
done
    
    