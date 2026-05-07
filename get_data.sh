#!/bin/bash

# ==============================================================================
# Data Setup Script for Thesis Project
# Description: Downloads CCPP and SARCOS datasets and runs preprocessing.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting data setup..."

# 1. Create directory structure
echo "📂 Creating directories..."
mkdir -p raw_data/ccpp
mkdir -p raw_data/sarcos
mkdir -p array_data

# 2. Download CCPP Dataset (UCI Repository)
echo "📥 Downloading CCPP dataset..."
# The CCPP dataset is usually a ZIP file
curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip" -o raw_data/ccpp/CCPP.zip
unzip -o raw_data/ccpp/CCPP.zip -d raw_data/ccpp/
rm raw_data/ccpp/CCPP.zip

# 3. Download SARCOS Dataset (Train and Test)
echo "📥 Downloading SARCOS dataset..."
curl -L "http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat" -o raw_data/sarcos/sarcos_inv.mat
curl -L "http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat" -o raw_data/sarcos/sarcos_inv_test.mat

# 4. Generate Synthetic Data
echo "🧪 Generating synthetic data..."
python3 src/synthetic_data/data_generation.py

# 5. Run Preprocessing
echo "⚙️ Running data preprocessing..."
python3 src/preprocess_data.py

echo "✅ Data setup complete! All files are in raw_data and array_data."