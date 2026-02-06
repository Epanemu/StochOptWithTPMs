#!/bin/bash
# LOCAL Multirun script for testing SPN x Robust x SAA on Newsvendor problems
# This script runs experiments LOCALLY (not on SLURM cluster)
# For SLURM cluster execution, use: run_multirun_slurm.sh
#
# This script runs experiments with:
# - Methods: tpm (SPN), robust, sample_average (SAA)
# - Sample sizes: 100, 1000, 10000 (opt and train equal)
# - Products: 1, 5, 10, 50, 100
# - Different distribution parameters
#
# Total experiments: 3 methods × 3 sample sizes × 5 problem variants = 45 runs

echo "Starting multirun experiments..."
echo "This will run 45 experiments (3 methods × 3 sample sizes × 5 problem variants)"
echo ""

# Using Hydra's multirun with glob syntax
# We need to run three separate multiruns to ensure samples.train equals samples.opt

# Run 1: opt=train=100
echo "Running experiments with 100 samples..."
python main.py \
  --config-name=multirun_config \
  --multirun \
  hydra/launcher=basic \
  method=tpm,robust,sample_average \
  samples.opt=100 \
  samples.train=100 \
  problem=newsvendor,newsvendor_5prod_low,newsvendor_10prod_med,newsvendor_50prod_high,newsvendor_100prod_veryhigh

# Run 2: opt=train=1000
echo "Running experiments with 1000 samples..."
python main.py \
  --config-name=multirun_config \
  --multirun \
  hydra/launcher=basic \
  method=tpm,robust,sample_average \
  samples.opt=1000 \
  samples.train=1000 \
  problem=newsvendor,newsvendor_5prod_low,newsvendor_10prod_med,newsvendor_50prod_high,newsvendor_100prod_veryhigh

# Run 3: opt=train=10000
echo "Running experiments with 10000 samples..."
python main.py \
  --config-name=multirun_config \
  --multirun \
  hydra/launcher=basic \
  method=tpm,robust,sample_average \
  samples.opt=10000 \
  samples.train=10000 \
  problem=newsvendor,newsvendor_5prod_low,newsvendor_10prod_med,newsvendor_50prod_high,newsvendor_100prod_veryhigh

echo ""
echo "All experiments completed!"
echo "Results are saved in the multirun directory and logged to MLflow"
