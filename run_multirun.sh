#!/bin/bash
# LOCAL Multirun script for testing SPN x Robust x SAA on Newsvendor problems
# This script runs experiments LOCALLY (not on SLURM cluster)
# For SLURM cluster execution, use: run_multirun_slurm.sh
#
# This script runs experiments with:
# - Methods: tpm_spn (SPN), tpm_cnet (CNets), tpm_tree_cnet (Tree trained as CNets), tpm_tree_greedy (Tree trained greedily), and 2 baseline methods: robust, sample_average (SAA)
# - Sample sizes: 100, 1000, 10000 (opt and train equal)
# - Products: 1, 2, 4, 5, 10, 50, 100
# - Different distribution parameters
#
# Total experiments: 6 methods × 3 sample sizes × 12 problem variants = 216 runs

echo "Starting multirun experiments..."
echo "This will run 216 experiments (6 methods × 3 sample sizes × 12 problem variants)"
echo ""

# Using Hydra's multirun with glob syntax
# We need to run three separate multiruns to ensure samples.train equals samples.opt

# Run 1: opt=train=100
echo "Running experiments with 100 samples..."
python main.py \
  --config-name=multirun_config \
  --multirun \
  hydra/launcher=basic \
  method=tpm_spn,tpm_cnet,tpm_tree_cnet,tpm_tree_greedy,robust,sample_average \
  samples.opt=100 \
  samples.train=100 \
  problem=news/1_norm,news/2_norm,news/2_exp,news/4_norm,news/4_exp,news/4_corr,news/5_norm,news/10_norm,news/10_exp,news/10_corr,news/50_norm,news/100_norm

# Run 2: opt=train=1000
echo "Running experiments with 1000 samples..."
python main.py \
  --config-name=multirun_config \
  --multirun \
  hydra/launcher=basic \
  method=tpm_spn,tpm_cnet,tpm_tree_cnet,tpm_tree_greedy,robust,sample_average \
  samples.opt=1000 \
  samples.train=1000 \
  problem=news/1_norm,news/2_norm,news/2_exp,news/4_norm,news/4_exp,news/4_corr,news/5_norm,news/10_norm,news/10_exp,news/10_corr,news/50_norm,news/100_norm

# Run 3: opt=train=10000
echo "Running experiments with 10000 samples..."
python main.py \
  --config-name=multirun_config \
  --multirun \
  hydra/launcher=basic \
  method=tpm_spn,tpm_cnet,tpm_tree_cnet,tpm_tree_greedy,robust,sample_average \
  samples.opt=10000 \
  samples.train=10000 \
  problem=news/1_norm,news/2_norm,news/2_exp,news/4_norm,news/4_exp,news/4_corr,news/5_norm,news/10_norm,news/10_exp,news/10_corr,news/50_norm,news/100_norm

echo ""
echo "All experiments completed!"
echo "Results are saved in the multirun directory and logged to MLflow"
