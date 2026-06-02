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
echo ""

# Using Hydra's multirun with glob syntax

python main.py \
  --config-name=multirun_config \
  --multirun \
  hydra/launcher=basic \
  seed=0,1,2,3,4 \
  problem=newsvendor \
  problem/newsvendor/dim=dim_2,dim_4,dim_8,dim_16,dim_32 \
  problem/newsvendor/distribution=norm,exp,uniform \
  method=robust,sample_average,nn_bce,quantile_nn,tpm_spn,tpm_tree_cnet

echo ""
echo "All experiments completed!"
echo "Results are saved in the multirun directory and logged to MLflow"
