#!/bin/bash
# SLURM Multirun script for testing SPN x Robust x SAA on Newsvendor problems
# This script submits each experiment as a separate SLURM job
#
# This script runs experiments with:
# - Methods: tpm_spn (SPN), tpm_cnet (CNet), tpm_tree_cnet (Tree trained as CNet), tpm_tree_greedy (Tree trained greedily), robust, sample_average (SAA)
# - Sample sizes: 100, 1000, 10000 (opt and train equal)
# - Products: 1, 5, 10, 50, 100
# - Different distribution parameters
#
# Total experiments: 6 methods × 3 sample sizes × 5 problem variants = 90 runs
# Each run will be submitted as a separate SLURM job

echo "=== Starting SLURM Multirun Experiments ==="
echo "This will submit 90 jobs to the SLURM cluster (6 methods × 3 sample sizes × 5 problem variants)"
echo ""

# Read MLflow server URI from connection file (if available)
CONNECTION_FILE="$HOME/mlflow/mlflow_server_connection.txt"
MLFLOW_URI=""

if [ -f "$CONNECTION_FILE" ]; then
    MLFLOW_URI=$(cat $CONNECTION_FILE)
    echo "Found MLflow server connection: $MLFLOW_URI"

    # Verify server is accessible
    if curl -s "$MLFLOW_URI/health" > /dev/null 2>&1; then
        echo "✓ MLflow server is responding"
        MLFLOW_ARG="mlflow.tracking_uri=\"$MLFLOW_URI\""
    else
        echo "⚠️  MLflow server not responding, aborting..."
        exit 1
    fi
else
    echo "No MLflow server connection file found"
    echo "To use a shared server, run: ./start_mlflow_server.sh"
    exit 1
fi
echo ""

# Ensure logs directory exists
mkdir -p logs

ml Python/3.12.3-GCCcore-13.3.0
ml Gurobi/13.0.0-GCCcore-13.3.0

# Activate virtual environment or set it up if needed
if [ -d ".venv" ]; then
    source .venv/bin/activate
    pip install -r requirements.txt
else
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

# Check if hydra-submitit-launcher is installed
if ! python -c "import hydra_plugins.hydra_submitit_launcher" 2>/dev/null; then
    echo "ERROR: hydra-submitit-launcher not found!"
    echo "Please install it with: pip install hydra-submitit-launcher"
    exit 1
fi

# Using Hydra's multirun with SLURM launcher
# Each job will be submitted separately to SLURM

# Run 1: opt=train=100
echo "→ Submitting experiments with 100 samples to SLURM..."
python main.py \
  --config-name=multirun_config \
  --multirun \
  $MLFLOW_ARG \
  method=tpm_spn,tpm_cnet,tpm_tree_cnet,tpm_tree_greedy,robust,sample_average \
  samples.opt=100 \
  samples.train=100 \
  problem=news/newsvendor,news/newsvendor_5prod_low,news/newsvendor_10prod_med,news/newsvendor_10prod_correlated #,news/newsvendor_50prod_high,news/newsvendor_100prod_veryhigh

# Run 2: opt=train=1000
echo "→ Submitting experiments with 1000 samples to SLURM..."
python main.py \
  --config-name=multirun_config \
  --multirun \
  $MLFLOW_ARG \
  method=tpm_spn,tpm_cnet,tpm_tree_cnet,tpm_tree_greedy,robust,sample_average \
  samples.opt=1000 \
  samples.train=1000 \
  problem=news/newsvendor,news/newsvendor_5prod_low,news/newsvendor_10prod_med,news/newsvendor_10prod_correlated #,news/newsvendor_50prod_high,news/newsvendor_100prod_veryhigh

# Run 3: opt=train=10000
echo "→ Submitting experiments with 10000 samples to SLURM..."
python main.py \
  --config-name=multirun_config \
  --multirun \
  $MLFLOW_ARG \
  method=tpm_spn,tpm_cnet,tpm_tree_cnet,tpm_tree_greedy,robust,sample_average \
  samples.opt=10000 \
  samples.train=10000 \
  problem=news/newsvendor,news/newsvendor_5prod_low,news/newsvendor_10prod_med,news/newsvendor_10prod_correlated #,news/newsvendor_50prod_high,news/newsvendor_100prod_veryhigh

echo ""
echo "=== All Jobs Submitted to SLURM! ==="
echo ""
echo "Monitor job status:"
echo "  squeue -u \$USER"
echo ""
if [ -n "$MLFLOW_URI" ]; then
    echo "View results in MLflow UI:"
    echo "  $MLFLOW_URI"
    echo ""
fi
echo "Results will be saved in the multirun directory and logged to MLflow"
echo ""
