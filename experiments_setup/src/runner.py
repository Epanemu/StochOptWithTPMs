import logging
import sys
import os
import time
from typing import Any
import numpy as np
import mlflow
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.DataHandler import DataHandler
# Import TPM trainers
try:
    from src.tpms.spn_tpm import SpnTPM
    from src.tpms.cnet_tpm import CNetTPM
except ImportError:
    logging.warning("Could not import TPM modules. TPM training will fail.")

log = logging.getLogger(__name__)

def train_tpm(cfg: DictConfig, data: np.ndarray, data_handler: DataHandler) -> Any:
    """
    Train a TPM (SPN or CNet) on the provided data.
    """
    if cfg.method.tpm.type == "spn":
        log.info("Training SPN...")
        tpm = SpnTPM()
        tpm.train(
            data,
            data_handler,
            min_instances_slice=cfg.method.tpm.min_instances_slice,
            n_clusters=cfg.method.tpm.n_clusters
        )
        return tpm

    elif cfg.method.tpm.type == "cnet":
        log.info("Training CNet...")
        tpm = CNetTPM()
        tpm.train(
            data,
            data_handler,
            min_instances_slice=cfg.method.tpm.min_instances_slice
        )
        return tpm
    else:
        raise ValueError(f"Unknown TPM type: {cfg.method.tpm.type}")

def run_experiment(cfg: DictConfig) -> None:
    """
    Main experiment execution function.
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.log_params(dict(cfg))

        # 1. Instantiate Problem
        log.info(f"Instantiating problem: {cfg.problem._target_}")
        problem = instantiate(cfg.problem, cfg=cfg.problem)

        # 2. Generate Data
        log.info(f"Generating {cfg.samples.train} training samples...")
        train_samples = problem.generate_samples(n_samples=cfg.samples.train, seed=cfg.seed)

        # 3. Setup DataHandler and TPM
        if cfg.method.name == "tpm":
            log.info("Generating TPM training data...")
            tpm_data, feat_names = problem.generate_tpm_data(train_samples, seed=cfg.seed)

            categ_map = {"sat": [0, 1]}
            if cfg.method.tpm.discrete:
                raise NotImplementedError("Discretization/binning for continuous xi and x is not yet implemented.")

            data_handler = DataHandler(
                tpm_data,
                feature_names=feat_names,
                categ_map=categ_map
            )

            # Train TPM
            tpm = train_tpm(cfg, tpm_data, data_handler)
        else:
            tpm = None
            data_handler = None
            tpm_data = None

        # 4. Build and Solve Model
        log.info(f"Building model for method: {cfg.method.name}")

        # For robust/sample average, we need scenarios
        if cfg.method.name in ["robust", "sample_average"]:
            # Use a subset of training samples
            opt_samples = train_samples[:cfg.samples.opt]
        else:
            opt_samples = None

        build_start_time = time.time()
        model = problem.build_model(
            method=cfg.method.name,
            tpm=tpm,
            data_handler=data_handler,
            scenarios=opt_samples,
            risk_level=cfg.risk_level
        )
        build_duration = time.time() - build_start_time
        mlflow.log_metric("build_duration", build_duration)

        log.info("Solving model...")
        solve_start_time = time.time()
        result = problem.solve()
        solve_duration = time.time() - solve_start_time

        log.info(f"Result: {result}")
        mlflow.log_metric("solve_duration", solve_duration)
        mlflow.log_metric("total_duration", build_duration + solve_duration)

        if result["objective"] is not None:
            mlflow.log_metric("objective", result["objective"])

        # 5. Evaluation / Verification
        log.info("Evaluating solution...")

        # Get solution vector
        try:
            x_sol = problem.get_solution()
        except ValueError:
            log.warning("Could not retrieve solution (model might be infeasible).")
            return

        # Validation on new samples
        val_seed = cfg.seed + 1 # Different seed
        # TODO add to config
        n_val = cfg.samples.get("validation", cfg.samples.test) # Default if not in config
        validation_samples = problem.generate_samples(n_samples=n_val, seed=val_seed)

        val_satisfied = problem.check_satisfaction(x_sol, validation_samples)
        val_prob_satisfied = np.mean(val_satisfied)

        log.info(f"Validation Satisfaction Probability: {val_prob_satisfied}")
        mlflow.log_metric("val_prob_satisfied", val_prob_satisfied)
        mlflow.log_metric("val_violation_prob", 1 - val_prob_satisfied)

        # Training set violation (in-sample)
        train_satisfied = problem.check_satisfaction(x_sol, train_samples)
        train_prob_satisfied = np.mean(train_satisfied)
        mlflow.log_metric("train_prob_satisfied", train_prob_satisfied)
        mlflow.log_metric("train_violation_prob", 1 - train_prob_satisfied)

        # TODO log the solution vector to mlflow

