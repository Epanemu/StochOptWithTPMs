import logging
import os
import time

import mlflow
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from stochopt.data.DataHandler import DataHandler
from stochopt.data.Types import DataLike

# Import TPM trainers
try:
    from stochopt.tpms.cnet_tpm import CNetTPM
    from stochopt.tpms.spn_tpm import SpnTPM
    from stochopt.tpms.tpm import TPM
    from stochopt.tpms.TreeTPM.tree_tpm import TreeTPM
except ImportError:
    logging.warning("Could not import TPM modules. TPM training will fail.")

log = logging.getLogger(__name__)


def train_tpm(cfg: DictConfig, data: DataLike, data_handler: DataHandler) -> TPM:
    """
    Train a TPM (SPN, CNet, or Tree) on the provided data.
    """
    tpm_cfg = cfg.method
    tpm_name = tpm_cfg.name

    encoded_data = data_handler.encode(data)

    if tpm_name == "spn":
        log.info("Training SPN...")
        tpm: TPM = SpnTPM(data_handler)
        tpm.train(
            encoded_data,
            min_instances_slice=tpm_cfg.min_instances_slice,
            n_clusters=tpm_cfg.n_clusters,
        )
        return tpm

    elif tpm_name == "cnet":
        log.info("Training CNet...")
        tpm = CNetTPM(data_handler)
        # Extract CNet specific params
        tpm.train(
            encoded_data,
            min_instances_slice=tpm_cfg.min_instances_slice,
            max_depth=tpm_cfg.max_depth,
            discretization_method=tpm_cfg.discretization_method,
            n_bins=tpm_cfg.n_bins,
        )
        return tpm

    elif tpm_name == "tree":
        log.info(f"Training TreeTPM (Learner: {tpm_cfg.learner})...")
        tpm = TreeTPM(data_handler)
        if tpm_cfg.learner == "greedy":
            tpm.train_greedy_top_down(
                encoded_data,
                min_samples=tpm_cfg.min_samples,
                max_depth=tpm_cfg.max_depth,
                val_ratio=tpm_cfg.val_ratio,
                alpha=tpm_cfg.alpha,
                max_branches=tpm_cfg.max_branches,
            )
        elif tpm_cfg.learner == "cnet":
            tpm.train(
                encoded_data,
                min_instances_slice=tpm_cfg.min_instances_slice,
                max_depth=tpm_cfg.max_depth,
                n_bins=tpm_cfg.n_bins,
            )
        else:
            raise ValueError(f"Unknown Tree learner: {tpm_cfg.learner}")
        return tpm

    else:
        raise ValueError(f"Unknown TPM name: {tpm_name}")


def run_experiment(cfg: DictConfig) -> None:
    """
    Main experiment execution function.
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        try:
            # Set meaningful run name and tags
            run_name = f"{cfg.method.name}_n{cfg.problem.n_products}_opt{cfg.samples.opt}_train{cfg.samples.train}"
            mlflow.set_tag("mlflow.runName", run_name)
            mlflow.set_tag("method", cfg.method.name)
            mlflow.set_tag(
                "n_products", str(cfg.problem.n_products)
            )  # String for categorical grouping
            mlflow.set_tag("samples.opt", str(cfg.samples.opt))
            mlflow.set_tag("samples.train", str(cfg.samples.train))
            mlflow.set_tag("samples.train_decisions", str(cfg.samples.train_decisions))
            mlflow.set_tag("problem_type", cfg.problem.get("name", "unknown"))
            mlflow.set_tag("status", "RUNNING")

            slurm_id = os.environ.get("SLURM_JOB_ID")
            if slurm_id:
                mlflow.set_tag("slurm_job_id", slurm_id)

            # Add date tag for grouping batches of experiments
            import datetime

            today = datetime.datetime.now().strftime("%Y-%m-%d")
            mlflow.set_tag("benchmark_date", today)

            # Log all parameters
            mlflow.log_params(dict(cfg))

            # 1. Instantiate Problem
            log.info(f"Instantiating problem: {cfg.problem._target_}")
            problem = instantiate(cfg.problem, solver=cfg.solver, _convert_="all")

            # 2. Generate Data
            log.info(f"Generating {cfg.samples.train} training samples...")
            train_samples = problem.generate_samples(
                n_samples=cfg.samples.train,
                seed=cfg.seed,
            )

            # 3. Setup DataHandler and TPM
            if cfg.method.get("type") == "tpm":
                log.info("Generating TPM training data...")
                tpm_data, feat_names = problem.generate_tpm_data(
                    n_decisions=cfg.samples.train_decisions,
                    train_samples=train_samples,
                    seed=cfg.seed,
                )

                categ_map: dict[int | str, list[int | str]] = {"sat": [0, 1]}

                data_handler = DataHandler(
                    tpm_data, y=None, categ_map=categ_map, feature_names=feat_names
                )

                # Train TPM
                log.info("Training TPM...")
                tpm_start_time = time.time()
                tpm = train_tpm(cfg, tpm_data, data_handler)
                tpm_train_duration = time.time() - tpm_start_time
            else:
                tpm = None
                data_handler = None
                tpm_data = None
                tpm_train_duration = 0.0
            mlflow.log_metric("tpm_train_duration", tpm_train_duration)

            # 4. Build and Solve Model
            log.info(f"Building model for method: {cfg.method.name}")

            # For robust/sample average, we need scenarios
            if cfg.method.name in ["robust", "sample_average"]:
                # Use a subset of training samples
                opt_samples = train_samples[: cfg.samples.opt]
            else:
                opt_samples = None

            build_start_time = time.time()

            build_method_name = cfg.method.name
            if cfg.method.get("type") == "tpm":
                build_method_name = "tpm"

            problem.build_model(
                method=build_method_name,
                tpm=tpm,
                data_handler=data_handler,
                scenarios=opt_samples,
                risk_level=cfg.risk_level,
            )
            build_duration = time.time() - build_start_time
            mlflow.log_metric("build_duration", build_duration)

            log.info("Solving model...")
            solve_start_time = time.time()
            result = problem.solve()
            solve_duration = time.time() - solve_start_time

            log.info(f"Result: {result}")
            mlflow.log_metric("solve_duration", solve_duration)
            mlflow.log_metric(
                "total_duration", build_duration + solve_duration + tpm_train_duration
            )

            # Log solver status
            solver_status = str(result.get("status", "UNKNOWN"))
            mlflow.set_tag("solver_status", solver_status)

            if result["objective"] is not None:
                mlflow.log_metric("objective", result["objective"])

            # If solver failed (infeasible/unbounded), mark run as failed
            if solver_status.lower() not in ["optimal", "ok", "success"]:
                log.warning(f"Solver status is {solver_status}, marking run as failed.")
                mlflow.set_tag("status", "FAILED")
                mlflow.set_tag("error_type", "SOLVER_FAILURE")
                mlflow.set_tag(
                    "error_message", f"Solver returned status: {solver_status}"
                )
                if result.get("objective") is None:
                    return

            # 5. Evaluation / Verification
            log.info("Evaluating solution...")

            # Get solution vector
            try:
                x_sol = problem.get_solution()
                # Log solution vector to mlflow
                solution_dict = {f"x_{i}": float(x_sol[i]) for i in range(len(x_sol))}
                mlflow.log_dict(solution_dict, "solution.json")
                log.info(f"Solution: {x_sol}")
            except ValueError as e:
                log.warning(f"Could not retrieve solution: {e}")
                mlflow.set_tag("status", "FAILED")
                mlflow.set_tag("error_type", "NO_SOLUTION")
                mlflow.set_tag("error_message", str(e))
                return

            # Validation on new samples
            val_seed = cfg.seed + 1  # Different seed
            n_val = cfg.samples.get("validation", cfg.samples.test)
            validation_samples = problem.generate_samples(
                n_samples=n_val, seed=val_seed
            )

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

            # If TPM exists, log the probability of the solution
            if tpm is not None:
                log.info("Calculating P(x_sol) from TPM...")
                # pad x_sol with None for marginalized variables and 1 for the satisfied constraint
                x_sol = np.array(
                    [None] * (tpm.data_handler.n_features - len(x_sol) - 1)
                    + list(x_sol)
                    + [1]
                )
                p_x_sol = tpm.probability(x_sol)
                mlflow.log_metric(
                    "true_tpm_prob_satisfied", np.exp(p_x_sol - problem.x_log_density)
                )
                log.info(
                    "P(x_sol) from true TPM: "
                    + f"{np.exp(p_x_sol - problem.x_log_density)}"
                )
                # TODO make this somehow neat? also above, passing cfg is not ideal
                # TODO check also the division by p(x) if not uniform
                p_x_sol_approx = tpm.probability_approx(x_sol, **cfg.method)
                mlflow.log_metric(
                    "approx_tpm_prob_satisfied",
                    np.exp(p_x_sol_approx - problem.x_log_density),
                )
                log.info(
                    "P(x_sol) from approx TPM: "
                    + f"{np.exp(p_x_sol_approx - problem.x_log_density)}"
                )

            # Mark as successful
            mlflow.set_tag("status", "SUCCESS")
            log.info("Experiment completed successfully")

        except MemoryError as e:
            log.error(f"Out of memory error: {e}")
            mlflow.set_tag("status", "OOM")
            mlflow.set_tag("error_type", "OUT_OF_MEMORY")
            mlflow.set_tag("error_message", str(e))
            raise

        except TimeoutError as e:
            log.error(f"Timeout error: {e}")
            mlflow.set_tag("status", "TIMEOUT")
            mlflow.set_tag("error_type", "TIMEOUT")
            mlflow.set_tag("error_message", str(e))
            raise

        except Exception as e:
            log.error(f"Experiment failed with error: {e}", exc_info=True)
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error_type", type(e).__name__)
            mlflow.set_tag("error_message", str(e))
            # Log full traceback
            import traceback

            mlflow.log_text(traceback.format_exc(), "error_traceback.txt")
            raise
