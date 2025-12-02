import logging
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pyomo.environ as pyo
from scipy.stats import norm, expon

from src.problem.base import BaseProblem

try:
    from src.tpms.tpm import TPM
except ImportError:
    logging.warning("Could not import TPM modules. TPM method will fail if used.")

class NewsvendorProblem(BaseProblem):
    """
    Newsvendor problem implementation.
    """
    def __init__(self, cfg: Any, solver: str = "gurobi"):
        super().__init__(cfg, solver)
        self.n_products = cfg.n_products
        self.costs = np.array(cfg.costs) if hasattr(cfg, 'costs') else np.ones(self.n_products)
        self.prices = np.array(cfg.prices) if hasattr(cfg, 'prices') else np.zeros(self.n_products)
        self.demand_dist = cfg.demand_dist # "normal", "exponential"
        self.demand_params = cfg.demand_params # dict with mean, std, etc.

    def generate_samples(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        samples = []
        for i in range(self.n_products):
            # Handle per-product parameters if list, else assume shared/scalar
            mean = self.demand_params.mean[i] if isinstance(self.demand_params.mean, (list, Any)) and len(self.demand_params.mean) == self.n_products else self.demand_params.mean

            if self.demand_dist == "normal":
                std = self.demand_params.std[i] if isinstance(self.demand_params.std, (list, Any)) and len(self.demand_params.std) == self.n_products else self.demand_params.std
                d = norm.rvs(loc=mean, scale=std, size=n_samples)
            elif self.demand_dist == "exponential":
                d = expon.rvs(scale=mean, size=n_samples)
            else:
                raise ValueError(f"Unknown distribution: {self.demand_dist}")
            samples.append(d.reshape(-1, 1))

        return np.concatenate(samples, axis=1)

        #  TODO implement the other abstract methods from the base class

    def build_model(
        self,
        method: str,
        tpm: Optional[TPM] = None,
        data_handler: Any = None,
        scenarios: Optional[np.ndarray] = None,
        risk_level: float = 0.05,
        **kwargs
    ) -> pyo.ConcreteModel:

        model = pyo.ConcreteModel()
        model.x = pyo.Var(range(self.n_products), domain=pyo.NonNegativeReals)
        demands = scenarios

        # Objective: Minimize Cost * x - Price * E[min(x, D)]
        # Simplified: Minimize sum(Cost * x)
        # TODO: Add expected sales term if prices are non-zero

        # TODO add costs/prices to config

        # Objective: Minimize sum(costs * x)
        model.obj = pyo.Objective(
            expr=sum(self.costs[i] * model.x[i] for i in range(self.n_products)),
            sense=pyo.minimize
        )

        if method == "robust":
            # Sample Robust: x >= d for all samples (or robust set)
            if demands is None:
                raise ValueError("Demands required for robust method")

            model.nsamples = pyo.Set(initialize=range(len(demands)))
            model.robust_constr = pyo.Constraint(
                model.nsamples, range(self.n_products),
                rule=lambda m, i, j: m.x[j] >= demands[i, j]
            )

        elif method == "sample_average":
            # Sample Average Approximation (SAA) with chance constraint
            if demands is None:
                raise ValueError("Demands required for sample average method")

            n_s = len(demands)
            model.nsamples = pyo.Set(initialize=range(n_s))
            model.y = pyo.Var(model.nsamples, domain=pyo.Binary) # y=1 if satisfied, 0 otherwise

            # Constraint: x >= D if y=1
            # Big-M formulation: x_j >= d_ij - M(1-y_i)
            # If y_i=1 (satisfied), x_j >= d_ij.
            # If y_i=0 (not satisfied), x_j >= d_ij - M (relaxed)

            M = np.max(demands) * 2

            model.chance_constr = pyo.Constraint(
                model.nsamples, range(self.n_products),
                rule=lambda m, i, j: m.x[j] >= demands[i, j] - M * (1 - m.y[i])
            )

            # Probability constraint: sum(y) >= (1-alpha) * N
            model.prob_constr = pyo.Constraint(
                expr=sum(model.y[i] for i in model.nsamples) >= (1 - risk_level) * n_s
            )

        elif method == "tpm":
            if tpm is None or data_handler is None:
                raise ValueError("TPM and DataHandler required for tpm method")

            # Unified TPM encoding
            model.tpm_block = pyo.Block()

            # Inputs: [xi (marginalized), x (vars), sat (1)]
            # xi columns are None (marginalized out)
            # x columns are model.x variables
            # sat column is 1 - meaning that the internal constraint is satisfied

            # TODO We can get the order of features from the DataHandler - some might be categorical or sth
            # TODO if the input is discrete, this will be different - raise an error for now
            inputs = [None] * self.n_products + [model.x[i] for i in range(self.n_products)] + [1]

            # Encode
            # TODO: Handle density of x

            # the tpm encode handles marginalization to P(x, sat)
            output = tpm.encode(model.tpm_block, inputs, **kwargs)

            target_log_prob = np.log(1 - risk_level)
            # TODO add the density of x here - either from TPM (encoded as another tpm_block and then subtracted from the output since both are log-probabilities) or from known density of x (this can be obtained from this class - it will be used in the generate_decision_samples method - make it uniform over the range of sampled demands for the newsvendor problem) - in that case it should multiply the (1-risk) by the density of x before logging

            model.chance_constr = pyo.Constraint(
                expr=output >= target_log_prob
            )

        self.model = model
        return model

