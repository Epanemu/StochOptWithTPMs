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
        if isinstance(self.demand_params.std, (list, Any)):
            assert len(self.demand_params.std) == self.n_products
        if isinstance(self.demand_params.mean, (list, Any)):
            assert len(self.demand_params.mean) == self.n_products
        self.x_density_type = cfg.x_density if hasattr(cfg, 'x_density') else "uniform"

    def generate_samples(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        samples = []
        for i in range(self.n_products):
            # Handle per-product parameters if list, else assume shared/scalar
            mean = self.demand_params.mean[i] if isinstance(self.demand_params.mean, (list, Any)) else self.demand_params.mean

            if self.demand_dist == "normal":
                std = self.demand_params.std[i] if isinstance(self.demand_params.std, (list, Any)) else self.demand_params.std
                d = norm.rvs(loc=mean, scale=std, size=n_samples)
            elif self.demand_dist == "exponential":
                d = expon.rvs(scale=mean, size=n_samples)
            else:
                raise ValueError(f"Unknown distribution: {self.demand_dist}")
            samples.append(d.reshape(-1, 1))

        return np.concatenate(samples, axis=1)

    def generate_decision_samples(self, n_samples: int, seed: Optional[int] = None, demands: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate decision variable samples (order quantities).

        Args:
            n_samples: Number of samples to generate.
            seed: Random seed.
            demands: Demand samples to determine bounds.
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: Decision samples with shape (n_samples, n_products).
        """
        if seed is not None:
            np.random.seed(seed)

        if self.x_density_type == "uniform":
            # Determine bounds from demand samples if provided
            if demands is not None:
                min_demand = np.min(demands, axis=0)
                max_demand = np.max(demands, axis=0)
            else:
                # Use distribution parameters as fallback
                min_demand = np.zeros(self.n_products)
                # Calculate max based on distribution type
                max_demand = []
                for i in range(self.n_products):
                    mean = self.demand_params.mean[i] if isinstance(self.demand_params.mean, (list, Any)) else self.demand_params.mean
                    if self.demand_dist == "normal":
                        # For normal distribution, use mean + 3*std (covers ~99.7% of values)
                        std = self.demand_params.std[i] if isinstance(self.demand_params.std, (list, Any)) else self.demand_params.std
                        max_val = mean + 3 * std
                    elif self.demand_dist == "exponential":
                        # For exponential,use mean * 3 (covers ~95% of values)
                        max_val = mean * 3
                    else:
                        # Fallback: use mean * 2
                        max_val = mean * 2
                    max_demand.append(max_val)
                max_demand = np.array(max_demand)

            # Generate uniform samples in [min_demand, max_demand]
            x_samples = []
            self.x_log_density = 0
            for i in range(self.n_products):
                x = np.random.uniform(min_demand[i], max_demand[i], size=n_samples)
                x_samples.append(x.reshape(-1, 1))
                self.x_log_density -= np.log(max_demand[i] - min_demand[i])

            return np.concatenate(x_samples, axis=1)

    def compute_satisfaction(self, xi: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Compute satisfaction status for newsvendor constraint: x >= demand.

        Args:
            xi: Demand samples with shape (n_samples, n_products).
            x: Decision samples with shape (n_samples, n_products).

        Returns:
            np.ndarray: Binary column vector (n_samples, 1) where 1 means satisfied.
        """
        # Constraint is satisfied if x_j >= xi_j for all products j
        satisfied = np.all(x >= xi, axis=1, keepdims=True).astype(float)
        return satisfied

    def get_feature_names(self) -> Tuple[List[str], List[str], str]:
        """
        Get feature names for TPM data.

        Returns:
            Tuple of (xi_names, x_names, sat_name).
        """
        xi_names = [f"demand_{i}" for i in range(self.n_products)]
        x_names = [f"order_{i}" for i in range(self.n_products)]
        sat_name = "sat"
        return xi_names, x_names, sat_name

    def get_solution(self) -> np.ndarray:
        """
        Extract the solution from the solved model.

        Returns:
            np.ndarray: Solution vector with shape (n_products,).

        Raises:
            ValueError: If model is not solved or infeasible.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        try:
            # TODO work with pyomo solve status here - infeasible, unbounded, timeout etc. - log that fact in mlflow? retrun it from this function and log it in mlflow in the runner calling this function
            solution = np.array([pyo.value(self.model.x[i]) for i in range(self.n_products)])
            if any(v is None for v in solution):
                raise ValueError("Model solution contains None values. Model may be infeasible.")
            return solution
        except Exception as e:
            raise ValueError(f"Could not extract solution: {e}")

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

            if hasattr(self, "x_log_density"):
                target_log_prob = np.log(1 - risk_level) + self.x_log_density
            else:
                model.x_density_block = pyo.Block()
                # marginalize out the satisfaction variable as well
                inputs_x_density = inputs[:-1] + [None]
                x_density = tpm.encode(model.x_density_block, inputs_x_density, **kwargs)

                target_log_prob = np.log(1 - risk_level) + x_density

            model.chance_constr = pyo.Constraint(
                expr=output >= target_log_prob
            )

        self.model = model
        return model

