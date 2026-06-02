from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import pyomo.environ as pyo
from scipy.stats import expon, norm, uniform

from stochopt.data.Features import Binary, Categorical
from stochopt.problem.base import BaseProblem
from stochopt.tpms.nn_pm import NNPM
from stochopt.tpms.tpm import TPM


class NewsvendorProblem(BaseProblem):
    """
    Newsvendor problem implementation.
    """

    def __init__(
        self,
        n_products: int,
        costs: npt.NDArray[np.float64],
        prices: npt.NDArray[np.float64],
        demand_dist: str,
        demand_params: Dict[str, Any],
        x_density_type: str = "uniform",
        correlated: bool = False,
        solver: str = "gurobi",
        **kwargs,
    ):
        super().__init__(solver)
        self.n_products = n_products
        self.costs = np.array(costs) if costs is not None else np.ones(self.n_products)
        if self.costs.ndim == 0:
            self.costs = np.full(self.n_products, self.costs)
        self.prices = (
            np.array(prices) if prices is not None else np.zeros(self.n_products)
        )
        if self.prices.ndim == 0:
            self.prices = np.full(self.n_products, self.prices)
        self.demand_dist = demand_dist  # "normal", "exponential"
        self.demand_params = (
            demand_params  # dict with mean, std, or min/max depending on distribution
        )
        if "std" in self.demand_params:
            if isinstance(self.demand_params["std"], (list, tuple, np.ndarray)):
                assert len(self.demand_params["std"]) == self.n_products
        if "mean" in self.demand_params:
            if isinstance(self.demand_params["mean"], (list, tuple, np.ndarray)):
                assert len(self.demand_params["mean"]) == self.n_products
        if "min" in self.demand_params and "max" in self.demand_params:
            if isinstance(self.demand_params["min"], (list, tuple, np.ndarray)):
                assert len(self.demand_params["min"]) == self.n_products
            if isinstance(self.demand_params["max"], (list, tuple, np.ndarray)):
                assert len(self.demand_params["max"]) == self.n_products
        self.x_density_type = x_density_type
        self.correlated = correlated

    def _generate_correlated_samples(
        self,
        n_samples: int,
        means: npt.NDArray[np.float64],
        stds: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        common = norm.rvs(size=(n_samples, 1))
        noise = norm.rvs(scale=0.1, size=(n_samples, len(stds)))
        samples = np.maximum(means + stds * (common + noise), 0)
        return np.array(samples, dtype=np.float64)

    def _generate_correlated_samples_in_total(
        self, n_samples: int, n_products: int, total: float, std: float
    ) -> npt.NDArray[np.float64]:
        # generates samples such that sum of demands is equal to total
        total_samples = norm.rvs(loc=total, scale=std, size=(n_samples, 1))
        proportions = np.random.dirichlet(np.ones(n_products), size=n_samples)
        return np.array(total_samples * proportions, dtype=np.float64)

    def generate_samples(
        self, n_samples: int, seed: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        if seed is not None:
            np.random.seed(seed)

        if self.correlated:
            # generate half of products correlated and half correlated in total
            correlated_samples = self._generate_correlated_samples(
                n_samples,
                self.demand_params["mean"][: self.n_products // 2],
                self.demand_params["std"][: self.n_products // 2],
            )
            correlated_total_samples = self._generate_correlated_samples_in_total(
                n_samples,
                self.n_products // 2,
                sum(self.demand_params["mean"][self.n_products // 2 :]),
                sum(self.demand_params["std"][self.n_products // 2 :]),
            )
            return np.concatenate(
                [correlated_samples, correlated_total_samples], axis=1
            )

        samples = []
        for i in range(self.n_products):
            # Handle per-product parameters if list, else assume shared/scalar

            if self.demand_dist == "normal":
                mean = (
                    self.demand_params["mean"][i]
                    if isinstance(self.demand_params["mean"], (list, tuple, np.ndarray))
                    else self.demand_params["mean"]
                )
                std = (
                    self.demand_params["std"][i]
                    if isinstance(self.demand_params["std"], (list, tuple, np.ndarray))
                    else self.demand_params["std"]
                )
                d = norm.rvs(loc=mean, scale=std, size=n_samples)
            elif self.demand_dist == "exponential":
                mean = (
                    self.demand_params["mean"][i]
                    if isinstance(self.demand_params["mean"], (list, tuple, np.ndarray))
                    else self.demand_params["mean"]
                )
                d = expon.rvs(scale=mean, size=n_samples)
            elif self.demand_dist == "uniform":
                min_val = (
                    self.demand_params["min"][i]
                    if isinstance(self.demand_params["min"], (list, tuple, np.ndarray))
                    else self.demand_params["min"]
                )
                max_val = (
                    self.demand_params["max"][i]
                    if isinstance(self.demand_params["max"], (list, tuple, np.ndarray))
                    else self.demand_params["max"]
                )
                d = uniform.rvs(loc=min_val, scale=max_val - min_val, size=n_samples)
            else:
                raise ValueError(f"Unknown distribution: {self.demand_dist}")
            samples.append(d.reshape(n_samples, 1).round())

        return np.concatenate(samples, axis=1, dtype=np.float64)

    def generate_decision_samples(
        self,
        n_samples: int,
        seed: Optional[int] = None,
        demands: Optional[npt.NDArray[np.float64]] = None,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """
        Generate decision variable samples (order quantities).

        Args:
            n_samples: Number of samples to generate.
            seed: Random seed.
            demands: Demand samples to determine bounds.
            **kwargs: Additional arguments.

        Returns:
            npt.NDArray[np.float64]: Decision samples with shape (n_samples, n_products).
        """
        if seed is not None:
            np.random.seed(seed)

        if self.x_density_type == "uniform":
            # Determine bounds from demand samples if provided
            if demands is None and "xi" in kwargs:
                demands = kwargs["xi"]
            if demands is not None:
                min_demand = np.min(demands, axis=0)
                max_demand = np.max(demands, axis=0)
            else:
                # Use distribution parameters as fallback
                min_demand = np.zeros(self.n_products)
                # Calculate max based on distribution type
                max_demand = []
                for i in range(self.n_products):
                    if self.demand_dist == "normal":
                        # For normal distribution, use mean + 3*std (covers ~99.7% of values)
                        mean = (
                            self.demand_params["mean"][i]
                            if isinstance(
                                self.demand_params["mean"],
                                (list, tuple, np.ndarray),
                            )
                            else self.demand_params["mean"]
                        )
                        std = (
                            self.demand_params["std"][i]
                            if isinstance(
                                self.demand_params["std"],
                                (list, tuple, np.ndarray),
                            )
                            else self.demand_params["std"]
                        )
                        max_val = mean + 3 * std
                    elif self.demand_dist == "exponential":
                        # For exponential,use mean * 4 (covers >98% of values)
                        mean = (
                            self.demand_params["mean"][i]
                            if isinstance(
                                self.demand_params["mean"],
                                (list, tuple, np.ndarray),
                            )
                            else self.demand_params["mean"]
                        )
                        max_val = mean * 4
                    elif self.demand_dist == "uniform":
                        min_demand[i] = (
                            self.demand_params["min"][i]
                            if isinstance(
                                self.demand_params["min"],
                                (list, tuple, np.ndarray),
                            )
                            else self.demand_params["min"]
                        )
                        max_val = (
                            self.demand_params["max"][i]
                            if isinstance(
                                self.demand_params["max"],
                                (list, tuple, np.ndarray),
                            )
                            else self.demand_params["max"]
                        )
                    else:
                        raise ValueError(
                            f"Unknown distribution type: {self.demand_dist}"
                        )
                    max_demand.append(max_val)
                max_demand = np.array(max_demand)

            # Generate uniform samples in [min_demand, max_demand]
            x_samples = []
            self.x_log_density = 0
            for i in range(self.n_products):
                x = np.random.uniform(min_demand[i], max_demand[i], size=n_samples)
                x_samples.append(x.reshape(-1, 1).round())
                self.x_log_density -= np.log(max_demand[i] - min_demand[i])

            return np.concatenate(x_samples, axis=1, dtype=np.float64)
        else:
            raise ValueError(f"Unknown distribution type: {self.demand_dist}")

    def compute_satisfaction(
        self, xi: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute satisfaction status for newsvendor constraint: x >= demand.

        Args:
            xi: Demand samples with shape (n_samples, n_products).
            x: Decision samples with shape (n_samples, n_products).

        Returns:
            npt.NDArray[np.bool_]: Binary column vector (n_samples, 1) where 1 means satisfied.
        """
        # Constraint is satisfied if x_j >= xi_j for all products j
        satisfied = np.array(np.all(x >= xi, axis=1, keepdims=True), dtype=bool)
        return satisfied

    def get_exact_prob_satisfied(self, x_sol: npt.NDArray[np.float64]) -> float:
        """
        Compute the exact probability of satisfaction for the given decision variables.
        """
        if self.correlated:
            raise NotImplementedError(
                "Exact probability not implemented for correlated demands."
            )

        prob = 1.0
        for i in range(self.n_products):
            # The demand is rounded in generate_samples: round(D)
            # We want P(round(D) <= x_sol[i])
            # Since x_sol[i] is integer, this is P(D < x_sol[i] + 0.5)
            x_val = x_sol[i] + 0.5

            if self.demand_dist == "normal":
                mean = (
                    self.demand_params["mean"][i]
                    if isinstance(self.demand_params["mean"], (list, tuple, np.ndarray))
                    else self.demand_params["mean"]
                )
                std = (
                    self.demand_params["std"][i]
                    if isinstance(self.demand_params["std"], (list, tuple, np.ndarray))
                    else self.demand_params["std"]
                )
                p_i = norm.cdf(x_val, loc=mean, scale=std)
            elif self.demand_dist == "exponential":
                mean = (
                    self.demand_params["mean"][i]
                    if isinstance(self.demand_params["mean"], (list, tuple, np.ndarray))
                    else self.demand_params["mean"]
                )
                p_i = expon.cdf(x_val, scale=mean)
            elif self.demand_dist == "uniform":
                min_val = (
                    self.demand_params["min"][i]
                    if isinstance(self.demand_params["min"], (list, tuple, np.ndarray))
                    else self.demand_params["min"]
                )
                max_val = (
                    self.demand_params["max"][i]
                    if isinstance(self.demand_params["max"], (list, tuple, np.ndarray))
                    else self.demand_params["max"]
                )
                p_i = uniform.cdf(x_val, loc=min_val, scale=max_val - min_val)
            else:
                raise ValueError(f"Unknown distribution: {self.demand_dist}")
            prob *= p_i

        return float(prob)

    def compute_margin(
        self, xi: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Compute continuous satisfaction margin for newsvendor constraint.
        Margin is min_j (x_j - xi_j).
        """
        return cast(npt.NDArray[np.float64], np.min(x - xi, axis=1, keepdims=True))

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

    def get_solution(self) -> npt.NDArray[np.float64]:
        """
        Extract the solution from the solved model.

        Returns:
            npt.NDArray[np.float64]: Solution vector with shape (n_products,).

        Raises:
            ValueError: If model is not solved or infeasible.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        try:
            # TODO work with pyomo solve status here - infeasible, unbounded, timeout...
            # - log that fact in mlflow? retrun it from this function and log it
            # in mlflow in the runner calling this function
            solution = np.array(
                [pyo.value(self.model.x[i]) for i in range(self.n_products)]
            )
            if any(v is None for v in solution):
                raise ValueError(
                    "Model solution contains None values. Model may be infeasible."
                )
            return solution
        except Exception as e:
            raise ValueError(f"Could not extract solution: {e}")

    def build_model(
        self,
        method: str,
        tpm: Optional[TPM | NNPM] = None,
        data_handler: Any = None,
        scenarios: Optional[npt.NDArray[np.float64]] = None,
        risk_level: float = 0.05,
        epsilon: float = 1e-6,
        **kwargs,
    ) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()
        model.x = pyo.Var(range(self.n_products), domain=pyo.Integers, bounds=(0, None))
        demands = scenarios

        # Objective: Minimize Cost * x - Price * E[min(x, D)]
        # Simplified: Minimize sum(Cost * x)
        # TODO: Add expected sales term if prices are non-zero

        # Objective: Minimize sum(costs * x)
        model.obj = pyo.Objective(
            expr=sum(self.costs[i] * model.x[i] for i in range(self.n_products)),
            sense=pyo.minimize,
        )

        if method == "robust":
            # Sample Robust: x >= d for all samples (or robust set)
            if demands is None:
                raise ValueError("Demands required for robust method")

            model.nsamples = pyo.Set(initialize=range(len(demands)))
            model.robust_constr = pyo.Constraint(
                model.nsamples,
                range(self.n_products),
                rule=lambda m, i, j: m.x[j] >= demands[i, j],
            )

        elif method == "sample_average":
            # Sample Average Approximation (SAA) with chance constraint
            if demands is None:
                raise ValueError("Demands required for sample average method")

            n_s = len(demands)
            model.nsamples = pyo.Set(initialize=range(n_s))
            model.y = pyo.Var(
                model.nsamples, domain=pyo.Binary
            )  # y=1 if satisfied, 0 otherwise

            # Constraint: x >= D if y=1
            # Big-M formulation: x_j >= d_ij * y_i
            # If y_i=1 (satisfied), x_j >= d_ij.
            # If y_i=0 (not satisfied), x_j >= 0

            model.chance_constr = pyo.Constraint(
                model.nsamples,
                range(self.n_products),
                rule=lambda m, i, j: m.x[j] >= demands[i, j] * m.y[i] + epsilon,
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
            if any(
                isinstance(f, Categorical) or isinstance(f, Binary)
                for f in data_handler.features[self.n_products : -1]
            ):
                raise ValueError("Categorical or binary features not supported yet")

            inputs = (
                [None] * self.n_products
                + [model.x[i] for i in range(self.n_products)]
                + [1]
            )

            # Encode
            # TODO: Handle density of x

            # the tpm encode handles marginalization to P(x, sat)
            output = tpm.encode(
                model.tpm_block, inputs, solver=self.solver_name, **kwargs
            )

            if hasattr(self, "x_log_density"):
                target_log_prob = np.log(1 - risk_level) + self.x_log_density
            else:
                model.x_density_block = pyo.Block()
                # marginalize out the satisfaction variable as well
                inputs_x_density = inputs[:-1] + [None]
                x_density = tpm.encode(
                    model.x_density_block,
                    inputs_x_density,
                    solver=self.solver_name,
                    **kwargs,
                )

                target_log_prob = np.log(1 - risk_level) + x_density

            model.chance_constr = pyo.Constraint(expr=output >= target_log_prob)

        elif method == "nn_pm":
            if tpm is None:
                raise ValueError("NNPM object required for nn_pm method")

            from stochopt.tpms.nn_pm import NNPM

            if not isinstance(tpm, NNPM):
                raise TypeError(f"Expected NNPM, got {type(tpm)}")

            model.nn_block = pyo.Block()
            nn_inputs = [model.x[i] for i in range(self.n_products)]

            logit_output = tpm.encode(
                model.nn_block, nn_inputs, solver=self.solver_name, **kwargs
            )

            # Constraint: sigmoid(logit) >= 1 - alpha
            # Equivalent: logit >= log((1 - alpha) / alpha)
            target_logit = float(np.log((1 - risk_level) / risk_level))
            model.chance_constr = pyo.Constraint(expr=logit_output >= target_logit)

        elif method == "quantile_nn":
            if tpm is None:
                raise ValueError("QuantileNN object required for quantile_nn method")

            from stochopt.quantile.quantile_nn import QuantileNN

            if not isinstance(tpm, QuantileNN):
                raise TypeError(f"Expected QuantileNN, got {type(tpm)}")

            model.nn_block = pyo.Block()
            nn_inputs = [model.x[i] for i in range(self.n_products)]

            margin_output = tpm.encode(
                model.nn_block, nn_inputs, solver=self.solver_name, **kwargs
            )

            # Constraint: The q-th quantile of the margin must be >= 0
            model.chance_constr = pyo.Constraint(expr=margin_output >= 0)

        self.model = model
        return model

    def get_categ_map(self) -> dict[str, list[int | str]]:
        """
        Get mapping of categorical features for all samples.

        Returns:
            dict[str, list[int | str]]: Mapping of categorical features.
        """
        return {"sat": [0, 1]}

    def get_discrete(self) -> list[str]:
        """
        Get a list of discrete continuous features for all samples.

        Returns:
            list[str]: List of discrete features.
        """
        xi, x, sat = self.get_feature_names()
        return xi + x  # sat feature is categorical
