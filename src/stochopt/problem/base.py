import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyomo.environ as pyo

logger = logging.getLogger(__name__)


class BaseProblem(ABC):
    """
    Abstract base class for optimization problems.
    """

    def __init__(self, solver: str = "gurobi"):
        """
        Initialize the problem.

        Args:
            solver: Name of the solver to use (default: "gurobi").
        """
        self.solver_name = solver
        self.model: Optional[pyo.ConcreteModel] = None

    @abstractmethod
    def build_model(
        self,
        method: str,
        tpm: Any = None,
        data_handler: Any = None,
        scenarios: Optional[npt.NDArray[np.float64]] = None,
        risk_level: float = 0.05,
        epsilon: float = 1e-6,
        **kwargs,
    ) -> pyo.ConcreteModel:
        """
        Build the Pyomo model for the problem.

        Args:
            method: Optimization method ("robust", "sample_average", "tpm").
            tpm: Trained TPM object (for "tpm" method).
            data_handler: DataHandler object (for "tpm" method).
            scenarios: Data scenarios (for "sample_average" or "robust" methods).
            risk_level: Risk level (alpha) for chance constraints.
            **kwargs: Additional arguments.

        Returns:
            pyo.ConcreteModel: The built Pyomo model.
        """
        pass

    @abstractmethod
    def generate_samples(
        self, n_samples: int, seed: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        """
        Generate data samples for the problem.

        Args:
            n_samples: Number of samples to generate.
            seed: Random seed.

        Returns:
            np.ndarray: Generated samples.
        """
        pass

    @abstractmethod
    def generate_decision_samples(
        self, n_samples: int, seed: Optional[int] = None, **kwargs
    ) -> npt.NDArray[np.float64]:
        """
        Generate samples for decision variables (x).

        Args:
            n_samples: Number of samples.
            seed: Random seed.
            **kwargs: Additional arguments (e.g., bounds based on xi).

        Returns:
            np.ndarray: Generated decision samples.
        """
        pass

    @abstractmethod
    def compute_satisfaction(
        self, xi: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute satisfaction status for given xi and x.

        Args:
            xi: Context samples (e.g., demand).
            x: Decision samples.

        Returns:
            np.ndarray: Satisfaction status (binary column vector).
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> Tuple[List[str], List[str], str]:
        """
        Get feature names for TPM data.

        Returns:
            Tuple of (xi_names, x_names, sat_name).
        """
        pass

    def check_satisfaction(
        self, x_sol: npt.NDArray[np.float64], scenarios: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """
        Check if solution satisfies constraints for given scenarios.
        This is a wrapper around compute_satisfaction for evaluation purposes.

        Args:
            x_sol: Solution vector (decision variables).
            scenarios: Scenario data (xi samples).

        Returns:
            np.ndarray: Binary array indicating satisfaction for each scenario.
        """
        # Expand x_sol to match number of scenarios
        n_scenarios = scenarios.shape[0]
        x_expanded = np.tile(x_sol, (n_scenarios, 1))
        return self.compute_satisfaction(scenarios, x_expanded).flatten()

    @abstractmethod
    def get_solution(self) -> npt.NDArray[np.float64]:
        """
        Extract the solution from the solved model.

        Returns:
            np.ndarray: Solution vector.

        Raises:
            ValueError: If model is not solved or infeasible.
        """
        pass

    def get_objective(self) -> float:
        """
        Extract the objective value from the solved model.

        Returns:
            float: Objective value.

        Raises:
            ValueError: If model is not solved or infeasible.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        try:
            return float(pyo.value(self.model.obj))
        except Exception as e:
            raise ValueError(f"Could not extract objective: {e}")

    def generate_tpm_data(
        self,
        n_decisions: int,
        train_samples: npt.NDArray[np.float64],
        cartesian_product: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.float64], list[str]]:
        """
        Generate training data for TPM (xi, x, sat).

        Args:
            n_decisions: Number of decision samples.
            train_samples: Context samples (xi).
            cartesian_product: If True, generate Cartesian product of
                train_samples and generated x_samples. If False,
                generate n_decisions (xi, x) pairs by sampling
                from train_samples.
            seed: Random seed.

        Returns:
            Tuple of (data array, feature names).
        """
        if seed is not None:
            np.random.seed(seed)

        n_samples = train_samples.shape[0]

        if cartesian_product:
            # Generate x samples (size n_decisions)
            # We pass train_samples to help determine bounds if needed
            x_samples = self.generate_decision_samples(
                n_decisions, seed=seed, xi=train_samples
            )

            # Pair each xi with each x (Cartesian product)
            # Repeat xi: [xi1, xi1, ..., xi2, xi2, ...]
            xi_expanded = np.repeat(train_samples, n_decisions, axis=0)
            # Tile x: [x1, x2, ..., x1, x2, ...]
            x_expanded = np.tile(x_samples, (n_samples, 1))
        else:
            # Sample n_decisions indices from train_samples with replacement
            indices = np.random.choice(n_samples, size=n_decisions, replace=True)
            xi_expanded = train_samples[indices]

            # Generate n_decisions x samples
            # We pass all train_samples to help determine bounds if needed
            x_expanded = self.generate_decision_samples(
                n_decisions, seed=seed, xi=train_samples
            )

        # Compute satisfaction on expanded data
        sat = self.compute_satisfaction(xi_expanded, x_expanded).astype(float)

        # Combine: [xi, x, sat]
        data = np.concatenate([xi_expanded, x_expanded, sat], axis=1)

        # Get feature names from subclass
        xi_names, x_names, sat_name = self.get_feature_names()
        feat_names = xi_names + x_names + [sat_name]

        return data, feat_names

    def solve(self, time_limit: Optional[float] = None) -> Dict[str, Any]:
        """
        Solve the built model.

        Args:
            time_limit: Time limit for the solver.

        Returns:
            Dict[str, Any]: Dictionary containing solution status, objective
                value, variable values, and MIP gap.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")

        solver = pyo.SolverFactory(self.solver_name)
        if time_limit is not None:
            if "cplex" in self.solver_name:
                solver.options["timelimit"] = time_limit
            elif "glpk" in self.solver_name:
                solver.options["tmlim"] = time_limit
            elif "xpress" in self.solver_name:
                solver.options["soltimelimit"] = time_limit
                # Use the below instead for XPRESS versions before 9.0
                # solver.options['maxtime'] = time_limit
            elif "highs" in self.solver_name:
                solver.options["time_limit"] = time_limit
            elif "gurobi" in self.solver_name:
                solver.options["TimeLimit"] = time_limit
                # solver.options["Aggregate"] = 0
                # solver.options["OptimalityTol"] = 1e-3
                # solver.options["IntFeasTol"] = self.MIO_EPS / 10
                # solver.options["FeasibilityTol"] = self.MIO_EPS / 10
            else:
                logger.warning(
                    f"Time limit not set! Not implemented for solver {self.solver_name}"
                )

        result = solver.solve(self.model, tee=False)

        status = str(result.solver.termination_condition)

        # Safely extract objective and solution
        obj_val = None
        try:
            obj_val = self.get_objective()
        except (ValueError, AttributeError):
            pass

        sol_val = None
        try:
            sol_val = self.get_solution()
        except (ValueError, AttributeError):
            pass

        res = {
            "status": status,
            "objective": obj_val,
            "solution": sol_val,
        }

        return res

    @abstractmethod
    def get_categ_map(self) -> dict[str, list[int | str]]:
        """
        Get mapping of categorical features for all samples.

        Returns:
            dict[str, list[int | str]]: Mapping of categorical features.
        """
        pass

    @abstractmethod
    def get_discrete(self) -> list[str]:
        """
        Get a list of discrete continuous features for all samples.

        Returns:
            list[str]: List of discrete features.
        """
        pass
