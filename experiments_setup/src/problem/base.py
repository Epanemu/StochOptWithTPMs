from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pyomo.environ as pyo

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
        scenarios: Optional[np.ndarray] = None,
        risk_level: float = 0.05,
        epsilon: float = 1e-6,
        **kwargs
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
    def generate_samples(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
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
    def generate_decision_samples(self, n_samples: int, seed: Optional[int] = None, **kwargs) -> np.ndarray:
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
    def compute_satisfaction(self, xi: np.ndarray, x: np.ndarray) -> np.ndarray:
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

    def check_satisfaction(self, x_sol: np.ndarray, scenarios: np.ndarray) -> np.ndarray:
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
    def get_solution(self) -> np.ndarray:
        """
        Extract the solution from the solved model.

        Returns:
            np.ndarray: Solution vector.

        Raises:
            ValueError: If model is not solved or infeasible.
        """
        pass

    # TODO: separate the n_decisions parameter from n_pairing parameter which would choose a subset of decision variables to pair with each training sample of xi
    def generate_tpm_data(self, n_decisions: int, train_samples: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, list[str]]:
        """
        Generate training data for TPM (xi, x, sat).

        Args:
            n_decisions: Number of decision samples.
            train_samples: Context samples (xi).
            seed: Random seed.

        Returns:
            Tuple of (data array, feature names).
        """
        if seed is not None:
            np.random.seed(seed)

        n_samples = train_samples.shape[0]

        # Generate x samples (size n_decisions)
        # We pass train_samples to help determine bounds if needed
        x_samples = self.generate_decision_samples(n_decisions, seed=seed, xi=train_samples)

        # Pair each xi with each x (Cartesian product)
        # Repeat xi: [xi1, xi1, ..., xi2, xi2, ...]
        train_samples_expanded = np.repeat(train_samples, n_decisions, axis=0)
        # Tile x: [x1, x2, ..., x1, x2, ...]
        x_samples_expanded = np.tile(x_samples, (n_samples, 1))

        # Compute satisfaction on expanded data
        sat = self.compute_satisfaction(train_samples_expanded, x_samples_expanded)

        # Combine: [xi, x, sat]
        data = np.concatenate([train_samples_expanded, x_samples_expanded, sat], axis=1)

        # Get feature names from subclass
        xi_names, x_names, sat_name = self.get_feature_names()
        feat_names = xi_names + x_names + [sat_name]

        return data, feat_names

    def solve(self) -> Dict[str, Any]:
        """
        Solve the built model.

        Returns:
            Dict[str, Any]: Dictionary containing solution status, objective value, and variable values.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")

        solver = pyo.SolverFactory(self.solver_name)
        result = solver.solve(self.model, tee=False)

        status = result.solver.termination_condition

        res = {
            "status": str(status),
            "objective": None,
            "solution": None,
        }

        return res
