from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pyomo.environ as pyo

class BaseProblem(ABC):
    """
    Abstract base class for optimization problems.
    """
    def __init__(self, cfg: Any, solver: str = "gurobi"):
        """
        Initialize the problem.

        Args:
            cfg: Hydra configuration object for the problem.
            solver: Name of the solver to use (default: "gurobi").
        """
        self.cfg = cfg
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

    def generate_tpm_data(self, train_samples: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, list[str]]:
        """
        Generate training data for TPM (xi, x, sat).

        Args:
            train_samples: Context samples (xi).
            seed: Random seed.

        Returns:
            Tuple of (data array, feature names).
        """
        if seed is not None:
            np.random.seed(seed)

        n_samples = train_samples.shape[0]

        # Generate x samples
        # We pass train_samples to help determine bounds if needed
        x_samples = self.generate_decision_samples(n_samples, seed=seed, xi=train_samples)

        # Compute satisfaction
        sat = self.compute_satisfaction(train_samples, x_samples)

        # Combine: [xi, x, sat]
        data = np.concatenate([train_samples, x_samples, sat], axis=1)

        # TODO  return feature names separately using another method
        # Feature names
        n_xi = train_samples.shape[1]
        n_x = x_samples.shape[1]
        feat_names = [f"xi_{i}" for i in range(n_xi)] + [f"x_{i}" for i in range(n_x)] + ["sat"]

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
