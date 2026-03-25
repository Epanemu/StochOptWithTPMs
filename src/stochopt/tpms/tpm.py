from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import pyomo.environ as pyo

from stochopt.data.DataHandler import DataHandler


class TPM(ABC):
    """
    Abstract base class for Tractable Probabilistic Models.
    """

    def __init__(self, data_handler: DataHandler):
        """
        Initialize TPM. Subclasses should call super().__init__().

        Args:
            data_handler: DataHandler
                The DataHandler object providing metadata about the features
                (e.g., categorical vs. continuous).
        """
        self.data_handler = data_handler

    @abstractmethod
    def train(self, data: npt.NDArray[np.float64], **kwargs: Any) -> "TPM":
        """
        Train the Tractable Probabilistic Model (TPM) using the provided data.

        Args:
            data: npt.NDArray[np.float64]
                The training data, typically a 2D array where rows are instances
                and columns are features.
            **kwargs: Any
                Hyperparameters for the training algorithm. Common keys include:
                - xi_indices: Indices of variables to marginalize (future feature).
                - min_instances_slice: Minimum instances per slice (for CNet/TreeTPM).
                - max_depth: Maximum tree depth.

        Returns:
            TPM: The trained instance (self).

        Note:
            Implementations must store the data_handler: self.data_handler = data_handler.
        """
        pass

    @abstractmethod
    def encode(
        self,
        model_block: pyo.Block,
        inputs: Sequence[
            Optional[Union[pyo.Var, float, npt.NDArray[np.float64], List[pyo.Var]]]
        ],
        solver: str = "gurobi",
        **kwargs: Any,
    ) -> pyo.Var:
        """
        Encode the TPM's log-probability function into a Pyomo model block.

        Args:
            model_block: pyo.Block
                The Pyomo Block to which constraints and variables will be added.
            inputs: List[Optional[Union[pyo.Var, float, npt.NDArray[np.float64], List[pyo.Var]]]]
                A list containing the inputs for each feature.
                - Use `None` for variables that should be marginalized out.
                - For categorical features, this may be a list of one-hot variables.
                - For continuous features, this is typically a single Pyomo variable or constant.
            solver: str
                The solver intended for the model (default "gurobi").
                Encoding details may vary depending on the solver's capabilities.
            **kwargs: Any
                Additional arguments for the encoding process.

        Returns:
            pyo.Var: A Pyomo variable representing the model's log-probability (or density).

        Raises:
            ValueError: If `len(inputs)` does not match `self.data_handler.n_features`.
        """
        pass

    @abstractmethod
    def log_probability(self, sample: npt.NDArray[np.float64], **kwargs: Any) -> float:
        """
        Calculate the exact log-probability (or log-probability density) of a given sample.

        Args:
            sample: npt.NDArray[np.float64]
                A 1D array representing a single instance.
            **kwargs: Any
                Additional arguments for the inference process.

        Returns:
            float: The log-probability (or log-density) value.
        """
        pass

    @abstractmethod
    def log_probability_approx(
        self, sample: npt.NDArray[np.float64], **kwargs: Any
    ) -> float:
        """
        Calculate an approximate log-probability (or log-density) of a given sample.

        This is useful for models where exact inference is intractable or
        computationally expensive, or for specific shortcut calculations.

        Args:
            sample: npt.NDArray[np.float64]
                A 1D array representing a single instance.
            **kwargs: Any
                Additional arguments for the approximation method.

        Returns:
            float: The approximate log-probability (or log-density) value.
        """
        pass
