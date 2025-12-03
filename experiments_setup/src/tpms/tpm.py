from abc import ABC, abstractmethod
from typing import Any, List, Optional
import numpy as np
import pyomo.environ as pyo
from src.data.DataHandler import DataHandler

class TPM(ABC):
    """
    Abstract base class for Tractable Probabilistic Models.
    """

    def __init__(self):
        """Initialize TPM. Subclasses should call super().__init__()."""
        self.data_handler: Optional[DataHandler] = None

    @abstractmethod
    def train(self, data: np.ndarray, data_handler: DataHandler, **kwargs) -> Any:
        """
        Train the TPM.

        Args:
            data: Training data (numpy array).
            data_handler: DataHandler object.
            **kwargs: Hyperparameters. Can include:
                - xi_indices: Indices of xi variables to marginalize (future feature)

        Returns:
            self

        Note:
            Implementations should store data_handler: self.data_handler = data_handler
        """
        pass

    @abstractmethod
    def encode(self, model_block: pyo.Block, inputs: List[Any], **kwargs) -> Any:
        """
        Encode the TPM into a Pyomo model block.

        Args:
            model_block: Pyomo Block to add constraints to.
            inputs: List of input variables/values for the TPM.
                    Use None for marginalized variables.
            **kwargs: Additional arguments.

        Returns:
            Output log-likelihood variable.

        Note:
            Implementations should validate: len(inputs) == self.data_handler.n_features
            If lengths don't match, raise ValueError.
        """
        pass

    @property
    @abstractmethod
    def out_node_id(self) -> int:
        """
        Return the ID of the output node (root).
        """
        pass
