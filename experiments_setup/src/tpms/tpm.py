from abc import ABC, abstractmethod
from typing import Any, List, Optional
import numpy as np
import pyomo.environ as pyo
from src.data.DataHandler import DataHandler

class TPM(ABC):
    """
    Abstract base class for Tractable Probabilistic Models.
    """

    # TODO also pass indices of xi vars - train a marginalized TPM - either marginalize it, or train it only on the respective vars - also store the datahandler in the object so that it can be used in encoding
    @abstractmethod
    def train(self, data: np.ndarray, data_handler: DataHandler, **kwargs) -> Any:
        """
        Train the TPM.

        Args:
            data: Training data (numpy array).
            data_handler: DataHandler object.
            **kwargs: Hyperparameters.

        Returns:
            self
        """
        pass

    # TODO implement a check for the lenght of inputs corresponding to the length expected by the datahandler - the n_features of the datahandler - if the lenghts are different raise an error
    @abstractmethod
    def encode(self, model_block: pyo.Block, inputs: List[Any], **kwargs) -> Any:
        """
        Encode the TPM into a Pyomo model block.

        Args:
            model_block: Pyomo Block to add constraints to.
            inputs: List of input variables/values for the TPM.
            **kwargs: Additional arguments.

        Returns:
            Outputs from the encoding (e.g., node values).
        """
        pass

    @property
    @abstractmethod
    def out_node_id(self) -> int:
        """
        Return the ID of the output node (root).
        """
        pass
