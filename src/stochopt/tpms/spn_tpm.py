import copy
import logging
from typing import Any, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pyomo.environ as pyo

from stochopt.data.DataHandler import DataHandler
from stochopt.tpms.SPN.spn import SPN
from stochopt.tpms.SPN.spn_enc import encode_spn
from stochopt.tpms.tpm import TPM

logger = logging.getLogger(__name__)


class SpnTPM(TPM):
    """
    SPN-based TPM implementation.
    """

    def __init__(self, data_handler: DataHandler) -> None:
        """
        Initialize the SpnTPM.
        """
        super().__init__(data_handler)
        self.model: SPN | None = None

    def train(self, data: npt.NDArray[np.float64], **kwargs: Any) -> "SpnTPM":
        """
        Train the Sum-Product Network (SPN).

        Args:
            data: npt.NDArray[np.float64]
                The training data.
            **kwargs: Any
                - min_instances_slice: int (default 200)
                - n_clusters: int (default 2)

        Returns:
            SpnTPM: The trained instance.
        """
        # TODO: put these 2 as standard parameters - so that they show up in docs
        min_instances_slice = kwargs.get("min_instances_slice", 200)
        n_clusters = kwargs.get("n_clusters", 2)

        self.model = SPN(
            data,
            self.data_handler,
            normalize_data=False,
            learn_mspn_kwargs={
                "min_instances_slice": min_instances_slice,
                "n_clusters": n_clusters,
            },
        )
        return self

    def encode(
        self,
        model_block: pyo.Block,
        inputs: List[
            Optional[Union[pyo.Var, float, npt.NDArray[np.float64], List[pyo.Var]]]
        ],
        solver: Optional[str] = None,
        **kwargs: Any,
    ) -> pyo.Var:
        """
        Encode the SPN into Pyomo constraints.

        Args:
            model_block: pyo.Block
                The Pyomo block.
            inputs: List[Optional[Union[pyo.Var, float, npt.NDArray[np.float64], List[pyo.Var]]]]
                The input list for features.
            solver: Optional[str]
                The solver to use. Used to adjust encoding for specific solvers like appsi_highs.
            **kwargs: Any
                - mio_epsilon: float (default 1e-4)
                - sum_approx: str (default "piecewise")
                - encoding_type_exp: str (default "SOS2")
                - encoding_type_log: str (default "SOS2")

        Returns:
            pyo.Var: Total log-probability variable.
        """
        if self.model is None or self.data_handler is None:
            raise ValueError("SPN model not trained or data_handler not set.")

        if len(inputs) != self.data_handler.n_features:
            raise ValueError(
                f"Input length mismatch: expected {self.data_handler.n_features} inputs "
                f"(based on data_handler), got {len(inputs)}"
            )

        keep_idx = []
        for i, val in enumerate(inputs):
            if val is not None:
                keep_idx.append(i)
        if len(inputs) != len(keep_idx):
            self.marginalized_model = copy.deepcopy(self.model)
            self.marginalized_model.marginalize(keep_idx)
        else:
            self.marginalized_model = self.model

        # TODO: make these standard parameters
        mio_epsilon = kwargs.get("mio_epsilon", 1e-4)
        sum_approx = kwargs.get("sum_approx", "piecewise")
        # HiGHS does not support SOS constraints, so we use BIGM_BIN instead
        if sum_approx == "piecewise" and kwargs.get("solver", "") == "appsi_highs":
            if kwargs.get("encoding_type_exp", "SOS2") == "SOS2":
                kwargs["encoding_type_exp"] = "BIGM_BIN"
            if kwargs.get("encoding_type_log", "SOS2") == "SOS2":
                kwargs["encoding_type_log"] = "BIGM_BIN"

        out_vars = encode_spn(
            self.marginalized_model,
            model_block,
            inputs,
            mio_epsilon=mio_epsilon,
            sum_approx=sum_approx,
            **kwargs,
        )
        return out_vars[self.marginalized_model.out_node_id]

    def probability(self, sample: npt.NDArray[np.float64], **kwargs: Any) -> float:
        """
        Calculate the exact log-probability.
        """
        if self.model is None or self.data_handler is None:
            raise ValueError("SPN model not trained or data_handler not set.")

        if len(sample) != self.data_handler.n_features:
            raise ValueError(
                f"Input length mismatch: expected {self.data_handler.n_features} inputs "
                f"(based on data_handler), got {len(sample)}"
            )

        keep_idx = []
        for i, val in enumerate(sample):
            if val is not None:
                keep_idx.append(i)
        if len(sample) != len(keep_idx):
            self.marginalized_model = copy.deepcopy(self.model)
            self.marginalized_model.marginalize(keep_idx)
        else:
            self.marginalized_model = self.model

        return float(self.marginalized_model.compute_ll(sample)[0])

    def probability_approx(
        self, sample: npt.NDArray[np.float64], **kwargs: Any
    ) -> float:
        """
        Calculate an approximate log-probability.

        Args:
            sample: npt.NDArray[np.float64]
                The input sample.
            **kwargs: Any
                - sum_approx: str ("piecewise" or "lower", default "piecewise")
        """
        if self.model is None or self.data_handler is None:
            raise ValueError("SPN model not trained or data_handler not set.")

        if len(sample) != self.data_handler.n_features:
            raise ValueError(
                f"Input length mismatch: expected {self.data_handler.n_features} inputs "
                f"(based on data_handler), got {len(sample)}"
            )

        keep_idx = []
        for i, val in enumerate(sample):
            if val is not None:
                keep_idx.append(i)
        if len(sample) != len(keep_idx):
            self.marginalized_model = copy.deepcopy(self.model)
            self.marginalized_model.marginalize(keep_idx)
        else:
            self.marginalized_model = self.model

        if kwargs.get("sum_approx", "piecewise") == "piecewise":
            return float(self.marginalized_model.compute_maxpw_approx(sample))
        elif kwargs.get("sum_approx", "piecewise") == "lower":
            return float(self.marginalized_model.compute_max_approx(sample))
        else:
            raise ValueError(
                f"Approximation method '{kwargs.get('sum_approx', 'piecewise')}' not implemented."
            )
