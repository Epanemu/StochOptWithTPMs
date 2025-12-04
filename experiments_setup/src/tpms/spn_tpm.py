from typing import Any, List
import numpy as np
import pyomo.environ as pyo
from src.tpms.tpm import TPM
from src.data.DataHandler import DataHandler
from src.tpms.SPN.SPN import SPN
from src.tpms.SPN.spn_enc import encode_spn
import copy

class SpnTPM(TPM):
    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, data: np.ndarray, data_handler: DataHandler, **kwargs) -> 'SpnTPM':
        # TODO: put these 2 as standard parameters - so that they show up in docs
        min_instances_slice = kwargs.get("min_instances_slice", 200)
        n_clusters = kwargs.get("n_clusters", 2)

        # Store data_handler for later use in encoding
        self.data_handler = data_handler

        self.model = SPN(
            data,
            data_handler,
            normalize_data=False,
            learn_mspn_kwargs={
                "min_instances_slice": min_instances_slice,
                "n_clusters": n_clusters
            }
        )
        return self

    def encode(self, model_block: pyo.Block, inputs: List[Any], **kwargs) -> Any:
        if self.model is None:
            raise ValueError("SPN model not trained.")

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

    @property
    def out_node_id(self) -> int:
        if self.model is None:
            raise ValueError("SPN model not trained.")
        return self.model.out_node_id
