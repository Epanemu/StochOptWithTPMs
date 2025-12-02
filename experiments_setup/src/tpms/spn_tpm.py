from typing import Any, List
import numpy as np
import pyomo.environ as pyo
from src.tpms.tpm import TPM
from src.data.DataHandler import DataHandler
from src.tpms.SPN.SPN import SPN
from src.tpms.SPN.spn_enc import encode_spn

class SpnTPM(TPM):
    def __init__(self):
        self.model = None

    def train(self, data: np.ndarray, data_handler: DataHandler, **kwargs) -> 'SpnTPM':
        min_instances_slice = kwargs.get("min_instances_slice", 200)
        n_clusters = kwargs.get("n_clusters", 2)

        self.model = SPN(
            data,
            data_handler,
            normalize_data=False,
            learn_mspn_kwargs={
                "min_instances_slice": min_instances_slice,
                "n_clusters": n_clusters
            }
        )
        # TODO marginalize out xi
        return self

    def encode(self, model_block: pyo.Block, inputs: List[Any], **kwargs) -> Any:
        if self.model is None:
            raise ValueError("SPN model not trained.")

        mio_epsilon = kwargs.get("mio_epsilon", 1e-4)
        sum_approx = kwargs.get("sum_approx", "piecewise")

        return encode_spn(
            self.model,
            model_block,
            inputs,
            mio_epsilon=mio_epsilon,
            sum_approx=sum_approx
        )

    @property
    def out_node_id(self) -> int:
        if self.model is None:
            raise ValueError("SPN model not trained.")
        return self.model.out_node_id
