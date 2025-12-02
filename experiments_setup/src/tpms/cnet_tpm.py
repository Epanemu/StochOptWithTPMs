from typing import Any, List
import numpy as np
import pyomo.environ as pyo
from src.tpms.tpm import TPM
from src.data.DataHandler import DataHandler
from src.tpms.CNet.cnet import learn_cnet, build_cnet_milp
from src.tpms.SPN.structure.Base import Context
from src.tpms.SPN.structure.leaves.parametric.Parametric import Bernoulli

class CNetTPM(TPM):
    def __init__(self):
        self.model = None
        self.root_id = None

    def train(self, data: np.ndarray, data_handler: DataHandler, **kwargs) -> 'CNetTPM':
        min_instances_slice = kwargs.get("min_instances_slice", 200)

        # CNet needs encoded data
        enc_data = data_handler.encode(data, normalize=False, one_hot=True)

        # Create Context (assuming Bernoulli for one-hot)
        var_types = Context(
            parametric_types=[Bernoulli] * enc_data.shape[1]
        ).add_domains(enc_data)

        self.model = learn_cnet(
            enc_data,
            var_types,
            cond="random",
            min_instances_slice=min_instances_slice,
            min_features_slice=1
        )
        return self

    def encode(self, model_block: pyo.Block, inputs: List[Any], **kwargs) -> Any:
        if self.model is None:
            raise ValueError("CNet model not trained.")

        cnet_outputs, root_id = build_cnet_milp(
            self.model,
            model_block,
            inputs
        )
        self.root_id = root_id
        return cnet_outputs

    @property
    def out_node_id(self) -> int:
        if self.root_id is None:
             # If encode hasn't been called, we might not know the root ID in the MILP dict context
             # But usually we need this after encoding.
             raise ValueError("CNet not encoded yet or root ID unknown.")
        return self.root_id
