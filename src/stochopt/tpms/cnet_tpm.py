import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Sequence

import numpy as np
import numpy.typing as npt
import pyomo.environ as pyo
from stochopt.data.DataHandler import DataHandler
from stochopt.data.Features import Contiguous, Categorical, Binary
from stochopt.tpms.CNet.cnet import build_cnet_milp, DecisionNode, LeafNode
from stochopt.tpms.CNet.cnet_learning import learn_cnet_tree
from stochopt.tpms.tpm import TPM

logger = logging.getLogger(__name__)


class CNetTPM(TPM):
    """
    CNet-based TPM with discretization support for continuous variables.

    Since CNet works only with discrete variables, continuous features are
    discretized using binning strategies before training and encoding.
    """

    def __init__(self) -> None:
        """
        Initialize the CNetTPM.
        """
        super().__init__()
        self.model: Optional[DecisionNode | LeafNode] = None
        self.root_id: Optional[int] = None
        self.discretization_info: Dict[int, Dict[str, Any]] = {}
        self.data_handler: Optional[DataHandler] = None

    # TODO check / fix this
    def train(
        self,
        data: npt.NDArray[np.float64],
        data_handler: DataHandler,
        discretization_method: Literal["uniform", "quantile", "kmeans"] = "quantile",
        n_bins: int = 10,
        **kwargs: Any,
    ) -> "CNetTPM":
        """
        Train the CNet model with automatic discretization of continuous variables.

        Args:
            data: npt.NDArray[np.float64]
                The training data.
            data_handler: DataHandler
                The data handler for feature metadata.
            discretization_method: Literal["uniform", "quantile", "kmeans"]
                Method to discretize continuous variables ("uniform", "quantile", "kmeans", default "quantile").
            n_bins: int
                Number of bins to use for discretization (default 10).
            **kwargs: Any
                Hyperparameters for the CNet learning (e.g., min_instances_slice=50, max_depth=10).

        Returns:
            CNetTPM: The trained instance.
        """
        min_instances_slice = kwargs.get("min_instances_slice", 50)
        max_depth = kwargs.get("max_depth", 10)

        # Discretize continuous features
        self.data_handler = data_handler
        discretized_data = self._discretize_data(data, discretization_method, n_bins)

        categ_map: dict[int | str, list[int | str]] = {}
        for i, feature in enumerate(data_handler.features):
            if i in self.discretization_info:
                categ_map[feature.name] = list(range(self.discretization_info[i]["n_bins"]))
            elif isinstance(feature, (Categorical, Binary)):
                categ_map[feature.name] = feature.orig_vals
            else:
                raise ValueError(f"Unsupported feature type: {type(feature)}")
        self.discrete_data_handler = DataHandler(
            discretized_data,
            categ_map=categ_map,
            feature_names=[feature.name for feature in data_handler.features],
        )

        # CNet works with categorical/discrete features
        # We use the discretized indices for training
        logger.info(f"Discretized data shape: {discretized_data.shape}")

        # Learn the CNet tree using our custom implementation
        self.model = learn_cnet_tree(
            self.discrete_data_handler,
            discretized_data.astype(np.float64),
            min_instances_slice=min_instances_slice,
            max_depth=max_depth,
        )
        return self

    def probability(self, sample: npt.NDArray[np.float64], **kwargs: Any) -> float:
        """
        Calculate the exact log-probability.

        Args:
            sample: npt.NDArray[np.float64]
                The input sample.
            **kwargs: Any
                Additional arguments.

        Returns:
            float: Log-probability value.
        """
        if self.model is None:
            return -np.inf

        # Discretize sample for inference
        d_sample = sample.copy()
        if hasattr(self, "discretization_info"):
            for feat_idx, info in self.discretization_info.items():
                bins = info["bins"]
                val = sample[feat_idx]
                bin_idx = np.digitize(val, bins) - 1
                d_sample[feat_idx] = np.clip(bin_idx, 0, info["n_bins"] - 1)

        return float(self.model.log_inference(d_sample.astype(int)))

    def probability_approx(self, sample: npt.NDArray[np.float64], **kwargs: Any) -> float:
        """
        Calculate an approximate log-probability. For CNet, this is the same as the exact.
        """
        return self.probability(sample)

    def _discretize_data(
        self,
        data: npt.NDArray[np.float64],
        method: Literal["uniform", "quantile", "kmeans"],
        n_bins: int,
    ) -> npt.NDArray[np.int64]:
        """
        Discretize continuous features in the data.

        Args:
            data: Raw data array
            method: Discretization method
            n_bins: Number of bins

        Returns:
            Discretized data array
        """
        discretized = np.empty_like(data, dtype=np.int64)
        if self.data_handler is None:
            raise ValueError("Data handler not initialized.")

        for feat_idx, feature in enumerate(self.data_handler.features):
            if isinstance(feature, Contiguous):
                # This is a continuous feature that needs discretization
                feat_data = data[:, feat_idx]

                if method == "uniform":
                    # Equal-width bins
                    bins = np.linspace(feat_data.min(), feat_data.max(), n_bins + 1)
                    bins[0] -= 1e-6  # Ensure first value is included
                    bins[-1] += 1e-6  # Ensure last value is included

                elif method == "quantile":
                    # Equal-frequency bins (quantiles)
                    bins = np.percentile(feat_data, np.linspace(0, 100, n_bins + 1))
                    bins = np.unique(bins)  # Remove duplicates
                    if len(bins) < n_bins + 1:
                        # If not enough unique quantiles, fall back to uniform
                        logger.warning(
                            f"Not enough unique quantiles for feature {feature.name}. "
                            f"Falling back to uniform discretization."
                        )
                        bins = np.linspace(feat_data.min(), feat_data.max(), n_bins + 1)
                    bins[0] -= 1e-6
                    bins[-1] += 1e-6

                elif method == "kmeans":
                    # K-means clustering for bin edges
                    from sklearn.cluster import KMeans

                    kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
                    kmeans.fit(feat_data.reshape(-1, 1))
                    centers = np.sort(kmeans.cluster_centers_.flatten())

                    # Create bins from cluster centers
                    bins = np.zeros(n_bins + 1)
                    bins[0] = feat_data.min() - 1e-6
                    bins[-1] = feat_data.max() + 1e-6
                    bins[1:-1] = (centers[:-1] + centers[1:]) / 2

                else:
                    raise ValueError(f"Unknown discretization method: {method}")

                # Discretize the feature
                bin_indices = np.digitize(feat_data, bins) - 1
                bin_indices = np.clip(bin_indices, 0, len(bins) - 2)  # Ensure valid range

                discretized[:, feat_idx] = bin_indices

                # Store discretization info for later use in encoding
                self.discretization_info[feat_idx] = {
                    "bins": bins,
                    "n_bins": len(bins) - 1,
                    "method": method,
                    "feature_name": feature.name,
                }

        return discretized

    def encode(
        self,
        model_block: pyo.Block,
        inputs: List[Optional[Union[pyo.Var, float, list[float], List[pyo.Var]]]],
        solver: str = "gurobi",
        **kwargs: Any,
    ) -> pyo.Var:
        """
        Encode the CNet into Pyomo constraints with discretization handling.

        Args:
            model_block: pyo.Block
                The Pyomo block to add variables and constraints to.
            inputs: List[Optional[Union[pyo.Var, float, list[float], List[pyo.Var]]]]
                The input list for features.
            solver: str
                The solver to consider for encoding (default "gurobi").
            **kwargs: Any
                Additional arguments.

        Returns:
            pyo.Var: Total log-probability variable.
        """
        if self.model is None:
            raise ValueError("CNet model not trained.")

        structured_inputs = self._create_discretized_inputs(model_block, inputs)

        log_prob_vars, root_id = build_cnet_milp(
            self.model, model_block, structured_inputs, solver=solver, **kwargs
        )

        root_var: pyo.Var = log_prob_vars[root_id]
        return root_var

    def _create_discretized_inputs(
        self, model_block: pyo.Block, inputs: List[Any]
    ) -> List[List[Any]]:
        """
        Create discretized/binned versions of continuous input variables.
        Returns a nested list: [feat_idx][val]
        """
        structured_inputs = []
        input_idx = 0  # Track position in inputs list

        if self.data_handler is None:
            raise ValueError("Data handler not initialized.")

        for feat_idx, feature in enumerate(self.data_handler.features):
            if inputs[input_idx] is None:
                # Marginalized variable
                if feat_idx in self.discretization_info:
                    n_bins = self.discretization_info[feat_idx]["n_bins"]
                    structured_inputs.append([None] * n_bins)
                else:
                    # Categorical/Discrete?
                    # We need to know the domain size
                    dom_size = len(feature.numeric_vals) if hasattr(feature, "numeric_vals") else 2
                    structured_inputs.append([None] * dom_size)
                input_idx += 1
                continue

            if feat_idx in self.discretization_info:
                continuous_var = inputs[input_idx]
                disc_info = self.discretization_info[feat_idx]
                bins = disc_info["bins"]
                n_bins = disc_info["n_bins"]

                bin_vars = pyo.Var(range(n_bins), domain=pyo.Binary)
                model_block.add_component(f"bin_vars_feat{feat_idx}", bin_vars)

                model_block.add_component(
                    f"one_bin_feat{feat_idx}",
                    pyo.Constraint(expr=sum(bin_vars[i] for i in range(n_bins)) == 1),
                )

                for i in range(n_bins):
                    lb, ub = bins[i], bins[i + 1]
                    M = max(abs(bins[0]), abs(bins[-1])) * 10

                    model_block.add_component(
                        f"bin_lb_feat{feat_idx}_bin{i}",
                        pyo.Constraint(expr=continuous_var >= lb - M * (1 - bin_vars[i])),
                    )
                    model_block.add_component(
                        f"bin_ub_feat{feat_idx}_bin{i}",
                        pyo.Constraint(expr=continuous_var <= ub + M * (1 - bin_vars[i])),
                    )

                structured_inputs.append([bin_vars[i] for i in range(n_bins)])

            else:
                # Already discrete or categorical
                # We assume the input here is a list of variables if it was already one-hot,
                # or a single variable if it's meant to be used as index.
                # Actually, our TPM layer usually passes a list of one-hot variables for discrete features.
                # Let's assume 'inputs[input_idx]' is already a list or we wrap it.
                val = inputs[input_idx]
                if isinstance(val, list):
                    structured_inputs.append(val)
                else:
                    # If it's a single variable, we might need to one-hot it if the CNet expects it.
                    # But CNet expects structured [feat][val].
                    # Let's assume it's already structured or provided as such.
                    structured_inputs.append([val])

            input_idx += 1

        return structured_inputs
