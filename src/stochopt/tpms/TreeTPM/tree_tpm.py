import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pyomo.environ as pyo

from stochopt.data.DataHandler import DataHandler
from stochopt.tpms.tpm import TPM

from .base import MIN_LOG_PROB
from .histograms import JointHistogram
from .nodes import DecisionNode, LeafNode, TreeNode

logger = logging.getLogger(__name__)


class TreeTPM(TPM):
    """
    Tractable Probabilistic Model based on a Decision Tree structure with
    histograms at the leaves.

    This model supports both categorical and continuous features and allows
    for exact marginalization and MILP encoding.
    """

    def __init__(self, data_handler: DataHandler, root: Optional[TreeNode] = None):
        """
        Initialize the TreeTPM.

        Args:
            root: Optional[TreeNode]
                The root node of the tree. If None, the tree must be trained
                before use.
        """
        super().__init__(data_handler)
        self.root = root
        self.marginalized_root: Optional[TreeNode] = None
        self.marginalized_keep_indices: Optional[List[int]] = None

    def train(self, data: npt.NDArray[np.float64], **kwargs: Any) -> "TreeTPM":
        """
        Train the TreeTPM by first learning a CNet and then converting it to a
        tree structure.

        Args:
            data: npt.NDArray[np.float64]
                The training data.
            **kwargs: Any
                - min_instances_slice: int (default 50)
                - max_depth: int (default 10)
                - n_bins: int (default 5)

        Returns:
            TreeTPM: The trained instance.
        """
        from stochopt.tpms.cnet_tpm import CNetTPM

        from .mapping import cnet_to_tree

        min_instances = kwargs.get("min_instances_slice", 50)
        max_depth = kwargs.get("max_depth", 10)
        n_bins = kwargs.get("n_bins", 5)

        logger.info("Discretizing data for CNet learning...")
        # Use CNetTPM's discretization logic or reproduce it here
        # For simplicity and consistency, we'll use a temporary CNetTPM-like discretization
        # but TreeTPMs root will eventually hold JointHistograms which can handle original scale.
        # However, the tree structure itself is learned on discretized data.

        # We'll use a copy of CNetTPM's logic
        cnet_temp = CNetTPM(self.data_handler)
        cnet_temp.train(
            data,
            n_bins=n_bins,
            min_instances_slice=min_instances,
            max_depth=max_depth,
        )

        if cnet_temp.model is None:
            raise ValueError("Failed to train CNet temporary model")
        self.root = cnet_to_tree(
            cnet_temp.model, getattr(cnet_temp, "discretization_info", {})
        )
        self.marginalized_root = None
        self.marginalized_keep_indices = None
        return self

    def train_greedy_top_down(
        self, data: npt.NDArray[np.float64], **kwargs: Any
    ) -> "TreeTPM":
        """
        Train the TreeTPM using a greedy top-down splitting algorithm.

        Args:
            data: npt.NDArray[np.float64]
                The training data.
            **kwargs: Any
                - min_samples: int (default 10)
                - max_depth: int (default 10)
                - val_ratio: float (default 0.2)
                - alpha: float (default 0.1)
                - max_branches: int (default 5)

        Returns:
            TreeTPM: The trained instance.
        """
        from .learning import GreedyTopDownLearner

        learner = GreedyTopDownLearner(
            min_samples=kwargs.get("min_samples", 10),
            max_depth=kwargs.get("max_depth", 10),
            val_ratio=kwargs.get("val_ratio", 0.2),
            data_handler=self.data_handler,
            alpha=kwargs.get("alpha", 0.1),
            max_branches=kwargs.get("max_branches", 5),
        )

        logger.info("Training TreeTPM using Greedy Top-Down Learner...")
        self.root = learner.learn(data)
        self.marginalized_root = None
        self.marginalized_keep_indices = None
        return self

    def log_probability(self, sample: npt.NDArray[np.float64], **kwargs: Any) -> float:
        """
        Calculate the log-probability of a given sample.

        Args:
            sample: npt.NDArray[np.float64]
                The input sample.
            **kwargs: Any
                Additional arguments (ignored).

        Returns:
            float: Log-probability value.
        """
        if self.root is None:
            return float(-np.inf)

        keep_indices = [i for i, val in enumerate(sample) if val is not None]
        if len(keep_indices) < len(sample):
            if (
                self.marginalized_root is not None
                and self.marginalized_keep_indices == keep_indices
            ):
                return self.marginalized_root.log_inference(sample)
            else:
                raise ValueError(
                    "Marginalized root not found or it marginalizes other variables"
                )
        return self.root.log_inference(sample)

    def marginalize(self, var_indices_to_keep: List[int]) -> "TreeTPM":
        """
        Create a new TreeTPM representing the marginal log-probability
        over the specified variable indices.

        Args:
            var_indices_to_keep: List[int]
                The indices of the variables to keep.

        Returns:
            TreeTPM: A new marginalized TreeTPM instance.
        """
        if self.root is None:
            return TreeTPM(self.data_handler)
        new_root = self.root.marginalize(set(var_indices_to_keep))
        m_tree = TreeTPM(self.data_handler, new_root)
        return m_tree

    def encode(
        self,
        model_block: pyo.Block,
        inputs: List[
            Optional[Union[pyo.Var, float, npt.NDArray[np.float64], List[pyo.Var]]]
        ],
        solver: str = "appsi_highs",
        **kwargs: Any,
    ) -> pyo.Var:
        """
        Encodes the TreeTPM into Pyomo constraints within a given block.

        Args:
            model_block: pyo.Block
                The Pyomo block to add variables and constraints to.
            inputs: List[Optional[Union[pyo.Var, float, npt.NDArray[np.float64], List[pyo.Var]]]]
                The inputs for each feature. None indicates marginalization.
            solver: str
                The solver to consider for encoding (default "appsi_highs").
            **kwargs: Any
                Additional arguments.

        Returns:
            pyo.Var: The Pyomo variable representing the total log-probability.

        Raises:
            ValueError: If the tree is not trained or input length mismatches.
        """
        if self.root is None:
            raise ValueError("TreeTPM has no root node trained.")

        if (
            self.data_handler is not None
            and len(inputs) != self.data_handler.n_features
        ):
            raise ValueError(
                f"Input length mismatch: expected {self.data_handler.n_features} inputs, got {len(inputs)}"
            )

        # 1. Automated Marginalization
        keep_indices = [i for i, val in enumerate(inputs) if val is not None]
        if len(keep_indices) < len(inputs):
            if (
                self.marginalized_root is not None
                and self.marginalized_keep_indices == keep_indices
            ):
                model_to_encode = self.marginalized_root
            else:
                logger.info(
                    f"TreeTPM: Auto-marginalizing variables for encoding. Keeping {keep_indices}."
                )
                model_to_encode = self.root.marginalize(set(keep_indices))
                # cache the marginalized tree
                self.marginalized_root = model_to_encode
                self.marginalized_keep_indices = keep_indices
        else:
            model_to_encode = self.root

        # 2. Collect unique nodes, assign them ids as an attribute
        all_nodes: List[TreeNode] = []
        node_map: Dict[
            int, int
        ] = {}  # Use a local map instead of attributes to avoid persistence issues

        def _collect(n):
            if id(n) in node_map:
                return
            node_map[id(n)] = len(all_nodes)
            n.node_id = node_map[
                id(n)
            ]  # Assign for convenience but don't rely on it for "seen" check
            all_nodes.append(n)
            if isinstance(n, DecisionNode):
                for child in n.children:
                    _collect(child)

        _collect(model_to_encode)
        node_ids = [n.node_id for n in all_nodes]
        model_block.log_prob = pyo.Var(node_ids, bounds=(-1000, 10))

        # 3. Add recursive constraints
        self._add_node_constraints(model_block, model_to_encode, inputs)

        return model_block.log_prob[model_to_encode.node_id]

    def _add_node_constraints(
        self, model_block: pyo.Block, node: TreeNode, inputs: List[Any]
    ) -> None:
        """
        Helper method to recursively add constraints for each tree node.

        Args:
            model_block: pyo.Block
                The Pyomo block.
            node: TreeNode
                The current node being encoded.
            inputs: List[Any]
                The list of inputs for features.
        """
        node_id = node.node_id
        M = 1000.0

        if isinstance(node, LeafNode):
            h = node.histogram
            term = pyo.Var(bounds=(-1000, 10))
            model_block.add_component(f"node_{node_id}_h", term)

            if isinstance(h, JointHistogram):
                # Multi-variate encoding (covers univariate cases too)
                shape = h.log_probs.shape

                # Create joint indicators for each cell
                joint_indices = list(np.ndindex(shape))
                joint_inds = pyo.Var(joint_indices, domain=pyo.Binary)
                model_block.add_component(f"node_{node_id}_h_joint", joint_inds)
                model_block.add_component(
                    f"node_{node_id}_h_one_joint",
                    pyo.Constraint(
                        expr=sum(joint_inds[idx] for idx in joint_indices) == 1
                    ),
                )

                # Compute log-prob: sum over cells
                # We subtract the density correction (log-widths and log-sizes) to match log_inference
                def _get_corrected_cell_lp(idx):
                    lp = h.log_probs[idx]
                    if lp <= MIN_LOG_PROB:
                        return MIN_LOG_PROB
                    corr = 0
                    for i, b_idx in enumerate(idx):
                        if h.feature_types[i] == "continuous":
                            corr += np.log(
                                max(1e-12, h.bins[i][b_idx + 1] - h.bins[i][b_idx])
                            )
                        else:
                            corr += np.log(len(h.bins[i][b_idx]))
                    return lp - corr

                expr = sum(
                    joint_inds[idx] * _get_corrected_cell_lp(idx)
                    for idx in joint_indices
                )
                model_block.add_component(
                    f"node_{node_id}_h_lp", pyo.Constraint(expr=term == expr)
                )

                # Link variables to joint indicators
                for i, var_idx in enumerate(h.scope):
                    var_inputs = inputs[var_idx]
                    n_bins = shape[i]

                    # Univariate indicators for this variable
                    v_inds = pyo.Var(range(n_bins), domain=pyo.Binary)
                    model_block.add_component(
                        f"node_{node_id}_h_v{var_idx}_inds", v_inds
                    )
                    model_block.add_component(
                        f"node_{node_id}_h_v{var_idx}_sum",
                        pyo.Constraint(expr=sum(v_inds[b] for b in range(n_bins)) == 1),
                    )

                    # Link joint indicator to univariate indicator: joint_ind[idx] <= v_ind[idx_i]
                    for idx in joint_indices:
                        model_block.add_component(
                            f"node_{node_id}_h_v{var_idx}_link_{idx}",
                            pyo.Constraint(expr=joint_inds[idx] <= v_inds[idx[i]]),
                        )

                    # Link v_inds to real inputs
                    if h.feature_types[i] == "categorical":

                        def _to_int(val):
                            if isinstance(val, (np.integer, np.floating)):
                                return int(val.item())
                            return int(val)

                        if not isinstance(var_inputs, (list, np.ndarray, pyo.Var)):
                            # Scalar value input: fix the indicator
                            val_idx = -1
                            try:
                                v_val = _to_int(var_inputs)
                                for j, group in enumerate(h.bins[i]):
                                    if v_val in group:
                                        val_idx = j
                                        break
                            except (ValueError, TypeError):
                                # Fallback for Pyomo variables used as discrete values
                                for j, group in enumerate(h.bins[i]):
                                    model_block.add_component(
                                        f"node_{node_id}_h_v{var_idx}_b{j}_link",
                                        pyo.Constraint(
                                            expr=v_inds[j]
                                            == sum(
                                                1.0 if v == var_inputs else 0.0
                                                for v in group
                                            )
                                        ),
                                    )
                                continue

                            if val_idx != -1:
                                model_block.add_component(
                                    f"node_{node_id}_h_v{var_idx}_fixed",
                                    pyo.Constraint(expr=v_inds[val_idx] == 1),
                                )
                        else:
                            # One-hot vector inputs
                            for j, group in enumerate(h.bins[i]):
                                model_block.add_component(
                                    f"node_{node_id}_h_v{var_idx}_b{j}_cat",
                                    pyo.Constraint(
                                        expr=v_inds[j]
                                        == sum(
                                            var_inputs[_to_int(v)]
                                            for v in group
                                            if _to_int(v) < len(var_inputs)
                                        )
                                    ),
                                )

                    else:  # Continuous
                        x_var = var_inputs
                        edges = h.bins[i]
                        for b in range(n_bins):
                            lb, ub = edges[b], edges[b + 1]
                            # TODO add proper tighter M - lb - cmin and cmax - ub
                            model_block.add_component(
                                f"node_{node_id}_h_v{var_idx}_b{b}_lb",
                                pyo.Constraint(expr=x_var >= lb - M * (1 - v_inds[b])),
                            )
                            model_block.add_component(
                                f"node_{node_id}_h_v{var_idx}_b{b}_ub",
                                pyo.Constraint(expr=x_var <= ub + M * (1 - v_inds[b])),
                            )

            model_block.add_component(
                f"node_{node_id}_lp_final",
                pyo.Constraint(expr=model_block.log_prob[node_id] == term),
            )

        elif isinstance(node, DecisionNode):
            d_var = node.split_var
            var_inputs = inputs[d_var]

            n_branches = len(node.children)
            indicators = pyo.Var(range(n_branches), domain=pyo.Binary)
            model_block.add_component(f"node_{node_id}_inds", indicators)

            # DecisionNodes partition the space, so exactly one branch must be active
            model_block.add_component(
                f"node_{node_id}_one_branch",
                pyo.Constraint(expr=sum(indicators[i] for i in range(n_branches)) == 1),
            )

            for i, child in enumerate(node.children):
                child_id = child.node_id
                log_w = np.log(max(1e-12, node.weights[i]))
                ind = indicators[i]

                # Link indicator to variable condition
                if node.feature_type == "continuous":
                    lb = node.split_bins[i]
                    ub = node.split_bins[i + 1]
                    # TODO again M can be computed as above
                    model_block.add_component(
                        f"node_{node_id}_b{i}_lb",
                        pyo.Constraint(expr=var_inputs >= lb - M * (1 - ind)),
                    )
                    model_block.add_component(
                        f"node_{node_id}_b{i}_ub",
                        pyo.Constraint(expr=var_inputs <= ub + M * (1 - ind)),
                    )

                else:  # Categorical
                    group = node.split_bins[i]

                    if isinstance(var_inputs, (list, np.ndarray)) or (
                        hasattr(var_inputs, "is_indexed") and var_inputs.is_indexed()
                    ):
                        model_block.add_component(
                            f"node_{node_id}_b{i}_cat",
                            pyo.Constraint(
                                expr=ind
                                == sum(
                                    var_inputs[v] for v in group if v < len(var_inputs)
                                )
                            ),
                        )
                    elif hasattr(var_inputs, "is_expression_type") or isinstance(
                        var_inputs, pyo.Var
                    ):
                        # Pyomo scalar var vs. categorical group (binary indicators usually)
                        # TOOD not sure if the below is correct, it assumes var_inputs is a scalar, but it is easier to just assume one-hot encoding for categorical variables
                        model_block.add_component(
                            f"node_{node_id}_b{i}_v_link",
                            pyo.Constraint(
                                expr=ind
                                == sum(1.0 if v == var_inputs else 0.0 for v in group)
                            ),
                        )
                    else:
                        matches = 1.0 if (var_inputs in group) else 0.0
                        model_block.add_component(
                            f"node_{node_id}_b{i}_const",
                            pyo.Constraint(expr=ind == matches),
                        )

                # Log-prob coupling: log_prob[node] = log_w + log_prob[child] IF ind == 1
                # TODO again M can be computed better, I think
                model_block.add_component(
                    f"node_{node_id}_b{i}_lp_ub",
                    pyo.Constraint(
                        expr=model_block.log_prob[node_id]
                        <= log_w + model_block.log_prob[child_id] + M * (1 - ind)
                    ),
                )
                model_block.add_component(
                    f"node_{node_id}_b{i}_lp_lb",
                    pyo.Constraint(
                        expr=model_block.log_prob[node_id]
                        >= log_w + model_block.log_prob[child_id] - M * (1 - ind)
                    ),
                )

                self._add_node_constraints(model_block, child, inputs)

    def log_probability_approx(
        self, sample: npt.NDArray[np.float64], **kwargs: Any
    ) -> float:
        """
        Calculate an approximate log-probability. For TreeTPM, this currently
        just calls the exact log-probability.

        Args:
            sample: npt.NDArray[np.float64]
                The input sample.
            **kwargs: Any
                Additional arguments.

        Returns:
            float: Log-probability value.
        """
        return self.log_probability(sample, **kwargs)
