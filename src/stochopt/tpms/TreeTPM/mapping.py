from typing import Any, Dict, Optional, Union

import numpy as np

# Import CNet implementation for mapping
import stochopt.tpms.CNet.cnet_learning as cnet_impl

from .histograms import JointHistogram
from .nodes import DecisionNode, LeafNode, TreeNode


def cnet_to_tree(
    node: Union[cnet_impl.DecisionNode, cnet_impl.LeafNode],
    discretization_info: Optional[Dict[int, Dict[str, Any]]] = None,
) -> TreeNode:
    """
    Converts a CNet (custom implementation) to the General TreeTPM format.

    Args:
        node: Union[cnet_impl.DecisionNode, cnet_impl.LeafNode]
            The root (or sub-root) node of a CNet.
        discretization_info: dict
            Optional dictionary mapping feature indices to discretization metadata.

    Returns:
        TreeNode: The corresponding root of a TreeTPM tree.
    """
    if discretization_info is None:
        discretization_info = {}

    if isinstance(node, cnet_impl.DecisionNode):
        children = []
        weights = []
        for val, child in node.branches.items():
            children.append(cnet_to_tree(child, discretization_info))
            weights.append(node.probs[val])

        # Get split bins if available (for continuous or grouped categorical)
        split_bins = None
        feature_type = "categorical"
        if node.decision_var in discretization_info:
            split_bins = discretization_info[node.decision_var]["bins"]
            feature_type = "continuous"
        else:
            # Check if it's a known categorical with predefined bins/values
            possible_values = sorted(list(node.branches.keys()))
            split_bins = [{v} for v in possible_values]

        return DecisionNode(
            node.decision_var, children, feature_type, split_bins, weights
        )

    elif isinstance(node, cnet_impl.LeafNode):
        scope_list = node.scope
        n_vars = len(scope_list)

        # Determine domain sizes from log_factors
        domain_sizes = [lf.shape[0] for lf in node.log_factors]

        # Build the full joint probability table for the Chow-Liu tree
        joint_log_probs = np.zeros(domain_sizes)

        # Iterate through all possible combinations of values in the scope
        for idx in np.ndindex(*domain_sizes):
            log_p = 0.0
            for i in range(n_vars):
                val_i = idx[i]
                parent_local_idx = node.tree[i]

                if parent_local_idx == -1:
                    # Root variable: P(X_i)
                    log_p += node.log_factors[i][val_i, 0]
                else:
                    # Child variable: P(X_i | X_parent)
                    val_p = idx[parent_local_idx]
                    log_p += node.log_factors[i][val_i, val_p]
            joint_log_probs[idx] = log_p

        # Create bins and types
        edges = {}
        bins = {}
        feature_types = []
        for i, var_idx in enumerate(scope_list):
            if var_idx in discretization_info:
                info = discretization_info[var_idx]
                edges[var_idx] = info["bins"]
                feature_types.append("continuous")
            else:
                ds = domain_sizes[i]
                bins[var_idx] = [{v} for v in range(ds)]
                feature_types.append("categorical")

        jh = JointHistogram(scope_list, edges, bins, joint_log_probs, feature_types)
        return LeafNode(scope_list, jh)

    else:
        raise ValueError(f"Unsupported node type for conversion: {type(node)}")
