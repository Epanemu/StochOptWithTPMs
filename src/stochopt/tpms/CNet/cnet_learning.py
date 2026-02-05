"""
Clean CNet Learning Implementation
- OR tree structure (DecisionNode)
- Chow-Liu tree leaves (LeafNode with CLTree)
- Specialized for binary/discrete features
"""

from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
from stochopt.data.DataHandler import DataHandler


class LeafNode:
    """
    Chow-Liu tree leaf node in a CNet.
    """

    def __init__(
        self,
        scope: List[int],
        tree: List[int],
        log_factors: List[npt.NDArray[np.float64]],
    ):
        """
        Initialize the LeafNode.

        Args:
            scope: List[int]
                List of variable indices in this subtree.
            tree: List[int]
                List where `tree[i]` = parent index of variable `scope[i]` (-1 if root).
            log_factors: List[npt.NDArray[np.float64]]
                List of log-probability tables (CPDs) for each variable in scope.
        """
        self.scope: List[int] = sorted(list(scope))
        self.tree: List[int] = tree
        self.log_factors: List[npt.NDArray[np.float64]] = log_factors

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}LEAF(CLTree, scope={self.scope}, |scope|={len(self.scope)})"

    def log_inference(self, x: npt.NDArray[np.float64]) -> float:
        """
        Compute P(x) for this Chow-Liu tree.

        Args:
            x: npt.NDArray[np.float64]
                The input sample.

        Returns:
            np.float64: Probability value.
        """
        total_log_prob = 0.0

        for i in range(len(self.scope)):
            var_idx = self.scope[i]
            v_i = int(x[var_idx])
            parent_local_idx = self.tree[i]

            if parent_local_idx == -1:
                # Root variable: P(X_i)
                v_p = 0  # Dummy parent index for root lookup
            else:
                parent_var_idx = self.scope[parent_local_idx]
                v_p = int(x[parent_var_idx])

            # log_factors[i] is [val_i, parent_val]
            total_log_prob += self.log_factors[i][v_i][v_p]

        return float(total_log_prob)


class DecisionNode:
    """
    OR tree decision node that splits on a single variable.
    """

    def __init__(
        self,
        scope: List[int],
        decision_var: int,
        probs: Dict[int, float],
        branches: Dict[int, Any],
    ):
        """
        Initialize the DecisionNode.

        Args:
            scope: List[int]
                List of all variable indices in this subtree.
            decision_var: int
                Variable index to split on.
            probs: Dict[int, float]
                Mapping from value to its corresponding probability P(decision_var=value).
            branches: Dict[int, Any]
                Mapping from value to the child node (DecisionNode or LeafNode).
        """
        self.scope: List[int] = sorted(list(scope))
        self.decision_var: int = decision_var
        self.probs: Dict[int, float] = probs
        self.branches: Dict[int, Any] = branches

    def __repr__(self, level: int = 0) -> str:
        indent = "  " * level
        s = f"{indent}DECISION on var[{self.decision_var}]"
        for val in sorted(self.probs.keys()):
            prob = self.probs[val]
            s += f"\n{indent}├─ val={val} (P={prob:.3f}):"
            s += f"\n{self.branches[val].__repr__(level + 1)}"
        return s

    def log_inference(self, x: npt.NDArray[np.float64]) -> float:
        """
        Compute P(x) by selecting the appropriate branch and multiplying its
        probability.
        """
        val = int(x[self.decision_var])
        if val in self.branches:
            return float(np.log(self.probs[val]) + self.branches[val].log_inference(x))
        else:
            # Handle unseen values by returning 0 (strict) or smoothing
            return float(-np.inf)


def compute_mutual_information(
    data_handler: DataHandler, data: npt.NDArray[np.float64], u_idx: int, v_idx: int
) -> float:
    """
    Compute mutual information I(U;V) between two variables.
    """
    u_data = data[:, u_idx].astype(int)
    v_data = data[:, v_idx].astype(int)

    # Determine domain sizes from DataHandler
    range_u = int(data_handler.features[u_idx].n_values)
    range_v = int(data_handler.features[v_idx].n_values)

    # Count joint occurrences
    counts = np.zeros((range_u, range_v))
    np.add.at(counts, (u_data, v_data), 1)

    n_samples = len(data)
    probs_joint = counts / n_samples

    # Marginals
    probs_u = probs_joint.sum(axis=1)
    probs_v = probs_joint.sum(axis=0)

    # MI = sum P(u,v) log(P(u,v) / (P(u)P(v)))
    mask = probs_joint > 0
    probs_indep = np.outer(probs_u, probs_v)

    mi = np.sum(
        probs_joint[mask]
        * np.log((probs_joint[mask] + 1e-10) / (probs_indep[mask] + 1e-10))
    )
    return float(max(0.0, mi))


def learn_chow_liu_tree(
    data_handler: DataHandler,
    data: npt.NDArray[np.float64],
    scope: List[int],
    alpha: float = 0.1,
) -> LeafNode:
    """
    Learn a Chow-Liu tree (Maximum Spanning Tree of Mutual Information)
    on the specified scope.
    """
    scope_list = sorted(list(scope))
    n_vars = len(scope_list)
    n_samples = len(data)

    if n_vars == 0:
        return LeafNode(scope=[], tree=[], log_factors=[])

    # Domain sizes from DataHandler
    domain_sizes = [int(data_handler.features[i].n_values) for i in scope_list]

    if n_vars == 1:
        # Single variable - marginal distribution
        var_idx = scope_list[0]
        dom_size = domain_sizes[0]
        probs = np.full(dom_size, alpha / (n_samples + alpha * dom_size))
        if n_samples > 0:
            vals, counts = np.unique(data[:, var_idx], return_counts=True)
            for v, c in zip(vals, counts):
                v_int = int(v)
                if v_int < dom_size:
                    probs[v_int] = (c + alpha) / (n_samples + alpha * dom_size)

        lf = np.log(probs).reshape(-1, 1)
        return LeafNode(scope=scope_list, tree=[-1], log_factors=[lf])

    # Build MI graph
    G = nx.Graph()
    for i in range(n_vars):
        G.add_node(i)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            mi = compute_mutual_information(
                data_handler, data, scope_list[i], scope_list[j]
            )
            G.add_edge(
                i, j, weight=-mi
            )  # Minimum spanning tree treats weights as distance

    # Minimum spanning tree (with negative weights = Maximum Spanning Tree)
    mst = nx.minimum_spanning_tree(G)

    # Orient edges from root 0
    bfs_edges = list(nx.bfs_edges(mst, 0))
    tree = [-1] * n_vars
    for parent, child in bfs_edges:
        tree[child] = parent

    # Learn parameters
    log_factors = []
    for i in range(n_vars):
        var_idx = scope_list[i]
        dom_size = int(domain_sizes[i])
        parent_local = tree[i]

        if parent_local == -1:
            # Root: log P(X_i)
            probs = np.full(dom_size, alpha / (n_samples + alpha * dom_size))
            if n_samples > 0:
                vals, counts = np.unique(data[:, var_idx], return_counts=True)
                for v, c in zip(vals, counts):
                    v_int = int(v)
                    if v_int < dom_size:
                        probs[v_int] = (c + alpha) / (n_samples + alpha * dom_size)

            lf = np.log(probs).reshape(-1, 1)
        else:
            # Child: log P(X_i | X_parent)
            parent_idx = scope_list[parent_local]
            parent_dom_size = int(domain_sizes[parent_local])
            lf = np.zeros((dom_size, parent_dom_size))

            pair_data = data[:, [var_idx, parent_idx]].astype(int)
            counts_joint = np.zeros((dom_size, parent_dom_size))
            # Ensure we only use values within bounds
            mask = (pair_data[:, 0] < dom_size) & (pair_data[:, 1] < parent_dom_size)
            np.add.at(counts_joint, (pair_data[mask, 0], pair_data[mask, 1]), 1)

            # Laplace smoothing
            counts_joint += alpha
            counts_parent = counts_joint.sum(axis=0)

            probs_cond = counts_joint / counts_parent[None, :]
            lf = np.log(probs_cond)

        log_factors.append(lf)

    return LeafNode(scope=scope_list, tree=tree, log_factors=log_factors)


def compute_entropy(data, var_idx):
    """Compute entropy H(X) in bits."""
    vals, counts = np.unique(data[:, var_idx], return_counts=True)
    probs = counts / len(data)
    return -np.sum(probs * np.log2(probs + 1e-10))


def get_best_split_variable(data, scope):
    """
    Select best variable to split on based on maximum Information Gain
    (averaged across other variables).
    """
    n_samples = data.shape[0]
    n_vars = len(scope)

    if n_vars <= 1:
        return -1

    best_gain = -1.0
    best_var = -1

    # Base entropy sum
    base_entropies = [compute_entropy(data, v) for v in scope]
    total_entropy_base = sum(base_entropies)

    for i, var_idx in enumerate(scope):
        vals, counts = np.unique(data[:, var_idx], return_counts=True)
        if len(vals) < 2:
            continue

        remaining_scope = [v for v in scope if v != var_idx]

        # Conditional entropy: Sum P(v) * H(Rest | v)
        cond_entropy = 0.0
        for val, count in zip(vals, counts):
            subset = data[data[:, var_idx] == val]
            w = count / n_samples
            # Average entropy of rest given this value
            subset_entropy = sum(compute_entropy(subset, v) for v in remaining_scope)
            cond_entropy += w * subset_entropy

        # Target entropy for comparison (excluding the split var itself)
        target_entropy = total_entropy_base - base_entropies[i]
        gain = target_entropy - cond_entropy

        if gain > best_gain:
            best_gain = gain
            best_var = var_idx

    return best_var


def learn_cnet_tree(
    data_handler: DataHandler,
    data: npt.NDArray[np.float64],
    scope: Optional[List[int]] = None,
    min_instances_slice: int = 20,
    max_depth: int = 10,
    depth: int = 0,
    alpha: float = 0.1,
) -> DecisionNode | LeafNode:
    """
    Recursively learn a Conditional Network (CNet) structure.

    Args:
        data_handler: DataHandler
            The data handler for feature metadata.
        data: npt.NDArray[np.float64]
            The input data.
        scope: Optional[List[int]]
            Indices of variables to consider.
        min_instances_slice: int
            Minimum samples to split (default 20).
        max_depth: int
            Maximum recursion depth (default 10).
        depth: int
            Current depth.
        alpha: float
            Smoothing parameter (default 0.1).

    Returns:
        Any: The root node of the learned CNet (DecisionNode or LeafNode).
    """
    n_samples, n_features = data.shape
    if scope is None:
        scope = list(range(n_features))

    # Base cases
    if n_samples < min_instances_slice or len(scope) <= 1 or depth >= max_depth:
        return learn_chow_liu_tree(data_handler, data, scope, alpha=alpha)

    # Find split var
    split_var = get_best_split_variable(data, scope)
    if split_var == -1:
        return learn_chow_liu_tree(data_handler, data, scope, alpha=alpha)

    # Get domain size for the split variable
    split_var_domain_size = data_handler.features[split_var].n_values

    # Branching
    # We should iterate over ALL possible values of split_var to ensure full coverage
    possible_values = range(split_var_domain_size)
    probs_dict = {}
    branches_dict: Dict[int, DecisionNode | LeafNode] = {}
    new_scope = [v for v in scope if v != split_var]

    for val in possible_values:
        mask = data[:, split_var] == val
        count = np.sum(mask)
        subset = data[mask]

        # Proper probability with smoothing
        probs_dict[val] = (count + alpha) / (n_samples + alpha * split_var_domain_size)

        if count == 0:
            # If no samples, use the parent data (current node's data) as a fallback
            # for the Chow-Liu tree structure in this branch.
            branches_dict[val] = learn_chow_liu_tree(
                data_handler, data, new_scope, alpha=alpha
            )
        else:
            branches_dict[val] = learn_cnet_tree(
                data_handler,
                subset,
                new_scope,
                min_instances_slice,
                max_depth,
                depth + 1,
                alpha=alpha,
            )

    return DecisionNode(
        scope=scope, decision_var=split_var, probs=probs_dict, branches=branches_dict
    )


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing CNet Implementation...")
    np.random.seed(42)
    N = 1000
    # X0 -> X1, X2
    x0 = np.random.randint(0, 2, N).astype(np.float64)
    x1 = np.array(
        [np.random.choice([0, 1], p=[0.8, 0.2] if v == 0 else [0.2, 0.8]) for v in x0],
        dtype=np.float64,
    )
    x2 = np.array(
        [np.random.choice([0, 1], p=[0.7, 0.3] if v == 0 else [0.3, 0.7]) for v in x0],
        dtype=np.float64,
    )
    # X3 independent
    x3 = np.random.randint(0, 3, N).astype(np.float64)

    data = np.stack([x0, x1, x2, x3], axis=1, dtype=np.float64)
    data_handler = DataHandler(
        data, categ_map={0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1, 2]}
    )

    cnet = learn_cnet_tree(data_handler, data, min_instances_slice=50)
    print("\nLearned Structure:")
    print(cnet)

    # Inference test
    test_x = np.array([0, 0, 0, 0])
    prob = cnet.log_inference(test_x)
    print(f"\nInference P({test_x}) = {prob:.6f}")
