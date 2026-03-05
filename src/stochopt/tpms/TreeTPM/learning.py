import logging
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from ...data.DataHandler import DataHandler
from ...data.Features import Contiguous
from .histograms import JointHistogram
from .nodes import DecisionNode, LeafNode, TreeNode

logger = logging.getLogger(__name__)


class GreedyTopDownLearner:
    """
    Learns a TreeTPM by recursively splitting the feature space in a greedy top-down
    fashion to separate dense clusters from empty or low-density regions.
    """

    def __init__(
        self,
        data_handler: DataHandler,
        min_samples: int = 10,
        max_depth: int = 10,
        val_ratio: float = 0.2,
        epsilon: float = 1e-6,
        alpha: float = 0.1,
        max_branches: int = 5,
    ):
        """
        Initialize the GreedyTopDownLearner.

        Args:
            data_handler: DataHandler
                Metadata about features.
            min_samples: int
                Minimum number of samples required to consider a split.
            max_depth: int
                Maximum depth of the resulting tree.
            val_ratio: float
                Fraction of data used for validation to evaluate split gains.
            epsilon: float
                Small value for numerical stability.
            alpha: float
                Laplace smoothing parameter.
            max_branches: int
                Maximum number of branches for a categorical split.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.val_ratio = val_ratio
        self.epsilon = epsilon
        self.data_handler = data_handler
        self.alpha = alpha
        self.max_branches = max_branches

        # Recover feature types from data_handler or data
        from stochopt.data.Features import Contiguous

        self.feature_types = []
        if self.data_handler:
            for feature in self.data_handler.features:
                self.feature_types.append(
                    "continuous" if isinstance(feature, Contiguous) else "categorical"
                )

    def learn(self, data: npt.NDArray[np.float64]) -> TreeNode:
        """
        Learn a TreeTPM structure from data.

        Args:
            data: npt.NDArray[np.float64]
                The training data of shape (N, D).

        Returns:
            TreeNode: The root node of the learned tree.
        """
        N, D = data.shape

        initial_bounds: List[Any] = []
        for i in range(D):
            feature = self.data_handler.features[i]
            if isinstance(feature, Contiguous):
                initial_bounds.append(feature.bounds)
            else:
                if hasattr(feature, "numeric_vals"):
                    initial_bounds.append(set(feature.numeric_vals))
                else:
                    initial_bounds.append(set(np.unique(data[:, i]).astype(int)))

        if N < self.min_samples * 2:
            return self._make_leaf(data, initial_bounds)

        # Split into train/val
        indices = np.random.permutation(N)
        val_size = int(N * self.val_ratio)
        train_data = data[indices[val_size:]]
        val_data = data[indices[:val_size]]

        return self._recursive_split(
            train_data, val_data, depth=0, current_bounds=initial_bounds
        )

    def _recursive_split(
        self,
        train: npt.NDArray[np.float64],
        val: npt.NDArray[np.float64],
        depth: int,
        current_bounds: List[Any],
    ) -> TreeNode:
        """
        Recursively split the data to build the tree.
        """
        N_attr = train.shape[1]

        # Stopping conditions
        if len(train) < self.min_samples or depth >= self.max_depth:
            return self._make_leaf(train, current_bounds)

        best_gain = -np.inf
        best_split = None  # (feat_idx, threshold, branches_data)

        # Calculate base LL on val data (using a single leaf)
        # TODO use the log-likelihood of the leaf when created for the splitting in the parent node
        try:
            base_leaf = self._make_leaf(train, current_bounds)
            base_ll = float(
                np.mean([base_leaf.log_inference(x) for x in val])
                if len(val) > 0
                else 0
            )
        except ValueError:
            base_ll = float("-inf")

        # Search for best split
        for i in range(N_attr):
            split_res: Optional[
                Tuple[
                    float,
                    int,
                    List[Any],
                    List[Set[int]] | List[npt.NDArray[np.float64]],
                    List[npt.NDArray[np.float64]],
                    List[float],
                ]
            ] = None
            if self.feature_types[i] == "continuous":
                split_res = self._find_best_continuous_split(
                    train, val, i, base_ll, current_bounds
                )
            else:
                split_res = self._find_best_categorical_split(
                    train, val, i, base_ll, current_bounds
                )

            if split_res and split_res[0] > best_gain:
                best_gain, best_split = split_res[0], split_res[1:]

        if best_split is None or best_gain <= 0:
            if best_gain == -np.inf:
                raise ValueError(
                    "No split found, possibly due to leaf histograms being too large."
                )
            return base_leaf

        feat_idx, bins, train_subsets, val_subsets, weights = best_split

        children = []
        for j in range(len(train_subsets)):
            # Update bounds for the branch
            new_bounds = list(current_bounds)
            new_bounds[feat_idx] = bins[j]

            child = self._recursive_split(
                train_subsets[j], val_subsets[j], depth + 1, new_bounds
            )
            children.append(child)

        split_bins: List[Set[int]] | npt.NDArray[np.float64]
        if self.feature_types[feat_idx] == "continuous":
            # bins is a list of (min, max) intervals
            # We want an array of edges: [min0, max0, max1, ...]
            # Assuming contiguous intervals from _find_best_continuous_split
            edges: List[float] = []
            for interval in bins:
                if not isinstance(interval, tuple):
                    raise ValueError("bins must be a list of tuples")
                edges.append(interval[1])
            split_bins = np.array(edges)
        else:
            # bins is a list of values or sets
            # DecisionNode expects list of sets
            bin_sets: List[Set[int]] = []
            for group in bins:
                if isinstance(group, tuple):
                    raise ValueError("bins must be a list of sets")
                bin_sets.append(set(group))
            split_bins = bin_sets

        return DecisionNode(
            feat_idx, children, self.feature_types[feat_idx], split_bins, weights
        )

    def _find_best_continuous_split(
        self,
        train: npt.NDArray[np.float64],
        val: npt.NDArray[np.float64],
        feat_idx: int,
        base_ll: float,
        current_bounds: List[Any],
    ) -> Optional[
        Tuple[
            float,
            int,
            List[Tuple[float, float]],
            List[npt.NDArray[np.float64]],
            List[npt.NDArray[np.float64]],
            List[float],
        ]
    ]:
        """
        Evaluate potential binary splits for a continuous feature and return
        the one with the highest log-likelihood gain.
        """
        current_feature_bounds = current_bounds[feat_idx]
        # Separation of clusters vs empty space:
        # Evaluate splits at potential "gaps" or just quantiles
        feat_vals = np.sort(train[:, feat_idx])

        # TODO make these candidate splits configurable, specified in the init
        candidates = np.unique(np.percentile(feat_vals, [25, 50, 75]))

        best_gain = -np.inf
        best_data = None

        for t in candidates:
            # Binary split: <= t and > t
            m_t = train[:, feat_idx] <= t
            m_v = val[:, feat_idx] <= t

            t_left, t_right = train[m_t], train[~m_t]
            v_left, v_right = val[m_v], val[~m_v]

            if (
                len(t_left) < self.min_samples / 2
                or len(t_right) < self.min_samples / 2
            ):
                continue

            # Quick LL estimation
            new_bounds_l = list(current_bounds)
            new_bounds_r = list(current_bounds)
            new_bounds_l[feat_idx] = (current_feature_bounds[0], t)
            new_bounds_r[feat_idx] = (t, current_feature_bounds[1])

            try:
                l_leaf = self._make_leaf(t_left, new_bounds_l)
                r_leaf = self._make_leaf(t_right, new_bounds_r)
            except ValueError:
                # maybe try to eval the split differently here?
                continue

            w_l = len(t_left) / len(train)
            w_r = len(t_right) / len(train)

            split_ll = 0
            if len(v_left) > 0:
                split_ll += np.sum([l_leaf.log_inference(x) for x in v_left])
            if len(v_right) > 0:
                split_ll += np.sum([r_leaf.log_inference(x) for x in v_right])

            gain = (split_ll / len(val)) - base_ll if len(val) > 0 else 0

            if gain > best_gain:
                best_gain = gain
                # Thresholds for DecisionNode compatibility:
                # Our DecisionNode._match needs boolean logic.
                # Let's use a trick: for continuous we'll store a custom 'Interval' object or just set range.
                # Actually, let's keep it simple: use a lambda or list for now.
                # User request: "splitting to a given branch on more than a single value"
                # For continuous, we can use (min, max) overlap.

                # We'll use a range-based condition
                # Use current_bounds which tracks the window of the feature in this branch
                c_min, c_max = current_feature_bounds
                c_left = (c_min, t)
                c_right = (t, c_max)
                best_data = (
                    feat_idx,
                    [c_left, c_right],
                    [t_left, t_right],
                    [v_left, v_right],
                    [w_l, w_r],
                )

        return (best_gain, *best_data) if best_data else None

    def _find_best_categorical_split(
        self,
        train: npt.NDArray[np.float64],
        val: npt.NDArray[np.float64],
        feat_idx: int,
        base_ll: float,
        current_bounds: List[Any],
    ) -> Optional[
        Tuple[
            float,
            int,
            List[Set[int]],
            List[npt.NDArray[np.float64]],
            List[npt.NDArray[np.float64]],
            List[float],
        ]
    ]:
        """
        Evaluate a multi-branch split for a categorical feature based on
        unique values, grouping them if they exceed max_branches.
        """
        current_feature_bounds = current_bounds[feat_idx]
        # Only consider splitting on values that are within the current branch's bounds
        unique_vals, counts = np.unique(train[:, feat_idx], return_counts=True)
        allowed_mask = np.array([v in current_feature_bounds for v in unique_vals])

        allowed_vals = unique_vals[allowed_mask]
        allowed_counts = counts[allowed_mask]

        if len(allowed_vals) < 2:
            return None

        # Determine groupings
        if len(allowed_vals) <= self.max_branches:
            branches = [{v} for v in allowed_vals]
            branch_counts = allowed_counts
        else:
            # Group rarest categories together
            branches = [set() for _ in range(self.max_branches)]
            branch_counts = np.zeros(self.max_branches)

            # Sort by frequency descending
            sort_idx = np.argsort(-allowed_counts)
            sorted_vals = allowed_vals[sort_idx]
            sorted_counts = allowed_counts[sort_idx]

            for v, c in zip(sorted_vals, sorted_counts):
                smallest_idx = np.argmin(branch_counts)
                branches[smallest_idx].add(v)
                branch_counts[smallest_idx] += c

        # Prepare subsets
        train_subsets = []
        val_subsets = []
        weights = []

        for group in branches:
            m_t = np.isin(train[:, feat_idx], list(group))
            m_v = np.isin(val[:, feat_idx], list(group))

            train_subsets.append(train[m_t])
            val_subsets.append(val[m_v])
            weights.append(len(train[m_t]) / len(train))

        # Evaluate split
        split_ll = 0
        valid_val = 0
        for j in range(len(branches)):
            if len(val_subsets[j]) == 0:
                continue

            new_bounds = list(current_bounds)
            new_bounds[feat_idx] = branches[j]
            try:
                leaf = self._make_leaf(train_subsets[j], new_bounds)
            except ValueError:
                return None
            split_ll += np.sum([leaf.log_inference(x) for x in val_subsets[j]])
            valid_val += len(val_subsets[j])

        gain = (split_ll / valid_val) - base_ll if valid_val > 0 else 0

        if gain > 0:
            return (gain, feat_idx, branches, train_subsets, val_subsets, weights)
        return None

    def _make_leaf(
        self, data: npt.NDArray[np.float64], current_bounds: List[Any]
    ) -> LeafNode:
        """
        Create a LeafNode with a JointHistogram for all features in the data,
        using the provided bounds to ensure all valid values have non-zero probability.
        """
        n_samples, n_vars = data.shape
        scope = list(range(n_vars))
        if n_vars == 0:
            return LeafNode([], JointHistogram([], {}, {}, np.array(0.0), []))

        # Determine bins per variable
        if n_vars > 8:
            n_bins_per_var = 2
        elif n_vars > 3:
            n_bins_per_var = 3
        else:
            n_bins_per_var = 5

        edges_dict = {}
        bins_dict = {}
        shape = []
        for i in range(n_vars):
            v = data[:, i]
            b_info = current_bounds[i]

            if self.feature_types[i] == "continuous":
                # Use percentiles of local data for inner edges, but keep branch bounds as outer
                c_min, c_max = b_info
                if len(v) > n_bins_per_var:
                    q = np.linspace(0, 100, n_bins_per_var + 1)
                    inner_edges = np.unique(np.percentile(v, q))
                    # Ensure they are within [c_min, c_max]
                    inner_edges = np.clip(inner_edges, c_min + 1e-6, c_max - 1e-6)
                    edges = np.unique(
                        np.concatenate(
                            [np.array([c_min]), inner_edges, np.array([c_max])]
                        )
                    )
                else:
                    edges = np.linspace(c_min, c_max, n_bins_per_var + 1)

                # Tiny margins for searchsorted
                edges[0] -= 1e-7
                edges[-1] += 1e-7
                edges_dict[i] = edges
                shape.append(len(edges) - 1)
            else:
                # Categorical: use b_info (set of allowed values)
                allowed_vals = sorted(list(b_info))
                unique_vals, counts = np.unique(v, return_counts=True)
                val_counts = dict(zip(unique_vals, counts))

                if len(allowed_vals) <= n_bins_per_var:
                    groups = [{val} for val in allowed_vals]
                else:
                    # Balanced frequency grouping of ALL allowed values
                    groups = [set() for _ in range(n_bins_per_var)]
                    group_counts = np.zeros(n_bins_per_var)

                    # Sort allowed values by their local frequency
                    sorted_allowed = sorted(
                        allowed_vals, key=lambda x: val_counts.get(x, 0), reverse=True
                    )

                    for val in sorted_allowed:
                        smallest_idx = np.argmin(group_counts)
                        groups[smallest_idx].add(val)
                        group_counts[smallest_idx] += val_counts.get(val, 0)

                bins_dict[i] = groups
                shape.append(len(groups))

        total_cells = int(np.prod(shape))
        if total_cells > 1_000_000:
            raise ValueError(
                f"Number of cells in joint histogram of shape {shape} is too large: {total_cells}. Try reducing n_bins_per_var."
            )

        # Compute joint counts
        counts = np.zeros(shape, dtype=np.int64)
        for x in data:
            indices = []
            for i in range(n_vars):
                val = x[i]
                if self.feature_types[i] == "continuous":
                    e = edges_dict[i]
                    idx = np.searchsorted(e, val, side="right") - 1
                    idx = np.clip(idx, 0, shape[i] - 1)
                else:
                    # Simple value lookup in groups
                    b = bins_dict[i]
                    idx = -1
                    for j, g in enumerate(b):
                        if val in g:
                            idx = j
                            break
                if idx != -1:
                    indices.append(idx)

            if len(indices) == n_vars:
                counts[tuple(indices)] += 1

        # Laplace smoothing
        probs = (counts + self.alpha) / (n_samples + self.alpha * total_cells)
        log_probs = np.log(probs + 1e-12)

        jh = JointHistogram(scope, edges_dict, bins_dict, log_probs, self.feature_types)
        return LeafNode(scope, jh)
