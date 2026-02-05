from abc import ABC, abstractmethod
from typing import List, Optional, Set, Union, cast

import numpy as np
import numpy.typing as npt

from .base import MIN_LOG_PROB
from .histograms import Histogram, JointHistogram


class TreeNode(ABC):
    """
    Abstract base class for a node in the TreeTPM.
    """

    node_id: int

    @abstractmethod
    def log_inference(self, x: npt.NDArray[np.float64]) -> float:
        """
        Calculate the log-probability of a sample at this node.

        Args:
            x: npt.NDArray[np.float64]
                The input sample.

        Returns:
            float: Log-probability value.
        """
        pass

    @abstractmethod
    def marginalize(self, vars_to_keep: Set[int]) -> "TreeNode":
        """
        Marginalize out the specified variables from this node.

        Args:
            vars_to_keep: Set[int]
                The indices of the variables to keep.

        Returns:
            TreeNode: A new node representing the marginal distribution.
        """
        pass

    @abstractmethod
    def flatten(self, vars_to_keep: Set[int]) -> Histogram:
        """
        Recursively flatten a sub-tree into a single Histogram.
        """
        pass


class LeafNode(TreeNode):
    """
    Representing a leaf in the tree, containing a joint histogram
    for its scope.
    """

    def __init__(self, scope: List[int], histogram: Histogram):
        """
        Initialize the LeafNode.

        Args:
            scope: List[int]
                The indices of variables present in this leaf.
            histogram: JointHistogram
                A JointHistogram object.
        """
        self.scope = sorted(list(scope))
        self.histogram = histogram

    def log_inference(self, x: npt.NDArray[np.float64]) -> float:
        """
        Log-inference for a leaf node: log-probability of its histogram.
        """
        return self.histogram.log_inference(x)

    def marginalize(self, vars_to_keep: Set[int]) -> "LeafNode":
        """
        Remove variables from the leaf's scope and marginalize its histograms.
        """
        new_hist = self.flatten(vars_to_keep)
        return LeafNode(new_hist.scope, new_hist)

    def flatten(self, vars_to_keep: Set[int]) -> Histogram:
        """Flatten a leaf: marginalize to kept variables."""
        keep_set = vars_to_keep & set(self.scope)
        return self.histogram.marginalize(keep_set)


class DecisionNode(TreeNode):
    """
    A branching node in the TreeTPM that splits based on a single variable's
    value.
    """

    def __init__(
        self,
        split_var: int,
        children: List[TreeNode],
        feature_type: str,
        split_bins: Union[npt.NDArray[np.float64], List[Set[int]]],
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize the DecisionNode.

        Args:
            split_var: int
                The index of the variable to split on.
            children: List[TreeNode]
                A list of child nodes corresponding to each bin/group.
            feature_type: str
                'continuous' or 'categorical'.
            split_bins: Union[npt.NDArray[np.float64], List[Set[int]] | List[float]]
                Bin edges (array) or categorical groups (list of sets).
            weights: Optional[List[float]]
                The mixing weights for each branch.
        """
        self.split_var = split_var
        self.children = children
        self.feature_type = feature_type
        self.split_bins = split_bins
        self.weights = (
            weights
            if weights
            else ([1.0 / len(children)] * len(children) if children else [])
        )

    def log_inference(self, x: npt.NDArray[np.float64]) -> float:
        """
        Discretize the input using split_bins and route to the corresponding child.
        """
        val = x[self.split_var]

        if self.feature_type == "continuous":
            # Continuous binning: find index i such that edges[i] <= val < edges[i+1]
            self.split_bins = cast(npt.NDArray[np.float64], self.split_bins)
            idx = int(np.digitize(val, self.split_bins) - 1)
            idx = int(np.clip(idx, 0, len(self.split_bins) - 2))
            # Optional: Return MIN_LOG_PROB if outside total range
            if val < self.split_bins[0] or val > self.split_bins[-1]:
                return MIN_LOG_PROB
        else:
            # Categorical grouping: find which set the value belongs to
            idx = -1
            for b_idx, group in enumerate(self.split_bins):
                if val in group:
                    idx = b_idx
                    break
            if idx == -1:
                return MIN_LOG_PROB

        return float(
            np.log(max(1e-12, self.weights[idx])) + self.children[idx].log_inference(x)
        )

    def marginalize(self, vars_to_keep: Set[int]) -> TreeNode:
        """
        If the split variable is marginalized out, the node collapses into a histogram.
        Otherwise, it recursively marginalizes its children.
        """
        if self.split_var not in vars_to_keep:
            # Flatten gets a joint histogram over the subtree
            final_h = self.flatten(vars_to_keep)
            return LeafNode(final_h.scope, final_h)
        else:
            new_children = [child.marginalize(vars_to_keep) for child in self.children]
            return DecisionNode(
                self.split_var,
                new_children,
                self.feature_type,
                self.split_bins,
                self.weights,
            )

    def flatten(self, vars_to_keep: Set[int]) -> Histogram:
        """Recursively flatten DecisionNode."""
        child_hists = [child.flatten(vars_to_keep) for child in self.children]

        # If we are keeping the split variable, expand and restrict each child
        if self.split_var in vars_to_keep:
            for i in range(len(child_hists)):
                if self.feature_type == "continuous":
                    edges = cast(npt.NDArray[np.float64], self.split_bins)
                    child_hists[i] = child_hists[i].expand_scope(
                        self.split_var,
                        edges,
                        [],
                        self.feature_type,
                        bin_restriction={i},
                    )
                else:
                    bins = cast(List[Set[int]], self.split_bins)
                    child_hists[i] = child_hists[i].expand_scope(
                        self.split_var,
                        np.array([]),
                        bins,
                        self.feature_type,
                        bin_restriction={i},
                    )

        # Unify scopes across all children before combination
        all_vars = sorted(list(set().union(*(h.scope for h in child_hists))))
        if not all_vars:
            # Every child returned empty scope (everything marginalized out)
            return JointHistogram([], {}, {}, np.array(0.0), [])

        # Collect bins/edges from wherever they are available in the children
        edges_map = {}
        bins_map = {}
        types_map = {}
        for h in child_hists:
            for i, v_idx in enumerate(h.scope):
                if v_idx not in types_map:
                    types_map[v_idx] = h.feature_types[i]
                    if h.feature_types[i] == "continuous":
                        if v_idx in h.edges:
                            edges_map[v_idx] = h.edges[v_idx]
                    else:
                        if v_idx in h.bins:
                            bins_map[v_idx] = h.bins[v_idx]

        # Ensure all histograms have the same global scope by adding some binning to the missing variables
        for i in range(len(child_hists)):
            for v_idx in all_vars:
                if v_idx not in child_hists[i].scope:
                    if types_map[v_idx] == "continuous":
                        child_hists[i] = child_hists[i].expand_scope(
                            v_idx, edges_map[v_idx], [], types_map[v_idx]
                        )
                    else:
                        child_hists[i] = child_hists[i].expand_scope(
                            v_idx, np.array([]), bins_map[v_idx], types_map[v_idx]
                        )

        # Combine child histograms using the weights
        base_h = None
        total_w = 0.0
        for w, h in zip(self.weights, child_hists):
            if base_h is None:
                base_h = h
                total_w = w
            else:
                base_h = base_h.combine(h, total_w / (total_w + w), w / (total_w + w))
                total_w += w
        if base_h is None:
            raise ValueError("No valid histograms to combine")

        return base_h
