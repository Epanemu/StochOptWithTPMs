from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from .base import MIN_LOG_PROB


class Histogram(ABC):
    """
    Abstract base class for histograms using log-probabilities.
    """

    scope: List[int]
    edges: Dict[int, npt.NDArray[np.float64]]
    bins: Dict[int, List[Set[int]]]
    log_probs: npt.NDArray[np.float64]
    feature_types: List[str]

    @abstractmethod
    def log_inference(self, x: npt.NDArray[np.float64]) -> float:
        """
        Calculate the log-probability of a value x.

        Args:
            x: npt.NDArray[np.float64]
                The input value.

        Returns:
            float: Log-probability (or density) value.
        """
        pass

    @abstractmethod
    def expand_scope(
        self,
        new_var_idx: int,
        new_var_edges: npt.NDArray[np.float64],
        new_var_bins: List[Set[int]],
        new_var_type: str,
        bin_restriction: Optional[Set[int]] = None,
    ) -> "Histogram":
        """
        Expand the histogram's scope by adding a single variable.

        Args:
            new_var_idx: int
                The index of the new variable.
            new_var_edges: npt.NDArray[np.float64]
                Bin edges for the new variable, if continuous.
            new_var_bins: List[Set[int]]
                Bin definitions for the new variable, if categorical.
            new_var_type: str
                'continuous' or 'categorical'.
            bin_restriction: Optional[Set[int]]
                Optional set of bin indices to restrict the mass to.
        """
        pass

    @abstractmethod
    def marginalize(self, vars_to_keep: Set[int]) -> "Histogram":
        """
        The integral of a normalized histogram is 1, so the log-integral is 0.

        Args:
            vars_to_keep: Set[int]
                The indices of the variables to keep.

        Returns:
            Histogram: A new histogram with the marginalized distribution.
        """
        pass


class JointHistogram(Histogram):
    """
    Multivariate histogram for joint distributions of mixed categorical and continuous
    variables. Stores log-probability mass. Density correction is performed during
    inference.
    """

    def __init__(
        self,
        scope: List[int],
        edges: Dict[int, npt.NDArray[np.float64]],
        bins: Dict[int, List[Set[int]]],
        log_probs: npt.NDArray[np.float64],
        feature_types: List[str],
    ):
        """
        Initialize the JointHistogram.

        Args:
            scope: List[int]
                List of variable indices in this joint distribution.
            edges: Dict[int, npt.NDArray[np.float64]]
                Map from variable index to bin edges (for continuous variables).
            bins: Dict[int, List[Set[int]]]
                Map from variable index to bin groups (for categorical variables).
            log_probs: npt.NDArray[np.float64]
                Multi-dimensional array of log-probability mass.
            feature_types: List[str]
                List of 'continuous' or 'categorical' for each variable.
        """
        self.scope = scope
        self.edges = edges
        self.bins = bins
        self.log_probs = log_probs
        self.feature_types = feature_types

        # Precompute maps and log-sizes for categorical values
        self.val_maps: Dict[int, Dict[int, Tuple[int, np.float64]]] = {}
        for i, var_idx in enumerate(scope):
            if feature_types[i] == "categorical":
                if var_idx not in bins:
                    raise ValueError(f"Missing categorical bins for variable {var_idx}")
                b = bins[var_idx]
                vm = {}
                for idx, group in enumerate(b):
                    log_group_size = np.log(len(group))
                    for val in group:
                        vm[int(val)] = (idx, log_group_size)
                self.val_maps[var_idx] = vm

    def __repr__(self) -> str:
        s = f"JointHistogram(scope={self.scope}, shape={self.log_probs.shape}, types={self.feature_types})\n"
        if self.log_probs.size > 50:
            s += "  (Log-probs table too large to display, showing first 50 entries)\n"

        # Try to show a table-like representation
        header = " | ".join([f"v{v}" for v in self.scope] + ["log_prob", "prob"])
        s += "  " + header + "\n"
        s += "  " + "-" * len(header) + "\n"

        count = 0
        for idx in np.ndindex(self.log_probs.shape):
            if count > 50:
                s += "  ...\n"
                break
            line_parts = []
            for i, v_idx in enumerate(self.scope):
                if self.feature_types[i] == "continuous":
                    edges = self.edges[v_idx]
                    line_parts.append(f"[{edges[idx[i]]:.2f}, {edges[idx[i]+1]:.2f})")
                else:
                    groups = self.bins[v_idx]
                    line_parts.append(str(groups[idx[i]]))
            lp = self.log_probs[idx]
            line_parts.append(f"{lp:.4f}")
            line_parts.append(f"{np.exp(lp):.4f}")
            s += "  " + " | ".join(line_parts) + "\n"
            count += 1
        return s

    def log_inference(self, x: npt.NDArray[np.float64]) -> float:
        """
        Compute joint log-probability (density) for a sample. x can be a vector or
        array matching the scope size.
        """
        if len(self.scope) == 0:
            # if we have no scope, we have a constant 1.0 histogram
            return 0.0

        indices = []
        total_log_correction = 0.0
        for i, var_idx in enumerate(self.scope):
            val = x[var_idx]

            if self.feature_types[i] == "categorical":
                if var_idx not in self.val_maps:
                    # Should not be reached if init worked correctly
                    return MIN_LOG_PROB
                if val not in self.val_maps[var_idx]:
                    return MIN_LOG_PROB
                idx, log_size = self.val_maps[var_idx][val]
                indices.append(idx)
                total_log_correction += float(log_size)
            else:
                v = float(val)
                if var_idx not in self.edges:
                    raise ValueError(f"Missing continuous edges for variable {var_idx}")
                edges = self.edges[var_idx]

                if v < edges[0] or v > edges[-1]:
                    return MIN_LOG_PROB
                idx = int(np.searchsorted(edges, v, side="right") - 1)
                idx = int(np.clip(idx, 0, self.log_probs.shape[i] - 1))
                indices.append(idx)
                total_log_correction += float(
                    np.log(max(1e-12, edges[idx + 1] - edges[idx]))
                )

        return float(self.log_probs[tuple(indices)] - total_log_correction)

    def marginalize(self, vars_to_keep: Set[int]) -> "JointHistogram":
        """
        Marginalize out variables NOT in vars_to_keep.
        Summing log-probs in mass space (logsumexp).
        """
        keep_local_indices = [i for i, v in enumerate(self.scope) if v in vars_to_keep]
        if not keep_local_indices:
            # if we marginalize out all variables, we are left with a constant 1.0
            # histogram
            return JointHistogram([], {}, {}, np.array(0.0), [])

        axes_to_sum = tuple(
            i for i in range(len(self.scope)) if i not in keep_local_indices
        )

        # Use logsumexp for stability
        new_log_probs = logsumexp(self.log_probs, axis=axes_to_sum)

        new_scope = [self.scope[i] for i in keep_local_indices]
        new_edges = {v: self.edges[v] for v in new_scope if v in self.edges}
        new_bins = {v: self.bins[v] for v in new_scope if v in self.bins}
        new_types = [self.feature_types[i] for i in keep_local_indices]

        return JointHistogram(new_scope, new_edges, new_bins, new_log_probs, new_types)

    def expand_scope(
        self,
        new_var_idx: int,
        new_var_edges: npt.NDArray[np.float64],
        new_var_bins: List[Set[int]],
        new_var_type: str,
        bin_restriction: Optional[Set[int]] = None,
    ) -> "JointHistogram":
        """
        Expand the joint histogram's scope by adding a single variable.
        """
        if new_var_idx in self.scope:
            return self

        # 1. Determine new scope and variable placement
        new_scope = sorted(self.scope + [new_var_idx])
        new_pos = int(new_scope.index(new_var_idx))

        # 2. Prepare new bins/edges and types
        new_edges = {k: v for k, v in self.edges.items()}
        new_bins = {k: v for k, v in self.bins.items()}

        if new_var_type == "continuous":
            if new_var_edges is None or not isinstance(new_var_edges, np.ndarray):
                raise ValueError("Continuous variable must take edges as numpy array")
            new_edges[new_var_idx] = new_var_edges
            n_bins_new = len(new_var_edges) - 1
        else:
            if new_var_bins is None:
                raise ValueError("Categorical variable must take bins as List[Set]")
            new_bins[new_var_idx] = new_var_bins
            n_bins_new = len(new_var_bins)

        new_types = list(self.feature_types)
        new_types.insert(new_pos, new_var_type)

        # 3. Shape for the new log_probs array
        new_shape = list(self.log_probs.shape)
        new_shape.insert(new_pos, n_bins_new)

        # 4. Fill the new log_probs array
        # Uniform expansion factor: log(1/n_bins)
        if bin_restriction:
            log_expansion_factor = -np.log(len(bin_restriction))
            final_log_probs = np.full(new_shape, MIN_LOG_PROB)
            for b_idx in bin_restriction:
                if 0 <= b_idx < n_bins_new:
                    target_slice: List[slice | int] = [slice(None)] * len(new_shape)
                    target_slice[new_pos] = b_idx
                    final_log_probs[tuple(target_slice)] = (
                        self.log_probs + log_expansion_factor
                    )
        else:
            log_expansion_factor = -np.log(n_bins_new)
            expanded_log_probs = np.expand_dims(self.log_probs, axis=new_pos)
            final_log_probs = expanded_log_probs + log_expansion_factor

        return JointHistogram(
            new_scope, new_edges, new_bins, final_log_probs, new_types
        )

    @staticmethod
    def unify_edges(
        edges1: npt.NDArray[np.float64], edges2: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Unify two bin definitions to the most granular common grid."""
        if not isinstance(edges1, np.ndarray) or not isinstance(edges2, np.ndarray):
            raise ValueError(
                "Continuous variable bin edges must be a numpy array of floats."
            )
        return np.union1d(edges1, edges2)

    @staticmethod
    def unify_bins(bins1: List[Set[int]], bins2: List[Set[int]]) -> List[Set[int]]:
        """Unify two bin definitions to the most granular common grid."""
        # 1. Collect all unique values from both sets of bins
        all_values = set()
        for b in bins1:
            all_values.update(b)
        for b in bins2:
            all_values.update(b)

        # 2. Map each value to a "signature": (index_in_bins1, index_in_bins2)
        # We use -1 if a value is not present in one of the bin sets.
        signatures: Dict[int, Tuple[int, int]] = {}
        for val in all_values:
            idx1 = -1
            for i, b in enumerate(bins1):
                if val in b:
                    idx1 = i
                    break
            idx2 = -1
            for i, b in enumerate(bins2):
                if val in b:
                    idx2 = i
                    break
            signatures[val] = (idx1, idx2)

        # 3. Group values by their signatures
        groups: Dict[Tuple[int, int], Set[int]] = {}
        for val, sig in signatures.items():
            if sig not in groups:
                groups[sig] = set()
            groups[sig].add(val)

        return list(groups.values())

    def combine(
        self, other: "JointHistogram", w_self: float, w_other: float
    ) -> "JointHistogram":
        """
        Combine two joint histograms (mixture). Assumes scopes match.
        """
        if self.scope != other.scope:
            raise NotImplementedError(
                "Combining JointHistograms with different scopes "
                "is not yet implemented."
            )

        l_w1 = np.log(w_self)
        l_w2 = np.log(w_other)

        # If bins are identical, simply logaddexp the weighted masses
        edges_match = True
        if set(self.edges.keys()) != set(other.edges.keys()):
            edges_match = False
        else:
            for k in self.edges:
                if not np.array_equal(self.edges[k], other.edges[k]):
                    edges_match = False
                    break

        bins_match = True
        if set(self.bins.keys()) != set(other.bins.keys()):
            bins_match = False
        else:
            for k in self.bins:
                if self.bins[k] != other.bins[k]:
                    bins_match = False
                    break

        if edges_match and bins_match:
            return JointHistogram(
                self.scope,
                self.edges,
                self.bins,
                np.logaddexp(l_w1 + self.log_probs, l_w2 + other.log_probs),
                self.feature_types,
            )

        # Different grids: union of partitions
        new_edges = {}
        new_bins: Dict[int, List[Set[int]]] = {}

        shapes = []

        # Reconstruct ordered list of bins for processing
        for i, var_idx in enumerate(self.scope):
            if self.feature_types[i] == "continuous":
                if var_idx not in self.edges or var_idx not in other.edges:
                    raise ValueError(f"Missing edges for {var_idx}")
                new_edge_set = self.unify_edges(
                    self.edges[var_idx], other.edges[var_idx]
                )
                new_edges[var_idx] = new_edge_set
                shapes.append(len(new_edge_set) - 1)
            else:
                if var_idx not in self.bins or var_idx not in other.bins:
                    raise ValueError(f"Missing bins for {var_idx}")
                new_bin_set = self.unify_bins(self.bins[var_idx], other.bins[var_idx])
                new_bins[var_idx] = new_bin_set
                shapes.append(len(new_bin_set))

        shape = tuple(shapes)

        # check if shape fits in memory
        if np.prod(shape) > 1e8:
            raise MemoryError(f"Shape of new log_probs table is too large: {shape}")
        new_log_probs = np.full(shape, -np.inf)

        # Precompute sub-cell mass redistribution mapping
        # maps to original bin indices and log mass fraction
        dim_maps_self = []
        dim_maps_other = []

        for i, var_idx in enumerate(self.scope):
            # mapping of the new bins to the original self / other bins
            # (or -1 in case of out-of-bounds value) and the fraction
            # of the original mass
            m_s, m_o = [], []
            if self.feature_types[i] == "continuous":
                for j in range(len(new_edges[var_idx]) - 1):
                    mid = (new_edges[var_idx][j] + new_edges[var_idx][j + 1]) / 2.0
                    idx_s = np.searchsorted(self.edges[var_idx], mid) - 1
                    idx_o = np.searchsorted(other.edges[var_idx], mid) - 1

                    # Fraction of parent bin mass: (new_width / old_width)
                    new_w = float(new_edges[var_idx][j + 1] - new_edges[var_idx][j])
                    if 0 <= idx_s < len(self.edges[var_idx]) - 1:
                        old_w_s = (
                            self.edges[var_idx][idx_s + 1] - self.edges[var_idx][idx_s]
                        )
                        f_s = np.log(new_w) - np.log(old_w_s)
                        m_s.append((idx_s, f_s))
                    else:
                        m_s.append((-1, -np.inf))

                    if 0 <= idx_o < len(other.edges[var_idx]) - 1:
                        old_w_o = (
                            other.edges[var_idx][idx_o + 1]
                            - other.edges[var_idx][idx_o]
                        )
                        f_o = np.log(new_w) - np.log(old_w_o)
                        m_o.append((idx_o, f_o))
                    else:
                        m_o.append((-1, -np.inf))
            else:
                nb_list = new_bins[var_idx]
                for j, group in enumerate(nb_list):
                    sample_val = next(iter(group))
                    idx_s, log_size_s = self.val_maps[var_idx].get(
                        sample_val, (-1, 0.0)
                    )
                    idx_o, log_size_o = other.val_maps[var_idx].get(
                        sample_val, (-1, 0.0)
                    )

                    # Fraction of parent bin mass: (new_group_size / old_group_size)
                    f_s = np.log(len(group)) - log_size_s if idx_s != -1 else -np.inf
                    f_o = np.log(len(group)) - log_size_o if idx_o != -1 else -np.inf
                    m_s.append((idx_s, f_s))
                    m_o.append((idx_o, f_o))
            dim_maps_self.append(m_s)
            dim_maps_other.append(m_o)

        for idx_new in np.ndindex(shape):
            # Self contribution
            l1_indices = [dim_maps_self[i][idx_new[i]][0] for i in range(len(shape))]
            l1_factors = [dim_maps_self[i][idx_new[i]][1] for i in range(len(shape))]
            if -1 in l1_indices:
                lp1 = -np.inf
            else:
                lp1 = self.log_probs[tuple(l1_indices)] + sum(l1_factors)

            # Other contribution
            l2_indices = [dim_maps_other[i][idx_new[i]][0] for i in range(len(shape))]
            l2_factors = [dim_maps_other[i][idx_new[i]][1] for i in range(len(shape))]
            if -1 in l2_indices:
                lp2 = -np.inf
            else:
                lp2 = other.log_probs[tuple(l2_indices)] + sum(l2_factors)

            new_log_probs[idx_new] = logsumexp([l_w1 + lp1, l_w2 + lp2])

        return JointHistogram(
            self.scope, new_edges, new_bins, new_log_probs, self.feature_types
        )
