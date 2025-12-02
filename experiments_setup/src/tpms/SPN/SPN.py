from enum import Enum
from typing import Any

import numpy as np
from spn.algorithms.Inference import EPSILON, log_likelihood
from spn.algorithms.LearningWrappers import learn_mspn
from spn.algorithms.Marginalization import marginalize
from spn.structure.Base import Context, Leaf
from spn.structure.Base import Node as SPFlow_Node
from spn.structure.Base import Product, Sum, get_topological_order
from spn.structure.StatisticalTypes import MetaType

from data.DataHandler import DataHandler
from data.Features import Binary, Categorical, Contiguous, Feature, Mixed
from data.Types import DataLike


class NodeType(
    Enum
):  # TODO make this into a class and subclasses so that isinstance leaf works on all 3 kinds of leaves
    SUM = 0
    PRODUCT = 1
    LEAF = 2
    LEAF_CATEGORICAL = 3
    LEAF_BINARY = 4


class Node:
    """A representation of a node in an SPN"""

    def __init__(
        self,
        node: SPFlow_Node,
        feature_list: list[Feature],
        normalize: bool,
        min_density: float,
    ):
        self.__normalize = normalize
        self.__min_density = min_density
        if isinstance(node, Leaf):
            self.densities = list(node.densities)
            if isinstance(node.scope, list):
                if len(node.scope) > 1:
                    raise NotImplementedError("Multivariate leaves are not supported.")
                self.scope = node.scope[0]
            else:
                self.scope = node.scope
            self.feature = feature_list[self.scope]
            if isinstance(self.feature, Categorical):
                self.type = NodeType.LEAF_CATEGORICAL
                self.options = self.feature.numeric_vals
            elif isinstance(self.feature, Binary):
                self.type = NodeType.LEAF_BINARY
            else:
                self.type = NodeType.LEAF
                # print(node.id, node.breaks, node.densities)
                self.discrete = self.feature.discrete
                if self.discrete:
                    self.breaks = [b - 0.5 for b in node.breaks]
                else:
                    self.breaks = list(node.breaks)
                dens = node.densities
                duplicate = np.isclose(dens[1:], dens[:-1], rtol=1e-10)
                self.densities = [dens[0]] + list(np.array(dens[1:])[~duplicate])
                self.breaks = (
                    [self.breaks[0]]
                    + list(np.array(self.breaks[1:-1])[~duplicate])
                    + [self.breaks[-1]]
                )
                # pruned_dens = [self.densities[0]]
                # pruned_breaks = [self.breaks[0]]
                # for i, d in enumerate(self.densities):
                #     if not np.isclose(pruned_dens[-1], d, atol=1e-10):
                #         pruned_dens.append(d)
                #         pruned_breaks.append(self.breaks[i])
                # self.densities = pruned_dens
                # self.breaks = pruned_breaks
        elif isinstance(node, Product):
            self.type = NodeType.PRODUCT
        elif isinstance(node, Sum):
            self.type = NodeType.SUM
            self.weights = node.weights
        else:
            raise ValueError("")
        self.name = node.name
        self.id = node.id
        # TODO make the predecessors also of this class, not the spflow one - or better yet, make this only a list of ids
        # TODO rework this so that the nodes are remembered in the SPN class, and not generated on demand
        self.predecessors = node.children if hasattr(node, "children") else []

    def get_breaks_densities(
        self, span_all=True, get_normalized=False
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """Retruns the breakpoints (rescaled to 0-1 range if get_normalized=True)

        Args:
            span_all (bool, optional): If True, the breaks will span the entire range of input feature. Defaults to True.

        Raises:
            ValueError: If called for a node that is not a leaf node over a contiguous feature
            AssertionError: If the feature bounds are not available

        Returns:
            tuple[np.ndarray[float], np.ndarray[float]]: Breakpoints, Density values
        """
        if not hasattr(self, "feature") or not isinstance(self.feature, Contiguous):
            raise ValueError("Only available to leaves over contiguous features")

        density_vals = self.densities
        breaks = self.breaks

        if span_all:
            lb, ub = (0, 1) if self.__normalize else self.feature.bounds
            if lb is None or ub is None:
                raise AssertionError("SPN input variables must have fixed bounds.")
            # if histogram is narrower than the input bounds
            if lb < breaks[0]:
                breaks = [lb] + breaks
                density_vals = [self.__min_density] + density_vals
            if ub > breaks[-1]:
                breaks = breaks + [ub]
                density_vals = density_vals + [self.__min_density]

        # if the breaks are not normalized, normalize them now
        if not self.__normalize and get_normalized:
            breaks = self.feature.encode(breaks, normalize=True, one_hot=False)

        return np.array(breaks), np.array(density_vals)


class SPN:
    N_PW_BREAKPOINTS = 5

    def __init__(
        self,
        data: DataLike,
        data_handler: DataHandler,
        normalize_data: bool = False,
        include_target: bool = False,
        # trunk-ignore(ruff/B006)
        learn_mspn_kwargs: dict[str, Any] = {},
    ):
        types = []
        domains = []
        if include_target:
            self.__feature_list = data_handler.features + [data_handler.target_feature]
        else:
            self.__feature_list = data_handler.features
        self.__include_target = include_target
        self.__data_handler = data_handler
        self.__normalize_data = normalize_data

        for feature in self.__feature_list:
            if isinstance(feature, Contiguous):
                if feature.discrete:
                    types.append(MetaType.DISCRETE)
                    domains.append(np.arange(feature.bounds[0], feature.bounds[1] + 1))
                else:
                    types.append(MetaType.REAL)
                    domains.append(np.asarray(feature.bounds))
            elif isinstance(feature, Categorical):
                types.append(MetaType.DISCRETE)
                domains.append(np.asarray(feature.numeric_vals))
            elif isinstance(feature, Binary):
                types.append(MetaType.BINARY)
                domains.append(np.asarray([0, 1]))
            elif isinstance(feature, Mixed):
                types.append(MetaType.REAL)
                domains.append(np.asarray(feature.bounds))
                # types.append(MetaType.DISCRETE) TODO add the doubling version to the mixed feature
            else:
                raise ValueError(f"Unsupported feature type of feature {feature}")

        # TODO: add domain handling to the vars
        # TODO: parametric types - types with distributions attached
        context = Context(
            meta_types=types,
            domains=domains,
            feature_names=[f.name for f in self.__feature_list],
        )
        enc_data = self.__encode_data(data)
        if len(domains) != data_handler.n_features + int(include_target):
            print("recomputing domains")
            context.add_domains(enc_data)

        self.__mspn = learn_mspn(enc_data, context, **learn_mspn_kwargs)
        self.__nodes = [
            Node(node, self.__feature_list, self.__normalize_data, self.min_density)
            for node in get_topological_order(self.__mspn)
        ]

    def __encode_data(self, data: DataLike):
        if self.__include_target:
            return self.__data_handler.encode_all(
                data, normalize=self.__normalize_data, one_hot=False
            )
        else:
            return self.__data_handler.encode(
                data, normalize=self.__normalize_data, one_hot=False
            )

    def compute_ll(self, data: DataLike):
        if len(data.shape) == 1:
            return self.compute_ll(data.reshape(1, -1))[0]
        return log_likelihood(self.__mspn, self.__encode_data(data))

    def compute_max_approx(self, data: DataLike, return_all: bool = False):
        if len(data.shape) != 1 or (data.shape[0] != 1 and len(data.shape) == 2):
            raise ValueError("Can do only one sample, so far...")

        input_data = self.__encode_data(data.reshape(1, -1))[0]

        node_vals = {}
        # node_ex_vals = {}
        for node in self.nodes:
            if node.type == NodeType.LEAF:
                for val, b in zip(
                    [self.min_density] + node.densities + [self.min_density],
                    node.breaks + [np.inf],
                ):
                    value = np.log(val)
                    if b > input_data[node.scope]:
                        break
            if node.type == NodeType.LEAF_BINARY:
                value = np.log(node.densities[input_data[node.scope].astype(int)])
            if node.type == NodeType.LEAF_CATEGORICAL:
                value = np.log(node.densities[input_data[node.scope].astype(int)])
            # node_ex_vals[node.id] = value
            if node.type == NodeType.PRODUCT:
                value = sum(node_vals[n.id] for n in node.predecessors)
                # node_ex_vals[node.id] = sum(
                #     node_ex_vals[n.id] for n in node.predecessors
                # )
            if node.type == NodeType.SUM:
                # print("Sum", [node_vals[n.id] for n in node.predecessors])
                value = max(
                    node_vals[n.id] + np.log(w)
                    for n, w in zip(node.predecessors, node.weights)
                )
                # node_ex_vals[node.id] = logsumexp(
                #     np.array([node_ex_vals[p.id] for p in node.predecessors]),
                #     b=node.weights,
                # )

            node_vals[node.id] = value

        if return_all:
            return node_vals
        return node_vals[self.__mspn.id]

    def compute_maxpw_approx(
        self, data: DataLike, maxlog: int, return_all: bool = False
    ):
        if len(data.shape) != 1 or (data.shape[0] != 1 and len(data.shape) == 2):
            raise ValueError("Can do only one sample, so far...")

        input_data = self.__encode_data(data.reshape(1, -1))[0]

        node_vals = {}
        # node_ex_vals = {}
        for node in self.nodes:
            if node.type == NodeType.LEAF:
                for val, b in zip(
                    [self.min_density] + node.densities + [self.min_density],
                    node.breaks + [np.inf],
                ):
                    value = np.log(val)
                    if b > input_data[node.scope]:
                        break
            if node.type == NodeType.LEAF_BINARY:
                value = np.log(node.densities[input_data[node.scope].astype(int)])
            if node.type == NodeType.LEAF_CATEGORICAL:
                value = np.log(node.densities[input_data[node.scope].astype(int)])
            # node_ex_vals[node.id] = value
            if node.type == NodeType.PRODUCT:
                value = sum(node_vals[n.id] for n in node.predecessors)
                # node_ex_vals[node.id] = sum(
                #     node_ex_vals[n.id] for n in node.predecessors
                # )
            if node.type == NodeType.SUM:
                # print("Sum", [node_vals[n.id] for n in node.predecessors])
                vals = np.array(
                    [
                        node_vals[n.id] + np.log(w)
                        for n, w in zip(node.predecessors, node.weights)
                    ]
                )
                max_value = vals.max()
                expvals = []
                for v in vals:
                    expval = self._exp_approx(v - max_value)
                    # print(" exp:", v, v-max_value, expval)
                    expvals.append(expval)
                logval = self._log_approx(sum(expvals), vals.shape[0])
                # print("LOG:", sum(expvals), logval)
                # print()
                value = max_value + logval

                # node_ex_vals[node.id] = logsumexp(
                #     np.array([node_ex_vals[p.id] for p in node.predecessors]),
                #     b=node.weights,
                # )

            node_vals[node.id] = value

        if return_all:
            return node_vals
        return node_vals[self.__mspn.id]

    def piecewise_linear(self, x, x_points, y_points):
        """
        Generated function!
        Performs piecewise linear interpolation.

        The function dynamically handles both ascending (Exp LUT) and
        descending (Log LUT) ordered breakpoint arrays by sorting them
        internally for consistent binary search logic.

        Args:
            x (float): The value at which to interpolate.
            x_points (list[float]): The input breakpoints (x-coordinates).
            y_points (list[float]): The output values (y-coordinates).

        Returns:
            float: The interpolated value.
        """
        nn = len(x_points)
        if nn < 2:
            return y_points[0] if y_points else np.nan

        # Determine if points are descending (like the Log points)
        is_descending = x_points[0] > x_points[1]

        if is_descending:
            # Create ascending copies for interpolation logic
            sorted_x_points = list(reversed(x_points))
            sorted_y_points = list(reversed(y_points))
        else:
            # Points are already ascending (like the Exp points)
            sorted_x_points = x_points
            sorted_y_points = y_points

        # 1. Check boundary conditions (clamping to the nearest breakpoint value)
        if x <= sorted_x_points[0]:
            return sorted_y_points[0]
        if x >= sorted_x_points[nn - 1]:
            return sorted_y_points[nn - 1]

        # 2. Find the interval [x_k, x_{k+1}] using a linear scan
        # In a real system, this would be a lookup/binary search for faster performance.
        kk = 0
        for i in range(nn - 1):
            if x < sorted_x_points[i + 1]:
                kk = i
                break

        # 3. Linear interpolation
        xk = sorted_x_points[kk]
        xk1 = sorted_x_points[kk + 1]
        yk = sorted_y_points[kk]
        yk1 = sorted_y_points[kk + 1]

        # Calculate the slope (m) and interpolate y = yk + m * (x - xk)
        slope = (yk1 - yk) / (xk1 - xk)
        return yk + slope * (x - xk)

    def _log_approx(self, val, maxval):
        x_points = np.logspace(
            np.log(1), np.log(maxval), num=self.N_PW_BREAKPOINTS, base=np.e
        )
        y_points = [np.log(x) for x in x_points]
        return self.piecewise_linear(val, x_points, y_points)

    def _exp_approx(self, val):
        # min_val = -4
        # min_val = -3
        min_val = np.log(0.001)
        if val < min_val:
            return 0  # do not approximate those where numerical inaccuracy would play a major role anyways
        # x_points = np.linspace(min_val, 0.0, self.N_PW_BREAKPOINTS).tolist()
        x_points = np.logspace(
            np.log(0.3), np.log(-min_val), num=self.N_PW_BREAKPOINTS - 1, base=np.e
        ).tolist()
        # x_points = np.logspace(np.log(0.5), np.log(-min_val), num=self.N_PW_BREAKPOINTS-1, base=np.e).tolist()
        x_points.insert(0, 0)
        x_points = [-v for v in reversed(x_points)]
        y_points = [np.exp(x) for x in x_points]
        return self.piecewise_linear(val, x_points, y_points)

    @property
    def nodes(self) -> list[Node]:
        """Nodes in topological ordering"""
        if not hasattr(self, "SPN__nodes"):
            self.__nodes = [
                Node(node, self.__feature_list, self.__normalize_data, self.min_density)
                for node in get_topological_order(self.__mspn)
            ]
        return self.__nodes

    @property
    def min_density(self) -> float:
        return EPSILON

    @property
    def out_node_id(self) -> float:
        return self.__mspn.id

    @property
    def spn_model(self):
        return self.__mspn

    def input_scale(self, feature_i):
        if self.__normalize_data:
            return 1
        else:
            return self.__data_handler.features[feature_i]._scale

    def marginalize(self, indices_to_keep: list[int]) -> None:
        self.__mspn = marginalize(self.__mspn, indices_to_keep)
        # print(self.__data_handler.features, )
        # print([n.scope for n in self.__nodes if n.type in [NodeType.LEAF, NodeType.LEAF_BINARY, NodeType.LEAF_CATEGORICAL]])
        self.__nodes = [
            Node(node, self.__feature_list, self.__normalize_data, self.min_density)
            for node in get_topological_order(self.__mspn)
        ]
        # print([n.scope for n in self.__nodes if n.type in [NodeType.LEAF, NodeType.LEAF_BINARY, NodeType.LEAF_CATEGORICAL]])

    # def normalize(self, feature_i, vals):
    #     if self.__normalize_data:
    #         return vals
    #     else:
    #         return self.__data_handler.features[feature_i].encode(
    #             vals, normalize=True, one_hot=False
    #         )
