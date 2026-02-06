from typing import Any, Generator, List, Tuple

import numpy as np
import pyomo.environ as pyo
from stochopt.tpms.CNet.cnet_learning import DecisionNode, LeafNode


def _traverse(node: Any) -> Generator[Any, None, None]:
    """
    Helper to traverse the tree and yield all nodes.
    """
    yield node
    if isinstance(node, DecisionNode):
        for child in node.branches.values():
            yield from _traverse(child)


def build_cnet_milp(
    cnet_model: Any,
    model_block: pyo.Block,
    inputs: List[List[Any]],
    solver: str = "gurobi",
    **kwargs: Any,
) -> Tuple[pyo.Var, int]:
    """
    Builds MILP constraints for a CNet structure on a Pyomo block.

    Args:
        cnet_model: Any
            Root node of the learned CNet (from cnet_learning.py).
        model_block: pyo.Block
            Pyomo block to add variables and constraints to.
        inputs: List[List[Any]]
            List of one-hot encoded Pyomo variables [feat_idx][val].
            Even binary variables should be passed as one-hot encoded lists.
        solver: str
            Solver name (default "gurobi").
        **kwargs: Any
            Additional arguments.

    Returns:
        Tuple[pyo.Var, int]: [model_block.log_prob, root_node_id].
    """
    all_nodes = list(_traverse(cnet_model))
    for i, n in enumerate(all_nodes):
        n.id = i

    node_ids = [n.id for n in all_nodes]

    # Add log_prob variables for each node
    model_block.log_prob = pyo.Var(node_ids, bounds=(-1000, 0))
    # if any of inputs is none, raise an error
    if any(any(inval is None for inval in inval_list) for inval_list in inputs):
        raise ValueError("Marginalized inputs cannot be modelled")

    # Recursive constraint adding
    _add_node_constraints(model_block, cnet_model, node_ids, inputs)

    return model_block.log_prob, node_ids[0]


def _add_node_constraints(
    model_block: pyo.Block, node: Any, node_ids: List[int], inputs: List[List[Any]]
) -> None:
    """
    Recursively adds Pyomo constraints for each node in the CNet.
    """
    node_id = node.id
    M = 1000.0

    if isinstance(node, LeafNode):
        # CLTree leaf: log P(x) = sum_i log P(x_i | x_parent(i))
        scope = node.scope
        tree = node.tree
        log_factors = node.log_factors
        terms = []

        for i in range(len(scope)):
            var_idx = scope[i]
            parent_local_idx = tree[i]

            # log_factors[i] shape: [dom_size, parent_dom_size]
            lf = log_factors[i]
            dom_size = lf.shape[0]

            term = pyo.Var(bounds=(-1000, 0))
            model_block.add_component(f"node_{node_id}_term_{i}", term)
            terms.append(term)

            if parent_local_idx == -1:
                # Root of CLTree: log P(x_i) = sum_v I(x_i=v) * log P(v)
                expr = sum(
                    inputs[var_idx][v] * max(lf[v, 0], -1000) for v in range(dom_size)
                )
                model_block.add_component(
                    f"node_{node_id}_root_{i}_c", pyo.Constraint(expr=term == expr)
                )
            else:
                # Child in CLTree: log P(x_i | x_p) = sum_v,u I(x_i=v, x_p=u) * log P(v|u)
                parent_var_idx = scope[parent_local_idx]
                parent_dom_size = lf.shape[1]

                sum_expr = 0
                for v in range(dom_size):
                    for u in range(parent_dom_size):
                        if lf[v, u] <= -np.inf:
                            continue

                        # Linearize I(x_i=v) * I(x_p=u)
                        z = pyo.Var(within=pyo.Binary)
                        model_block.add_component(f"node_{node_id}_i{i}_v{v}_u{u}", z)
                        model_block.add_component(
                            f"node_{node_id}_i{i}_v{v}_u{u}_c1",
                            pyo.Constraint(expr=z <= inputs[var_idx][v]),
                        )
                        model_block.add_component(
                            f"node_{node_id}_i{i}_v{v}_u{u}_c2",
                            pyo.Constraint(expr=z <= inputs[parent_var_idx][u]),
                        )
                        model_block.add_component(
                            f"node_{node_id}_i{i}_v{v}_u{u}_c3",
                            pyo.Constraint(
                                expr=z
                                >= inputs[var_idx][v] + inputs[parent_var_idx][u] - 1
                            ),
                        )

                        sum_expr += z * lf[v, u]
                model_block.add_component(
                    f"node_{node_id}_cond_{i}_c", pyo.Constraint(expr=term == sum_expr)
                )

        model_block.add_component(
            f"node_{node_id}_sum_c",
            pyo.Constraint(expr=model_block.log_prob[node_id] == sum(terms)),
        )

    elif isinstance(node, DecisionNode):
        # Decision split: log P(x) = log P(v) + log P(rest | v)
        d_var = node.decision_var
        for val, child in node.branches.items():
            child_id = child.id
            log_p = np.log(max(1e-10, node.probs[val]))
            indicator = inputs[d_var][int(val)]

            # Big-M constraints ensure log_prob_node = log_p + log_prob_child when indicator is 1
            # TODO compute tight big-M constraints
            model_block.add_component(
                f"node_{node_id}_v{val}_ub",
                pyo.Constraint(
                    expr=model_block.log_prob[node_id]
                    <= log_p + model_block.log_prob[child_id] + M * (1 - indicator)
                ),
            )
            model_block.add_component(
                f"node_{node_id}_v{val}_lb",
                pyo.Constraint(
                    expr=model_block.log_prob[node_id]
                    >= log_p + model_block.log_prob[child_id] - M * (1 - indicator)
                ),
            )

            _add_node_constraints(model_block, child, node_ids, inputs)
    else:
        raise NotImplementedError(
            f"Node type {type(node)} with no handling implemented."
        )
