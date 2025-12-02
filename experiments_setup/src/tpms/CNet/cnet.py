import numpy as np
import pyomo.environ as pyo
import spn.algorithms.Inference as spflow_inference
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_cnet

# from spn.structure.leaves.parametric.Bernoulli import create_bernoulli_leaf
from spn.structure.Base import Context, Product, Sum
from spn.structure.leaves.cltree.CLTree import CLTree
from spn.structure.leaves.parametric.Parametric import (
    Bernoulli,
    Categorical,
    Parametric,
)


class LeafNode:
    """Represents a terminal node: a base distribution (CLTree or Bernoulli)."""

    def __init__(self, scope, dist_type, params):
        self.scope = scope
        self.dist_type = dist_type
        self.params = params

    def __repr__(self, level=0):
        indent = "  " * level
        scope_str = str(sorted(list(self.scope)))
        if self.dist_type == "Bernoulli":
            p0 = self.params["p0"]
            return f"{indent}LEAF(dist=Bernoulli, scope={scope_str}, P(0)={p0:.3f})"
        if len(self.scope) == 1:
            p0 = np.exp(self.params["log_factors"][0][0][0])
            return f"{indent}LEAF(dist=CLTree, scope={scope_str}, P(0)={p0:.3f}, P(1)={1-p0:.3f})"
        return f"{indent}LEAF(dist=CLTree, scope={scope_str})"

    def inference(self, x):
        if self.dist_type == "Bernoulli":
            val = x[list(self.scope)[0]]
            p0 = self.params["p0"]
            return p0 if val == 0 else (1.0 - p0)

        # CLTree inference logic
        scope_list = sorted(list(self.scope))
        x_slice = x[scope_list]
        total_log_prob = 0.0
        for i in range(len(scope_list)):
            v_i = int(x_slice[i])
            parent_local_idx = self.params["tree"][i]
            v_p = 0 if parent_local_idx == -1 else int(x_slice[parent_local_idx])
            total_log_prob += self.params["log_factors"][i][v_i][v_p]
        return np.exp(total_log_prob)


# class FactorizationNode:
#     """Represents an 'AND' gate: a decomposition into independent factors."""

#     def __init__(self, scope, children):
#         self.scope = scope
#         self.children = children

#     def __repr__(self, level=0):
#         indent = "  " * level
#         scope_str = str(sorted(list(self.scope)))
#         s = f"{indent}FACTORIZATION (scope={scope_str})"
#         for child in self.children:
#             s += f"\n{child.__repr__(level + 1)}"
#         return s

#     def inference(self, x):
#         prob = 1.0
#         for child in self.children:
#             prob *= child.inference(x)
#         return prob


class DecisionNode:
    """Represents an 'OR' gate: a conditional branch on a variable."""

    def __init__(self, scope, decision_scope, p0, zero_branch, one_branch):
        self.scope = scope
        self.decision_scope = decision_scope
        self.p0 = p0  # Prior P(decision_var = 0)
        self.zero_branch = zero_branch
        self.one_branch = one_branch

    def __repr__(self, level=0):
        indent = "  " * level
        s = f"{indent}DECISION on var [{self.decision_scope}]"
        s += f"\n{indent}├─ If 0 (prob={self.p0:.3f}):"
        s += f"\n{self.zero_branch.__repr__(level + 1)}"
        s += f"\n{indent}└─ If 1 (prob={(1-self.p0):.3f}):"
        s += f"\n{self.one_branch.__repr__(level + 1)}"
        return s

    def inference(self, x):
        decision_value = x[self.decision_scope]
        if decision_value == 0:
            return self.p0 * self.zero_branch.inference(x)
        else:
            return (1.0 - self.p0) * self.one_branch.inference(x)


# class MixtureNode:
#     """Fallback for Sum nodes that aren't clean conditional decisions."""

#     def __init__(self, scope, outcomes):
#         self.scope = scope
#         self.outcomes = outcomes  # List of (weight, node) tuples

#     def __repr__(self, level=0):
#         indent = "  " * level
#         s = f"{indent}MIXTURE (scope={str(sorted(list(self.scope)))})"
#         for weight, node in self.outcomes:
#             s += f"\n{indent}  [weight={weight:.3f}]"
#             s += f"\n{node.__repr__(level + 2)}"
#         return s

#     def inference(self, x):
#         return sum(weight * node.inference(x) for weight, node in self.outcomes)


# --- The Intelligent Parser ---


def get_determinant_leaf(product_node):
    """Finds a determinant leaf (p=0 or p=1) within a Product node.
    This can be a Bernoulli leaf or a single-variable CLTree."""
    # Define a looser tolerance for floating point comparisons.
    # atol=1e-3 checks for equality up to 3 decimal places.
    # TOLERANCE = 1e-3
    # TOLERANCE = 5e-3
    TOLERANCE = 0.1

    # it is usually (or always?) the "last" leaf
    for i, child in enumerate(reversed(product_node.children)):
        leaf_info = None

        # # Case 1: The leaf is a standard Bernoulli node
        # if isinstance(child, Parametric):
        #     p0 = child.p[0]
        #     if np.isclose(p0, 1.0, atol=TOLERANCE) or np.isclose(
        #         p0, 0.0, atol=TOLERANCE
        #     ):
        #         leaf_info = {
        #             "scope": child.scope[0],
        #             "value": 0 if np.isclose(p0, 1.0, atol=TOLERANCE) else 1,
        #             "index": i,
        #         }

        # Case 2: The leaf is a CLTree modeling a single variable
        if isinstance(child, CLTree) and len(child.scope) == 1:
            # For a single-var CLTree, log_factors[0][0][0] is log(P(0))
            p0 = np.exp(child.log_factors[0][0][0])
            if np.isclose(p0, 1.0, atol=TOLERANCE) or np.isclose(
                p0, 0.0, atol=TOLERANCE
            ):
                leaf_info = {
                    "scope": child.scope[0],
                    "value": 0 if np.isclose(p0, 1.0, atol=TOLERANCE) else 1,
                    "index": i,
                }

        # If we found a determinant leaf, process it and return
        if leaf_info is not None:
            rest_children = [
                c
                for j, c in enumerate(reversed(product_node.children))
                if j != leaf_info["index"]
            ]

            if len(rest_children) == 1:
                rest_node = rest_children[0]
            else:
                rest_node = Product(children=rest_children)

            return {
                "scope": leaf_info["scope"],
                "value": leaf_info["value"],
                "rest_json": rest_node,
            }

    # No determinant leaf was found in this product node
    return None


def parse_cnet_as_decision_tree(node):
    """
    Parses an spflow CNet object into the semantic Decision/Factorization/Leaf tree.
    """
    if isinstance(node, (CLTree, Parametric)):
        params = (
            {"p0": node.p[0]}
            if isinstance(node, Parametric)
            else {"tree": node.tree, "log_factors": node.log_factors}
        )
        return LeafNode(
            set(node.scope),
            "Bernoulli" if isinstance(node, Parametric) else "CLTree",
            params,
        )

    if isinstance(node, Product):
        # children = [parse_cnet_as_decision_tree(c) for c in node.children]
        # scope = set().union(*(c.scope for c in children))
        # return FactorizationNode(scope, children)
        raise ValueError("Product node encountered.")

    if isinstance(node, Sum):
        # Look-ahead logic to find a decision variable
        if len(node.children) == 2 and all(
            isinstance(c, Product) for c in node.children
        ):
            info1 = get_determinant_leaf(node.children[0])
            info2 = get_determinant_leaf(node.children[1])

            # Check if we found a clean decision on the same variable
            if (
                info1
                and info2
                and info1["scope"] == info2["scope"]
                and info1["value"] != info2["value"]
            ):
                decision_scope = info1["scope"]
                if info1["value"] == 0:  # branch 1 is the '0' case
                    p0, zero_branch_json, one_branch_json = (
                        node.weights[0],
                        info1["rest_json"],
                        info2["rest_json"],
                    )
                else:  # branch 2 is the '0' case
                    p0, zero_branch_json, one_branch_json = (
                        node.weights[1],
                        info2["rest_json"],
                        info1["rest_json"],
                    )

                zero_branch = parse_cnet_as_decision_tree(zero_branch_json)
                one_branch = parse_cnet_as_decision_tree(one_branch_json)
                scope = set.union(zero_branch.scope, one_branch.scope, {decision_scope})
                return DecisionNode(scope, decision_scope, p0, zero_branch, one_branch)

        print(np.exp(node.children[0].children[-2].log_factors[0][0][0]))
        print(np.exp(node.children[1].children[-2].log_factors[0][0][0]))
        print(info1, info2)
        raise ValueError("Could not parse the sum node as decision node")
        # Fallback to generic mixture if no clean decision pattern is found
        # outcomes = [
        #     (node.weights[i], parse_cnet_as_decision_tree(c))
        #     for i, c in enumerate(node.children)
        # ]
        # scope = set().union(*(node.scope for _, node in outcomes))
        # return MixtureNode(scope, outcomes)

    raise TypeError(f"Unknown SPFlow node type: {type(node)}")


def build_cnet_milp(cnet_tree, mio_model, features):
    """
    Builds a Pyomo MILP model for the log-probability computation of a CNet.
    """
    node_map = {}

    def _get_node_id(node):
        obj_id = id(node)
        if obj_id not in node_map:
            node_map[obj_id] = len(node_map)
        return node_map[obj_id]

    mio_model.x = pyo.Var(range(len(features)), within=pyo.Binary)
    mio_model.fix_vars = pyo.ConstraintList()
    for i, var in enumerate(features):
        mio_model.fix_vars.add(mio_model.x[i] == var)
    node_indices = {_get_node_id(node) for node in _traverse(cnet_tree)}
    mio_model.log_prob = pyo.Var(node_indices, within=pyo.Reals, bounds=(-100.0, 0.0))
    mio_model.constraints = pyo.ConstraintList()
    _add_node_constraints(mio_model, cnet_tree, _get_node_id)
    return mio_model.log_prob, _get_node_id(cnet_tree)


def _traverse(node):
    """Helper to traverse the tree and yield all nodes."""
    yield node
    if hasattr(node, "children"):
        for child in node.children:
            yield from _traverse(child)
    elif hasattr(node, "outcomes"):
        for _, child in node.outcomes:
            yield from _traverse(child)
    elif hasattr(node, "zero_branch"):
        yield from _traverse(node.zero_branch)
        yield from _traverse(node.one_branch)


def _add_node_constraints(model, node, get_id_func):
    """Recursively adds Pyomo constraints for each node in the CNet."""
    node_id = get_id_func(node)

    if isinstance(node, LeafNode):
        if node.dist_type == "Bernoulli":
            var_idx = list(node.scope)[0]
            p0 = node.params["p0"]
            logp0 = np.log(p0) if p0 > 1e-9 else -100.0
            logp1 = np.log(1.0 - p0) if (1.0 - p0) > 1e-9 else -100.0
            model.constraints.add(
                model.log_prob[node_id]
                == model.x[var_idx] * logp1 + (1 - model.x[var_idx]) * logp0
            )

        elif node.dist_type == "CLTree":
            scope_list = sorted(list(node.scope))
            params = node.params
            log_factors = params["log_factors"]
            tree = params["tree"]
            terms = []

            for i in range(len(scope_list)):
                term_var = pyo.Var(within=pyo.Reals, bounds=(-100.0, 0.0))
                model.add_component(f"clt_{node_id}_term_{i}", term_var)
                terms.append(term_var)

                var_i_idx = scope_list[i]
                parent_local_idx = tree[i]

                if parent_local_idx == -1:
                    logp_i_0 = log_factors[i][0][0]
                    logp_i_1 = log_factors[i][1][0]
                    model.constraints.add(
                        term_var
                        == (1 - model.x[var_i_idx]) * logp_i_0
                        + model.x[var_i_idx] * logp_i_1
                    )
                else:
                    var_p_idx = scope_list[parent_local_idx]
                    x_i, x_p = model.x[var_i_idx], model.x[var_p_idx]

                    z_vars = {
                        (vi, vp): pyo.Var(within=pyo.Binary)
                        for vi in [0, 1]
                        for vp in [0, 1]
                    }
                    for key, var in z_vars.items():
                        model.add_component(
                            f"clt_{node_id}_z_{i}_{key[0]}{key[1]}", var
                        )

                    model.constraints.add(sum(z_vars.values()) == 1)
                    model.constraints.add(z_vars[(0, 0)] >= 1 - x_i - x_p)
                    model.constraints.add(z_vars[(0, 1)] >= (1 - x_i) + x_p - 1)
                    model.constraints.add(z_vars[(1, 0)] >= x_i + (1 - x_p) - 1)
                    model.constraints.add(z_vars[(1, 1)] >= x_i + x_p - 1)

                    log_f = log_factors[i]
                    model.constraints.add(
                        term_var
                        == sum(
                            z_vars[(vi, vp)] * log_f[vi][vp]
                            for vi in [0, 1]
                            for vp in [0, 1]
                        )
                    )

            model.constraints.add(model.log_prob[node_id] == sum(terms))

    # elif isinstance(node, FactorizationNode):
    #     child_ids = [get_id_func(c) for c in node.children]
    #     model.constraints.add(
    #         model.log_prob[node_id] == sum(model.log_prob[cid] for cid in child_ids)
    #     )
    #     for child in node.children:
    #         _add_node_constraints(model, child, get_id_func)

    # elif isinstance(node, DecisionNode):
    #     d_var = node.decision_scope
    #     logp0 = np.log(node.p0) if node.p0 > 1e-9 else -100.0
    #     logp1 = np.log(1.0 - node.p0) if (1.0 - node.p0) > 1e-9 else -100.0
    #     zero_id, one_id = get_id_func(node.zero_branch), get_id_func(node.one_branch)
    #     M = 100

    #     model.constraints.add(
    #         model.log_prob[node_id]
    #         <= logp0 + model.log_prob[zero_id] + M * model.x[d_var]
    #     )
    #     model.constraints.add(
    #         model.log_prob[node_id]
    #         >= logp0 + model.log_prob[zero_id] - M * model.x[d_var]
    #     )
    #     model.constraints.add(
    #         model.log_prob[node_id]
    #         <= logp1 + model.log_prob[one_id] + M * (1 - model.x[d_var])
    #     )
    #     model.constraints.add(
    #         model.log_prob[node_id]
    #         >= logp1 + model.log_prob[one_id] - M * (1 - model.x[d_var])
    #     )

    #     _add_node_constraints(model, node.zero_branch, get_id_func)
    #     _add_node_constraints(model, node.one_branch, get_id_func)

    # elif isinstance(node, MixtureNode):
    #     raise NotImplementedError(
    #         "General MixtureNodes cannot be linearly encoded without advanced techniques."
    #     )
    else:
        raise NotImplementedError("This node type is not implemented")


if __name__ == "__main__":
    # --- 1. Generate Synthetic Data ---
    # Create 100 samples of 4 binary variables
    # We'll make var 0 and 1 correlated, and var 2 and 3 correlated.
    np.random.seed(42)
    data = np.zeros((200, 4), dtype=int)
    # Group 1: X0 and X1 are often the same
    same_01 = np.random.randint(0, 2, 100)
    data[:100, 0] = same_01
    data[:100, 1] = same_01
    # Group 2: X2 and X3 are often different
    val_2 = np.random.randint(0, 2, 100)
    data[100:, 2] = val_2
    data[100:, 3] = 1 - val_2
    # Add some noise
    data[100:, 0] = np.random.randint(0, 2, 100)
    data[100:, 1] = np.random.randint(0, 2, 100)
    data[:100, 2] = np.random.randint(0, 2, 100)
    data[:100, 3] = np.random.randint(0, 2, 100)

    data = data.astype(np.float32)

    # --- 2. Learn the CNet using SPFlow ---
    # Define the types of variables (all Bernoulli for this binary data)
    # var_types = [create_bernoulli_leaf for _ in range(4)]
    var_types = Context(
        parametric_types=[
            Categorical,
            Categorical,
            Categorical,
            Categorical,
        ]
    ).add_domains(data)
    # Learn the CNet structure
    # spflow_cnet_structure = learn_cnet(data, var_types)
    # spflow_cnet_structure = learn_cnet(data, var_types, cond="random", min_instances_slice=100, min_features_slice=1)
    spflow_cnet_structure = learn_cnet(
        data, var_types, cond="random", min_instances_slice=20, min_features_slice=1
    )

    # Learn the parameters
    # learn_parametric(spflow_cnet_structure, data)

    # --- 3. Convert to our easy-to-read CNet class ---
    my_cnet = parse_cnet_as_decision_tree(spflow_cnet_structure)

    # --- 4. Print the readable CNet structure ---
    print("--- Readable CNet Structure ---")
    print(my_cnet)
    print("-" * 30)

    # --- 5. Perform Inference ---
    # Create a test data point
    test_point = np.array([0, 0, 1, 0], dtype=np.float32)

    # 5a. Inference using our CNetNode class
    prob_my_cnet = my_cnet.inference(test_point)
    print(f"Test Point: {test_point}")
    print(f"Probability from CNetNode class: {prob_my_cnet:.6f}")

    # 5b. Inference using SPFlow's built-in algorithm for verification
    # spflow expects a 2D array (batch of data)
    test_point_batch = test_point.reshape(1, -1).astype(int)
    # prob_spflow = spflow_inference.prob(spflow_cnet_structure, test_point_batch)[0]
    prob_spflow = log_likelihood(spflow_cnet_structure, test_point_batch)[0, 0]
    print(f"Probability from SPFlow        : {np.exp(prob_spflow):.6f}")

    # --- Verification ---
    assert np.isclose(
        prob_my_cnet, np.exp(prob_spflow), atol=1e-03
    ), "Inference results do not match!"
    print("\nSuccess: Inference results kinda match.")

    print("\n--- Building MILP Model ---")
    n_features = data.shape[1]
    milp_model, root_id = build_cnet_milp(my_cnet, n_features)
    # root_node_id =_get_node_id(my_cnet)
    # milp_model.cnet_constraint = pyo.Constraint(expr=milp_model.log_prob[root_id] >= -4.0)
    # milp_model.cnet_constraint0 = pyo.Constraint(expr=milp_model.x[0] == test_point[0])
    # milp_model.cnet_constraint1 = pyo.Constraint(expr=milp_model.x[1] == test_point[1])
    # milp_model.cnet_constraint2 = pyo.Constraint(expr=milp_model.x[2] == test_point[2])
    # milp_model.cnet_constraint3 = pyo.Constraint(expr=milp_model.x[3] == test_point[3])
    milp_model.obj = pyo.Objective(expr=1)
    print("MILP Model Built Successfully.")
    print("You can now solve this model with a MILP solver (e.g., CBC, Gurobi).")

    print("\n--- Solving MILP for Validation ---")
    # Fix the input variables to our test point
    for i in range(n_features):
        milp_model.x[i].fix(test_point[i])

    # You must have a MILP solver installed (e.g., cbc, glpk)
    # To install cbc: conda install -c conda-forge coincbc
    solver = pyo.SolverFactory("gurobi")
    results = solver.solve(milp_model, tee=False)  # tee=True to see solver output

    log_prob_milp = pyo.value(milp_model.log_prob[root_id])
    print(f"Log-Probability from MILP:     {np.exp(log_prob_milp):.6f}")

    # Final validation
    assert np.isclose(
        prob_my_cnet, np.exp(log_prob_milp)
    ), "MILP log-prob does not match direct calculation!"
    print("\nSuccess: MILP-computed log-probability matches the direct calculation.")
