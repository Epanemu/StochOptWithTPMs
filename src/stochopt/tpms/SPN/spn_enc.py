from typing import Any, Union, List
import numpy as np
import pyomo.environ as pyo

from .spn import SPN, NodeType

# from scipy.special import logsumexp


# issues with binding variables in lambda functions for constraints
# trunk-ignore-all(ruff/B023)


# def contains_positive_logdensities(spn: SPN) -> bool:
#     """Checks whether there is a possibility that the SPN will have input log density on a sum node > 0

#     Args:
#         spn (SPN): The SPN in question

#     Returns:
#         bool: True if maximal log density of inputs to a sum node is greater than 0, False otherwise
#     """
#     node_maxes = {}
#     positive_dens = False
#     for node in spn.nodes:
#         if node.type in [
#             NodeType.LEAF,
#             NodeType.LEAF_BINARY,
#             NodeType.LEAF_CATEGORICAL,
#         ]:
#             node_maxes[node.id] = max(np.log(node.densities))
#         elif node.type == NodeType.PRODUCT:
#             node_maxes[node.id] = sum(node_maxes[n.id] for n in node.predecessors)
#         elif node.type == NodeType.SUM:
#             for pred, w in zip(node.predecessors, node.weights):
#                 if node_maxes[pred.id] + np.log(w) > 0:
#                     positive_dens = True
#             node_maxes[node.id] = logsumexp(
#                 np.array(node_maxes[n.id] for n in node.predecessors), node.weights
#             )
#         else:
#             raise ValueError("Unknown node type")
#     return positive_dens


def encode_histogram_as_pwl(
    breaks: list[float],
    vals: list[float],
    in_var: pyo.Var,
    out_var: pyo.Var,
    encoding_type: str = "LOG",
) -> pyo.Piecewise:
    """
    Encodes a histogram as a piecewise-linear function in Pyomo.

    Args:
        breaks: list[float]
            List of histogram breakpoints.
        vals: list[float]
            List of histogram density values (log-space).
        in_var: pyo.Var
            The input variable.
        out_var: pyo.Var
            The output variable.
        encoding_type: str
            The piecewise-linear representation type (e.g., "LOG", "SOS2").

    Returns:
        pyo.Piecewise: The Pyomo Piecewise component.
    """
    breakpoints = [breaks[0]]
    for b in breaks[1:-1]:
        breakpoints += [b, b]
    breakpoints.append(breaks[-1])

    doubled_vals = []
    for d in vals:
        doubled_vals += [d, d]

    return pyo.Piecewise(
        out_var,
        in_var,
        pw_pts=breakpoints,
        pw_constr_type="EQ",
        pw_repn=encoding_type,
        f_rule=list(doubled_vals),
    )


def encode_histogram(
    breaks: list[float],
    vals: list[float],
    in_var: pyo.Var,
    out_var: pyo.Var,
    mio_block: pyo.Block,
    mio_epsilon: float,
) -> None:
    """
    Encodes a histogram into a MIP formulation using binary indicator variables.
    """
    n_bins = len(vals)
    M = max(1, breaks[-1] - breaks[0])

    mio_block.bins = pyo.Set(initialize=list(range(n_bins)))
    mio_block.not_in_bin = pyo.Var(mio_block.bins, domain=pyo.Binary)
    mio_block.one_bin = pyo.Constraint(
        expr=sum(mio_block.not_in_bin[i] for i in mio_block.bins) == n_bins - 1
    )

    mio_block.lower = pyo.Constraint(
        mio_block.bins,
        rule=lambda b, bin_i: b.not_in_bin[bin_i] * M >= breaks[bin_i] - in_var,
    )
    mio_block.upper = pyo.Constraint(
        mio_block.bins,
        rule=lambda b, bin_i: b.not_in_bin[bin_i] * M >= in_var - breaks[bin_i + 1] + mio_epsilon,
    )

    mio_block.output = pyo.Constraint(
        expr=sum((1 - mio_block.not_in_bin[i]) * vals[i] for i in range(n_bins)) == out_var
    )


def logsumexp_approximation_mip(
    block: pyo.Block,
    x_vars: Union[pyo.Var, List[pyo.Var]],
    L: float = 0.001,
    K_exp: int = 5,
    K_log: int = 5,
    encoding_type_exp: str = "SOS2",
    encoding_type_log: str = "SOS2",
    **kwargs: Any,
) -> pyo.Var:
    """
    Adds a piecewise linear MIP approximation of logsumexp() function to a
    Pyomo model.

    Assumes that the max was already subtracted (at least one x_i=0 and all x_i <= 0).

    Args:
        block: pyo.Block
            The Pyomo block to which variables and constraints will be added.
        x_vars: Union[pyo.Var, List[pyo.Var]]
            A Pyomo indexed Var or list of VarData objects representing the 'x_i' inputs.
        L: float
            The lowest value to consider in exp, all lower will be 0 (default 0.001).
        K_exp: int
            The number of breakpoints for the exp(x_i) approximation (default 5).
        K_log: int
            The number of breakpoints for the log() approximation (default 5).
        encoding_type_exp: str
            The piecewise-linear representation type for exp (default "SOS2").
        encoding_type_log: str
            The piecewise-linear representation type for log (default "SOS2").
        **kwargs: Any
            Additional arguments.

    Returns:
        pyo.Var: The Pyomo variable 'lse' representing log(sum(exp(x_i))).
    """

    # --- Setup ---
    # Get the index set for x_vars
    if isinstance(x_vars, pyo.Var):
        x_set = x_vars.index_set()
    else:
        # If passed a list of VarData, create a set
        x_set = pyo.RangeSet(1, len(x_vars))
        # And map them to a dictionary for consistent access
        x_vars_dict = {i: x_vars[i - 1] for i in x_set}
        x_vars = x_vars_dict

    N = len(x_set)
    if N == 0:
        # Nothing to approximate
        raise ValueError("Nothing to approximate, x is empty")

    if K_exp < 2:
        raise ValueError("K_exp must be at least 2.")
    if K_log < 2:
        raise ValueError("K_log must be at least 2.")

    x_breakpoints = np.logspace(np.log(0.3), np.log(-np.log(L)), num=K_exp - 1, base=np.e).tolist()
    x_breakpoints.insert(0, 0)
    x_breakpoints = [-v for v in reversed(x_breakpoints)]
    x_breakpoints[0] = np.log(L)
    x_breakpoints[-1] = 0
    y_breakpoints = np.exp(x_breakpoints)

    block.exp_vals = pyo.Var(
        x_set, bounds=(0, 1), within=pyo.NonNegativeReals, doc="w_i >= exp(x_i)"
    )

    block.pw_constr = pyo.Piecewise(
        x_set,
        block.exp_vals,
        x_vars,
        pw_pts=[-500, x_breakpoints[0]] + list(x_breakpoints),
        pw_constr_type="EQ",
        pw_repn=encoding_type_exp,
        f_rule=[0, 0] + list(y_breakpoints),
        unbounded_domain_var=True,
    )

    x_points = np.logspace(np.log(1), np.log(N), num=K_log, base=np.e)
    x_points[0] = 1
    x_points[-1] = N
    y_points = [np.log(x) for x in x_points]

    block.sum = pyo.Var(bounds=(1, N), within=pyo.NonNegativeReals)
    block.sum_constr = pyo.Constraint(expr=sum(block.exp_vals[i] for i in x_set) == block.sum)

    block.lse = pyo.Var(bounds=(0, np.log(N)), within=pyo.NonNegativeReals)
    block.pw_constr2 = pyo.Piecewise(
        block.lse,
        block.sum,
        pw_pts=list(x_points),
        pw_constr_type="EQ",
        pw_repn=encoding_type_log,
        f_rule=y_points,
    )
    return block.lse


def encode_spn(
    spn: SPN,
    mio_spn: pyo.Block,
    input_vars: list[Any],
    leaf_encoding: str = "histogram",
    mio_epsilon: float = 1e-6,
    sum_approx: str = "lower",
    **kwargs: Any,
) -> pyo.Var:
    """
    Encodes the SPN into a MIP formulation computing log-likelihood.

    Args:
        spn: SPN
            The trained SPN model.
        mio_spn: pyo.Block
            The Pyomo block to add variables and constraints to.
        input_vars: list[Any]
            List of Pyomo variables representing inputs.
        leaf_encoding: str
            Method for encoding leaves ("histogram" or a pwl type, default "histogram").
        mio_epsilon: float
            Tolerance for sharp inequalities (default 1e-6).
        sum_approx: str
            Approximation for sum nodes ("upper", "lower", or "piecewise", default "lower").
        **kwargs: Any
            Additional encoding hyperparameters (e.g., K_exp, K_log).

    Returns:
        pyo.Var: Indexed Pyomo variable `node_out` containing log-likelihoods per node.
    """
    node_ids = [node.id for node in spn.nodes]

    # node_type_ids = {t: [] for t in NodeType}
    # for node in spn.nodes:
    #     node_type_ids[node.type].append(node.id)
    #     node_ids.append(node.id)

    # mio_spn.node_type_sets = {
    #     t: pyo.Set(initialize=ids) for t, ids in node_type_ids.items()
    # }
    mio_spn.node_set = pyo.Set(initialize=node_ids)

    # values are log likelihoods - almost always negative - except in narrow peaks that go above 1
    # mio_spn.node_out = pyo.Var(mio_spn.node_set, within=pyo.NonPositiveReals)
    mio_spn.node_out = pyo.Var(mio_spn.node_set, within=pyo.Reals)
    # print(mio_spn.node_set, node_ids)

    # TODO nodes as blocks
    for node in spn.nodes:
        if node.type == NodeType.LEAF:
            # in_var = mio_spn.input[node.scope]
            in_var = input_vars[node.scope]

            # lb, ub = in_var.bounds

            # if lb is None or ub is None:
            #     raise AssertionError("SPN input variables must have fixed bounds.")

            # density_vals = node.densities
            # breakpoints = node.breaks
            # # if histogram is narrower than the input bounds
            # if lb < breakpoints[0]:
            #     breakpoints = [lb] + breakpoints
            #     density_vals = [spn.min_density] + density_vals
            # if ub > breakpoints[-1]:
            #     breakpoints = breakpoints + [ub]
            #     density_vals = density_vals + [spn.min_density]

            breakpoints, densities = node.get_breaks_densities(span_all=True)
            log_densities = [np.log(d) for d in densities]

            if leaf_encoding == "histogram":
                hist_block = pyo.Block()
                mio_spn.add_component(f"HistLeaf{node.id}", hist_block)
                encode_histogram(
                    breakpoints,
                    log_densities,
                    in_var,
                    mio_spn.node_out[node.id],
                    hist_block,
                    mio_epsilon,  # * spn.input_scale(node.scope),
                )
            else:
                pw_constr = encode_histogram_as_pwl(
                    list(breakpoints),
                    list(log_densities),
                    in_var,
                    mio_spn.node_out[node.id],
                    leaf_encoding,
                )
                mio_spn.add_component(f"PWLeaf{node.id}", pw_constr)

        elif node.type == NodeType.LEAF_CATEGORICAL:
            dens_ll = np.log(node.densities)
            in_vars = input_vars[node.scope]

            if isinstance(in_vars, pyo.Var):
                in_vars = [in_vars[k] for k in sorted(in_vars.keys())]

            if len(in_vars) <= 1:  # TODO make this more direct, not fixed to 1
                raise ValueError(
                    "The categorical values should be passed as a list of binary variables, representing a one-hot encoding."
                )
            # Do checks that the vars are binary?
            # check if the histogram always contains all values?
            # TODO use expr parameter of Constraint maker, instead of the rule=lambdas?

            constr = pyo.Constraint(
                rule=lambda b: (
                    b.node_out[node.id] == sum(var * dens for var, dens in zip(in_vars, dens_ll))
                )
            )
            mio_spn.add_component(f"CategLeaf{node.id}", constr)
        elif node.type == NodeType.LEAF_BINARY:
            constr = pyo.Constraint(
                rule=lambda b: (
                    b.node_out[node.id]
                    == (1 - input_vars[node.scope]) * np.log(node.densities[0])
                    + input_vars[node.scope] * np.log(node.densities[1])
                )
            )
            mio_spn.add_component(f"BinLeaf{node.id}", constr)
        elif node.type == NodeType.PRODUCT:
            constr = pyo.Constraint(
                rule=lambda b: (
                    b.node_out[node.id] == sum(b.node_out[ch.id] for ch in node.predecessors)
                )
            )
            mio_spn.add_component(f"ProdConstr{node.id}", constr)
        elif node.type == NodeType.SUM:
            # Sum node - approximated in log domain by max
            preds_set = [ch.id for ch in node.predecessors]
            n_preds = len(node.predecessors)
            weights = {ch.id: w for ch, w in zip(node.predecessors, node.weights)}

            # TODO testing this, if it works well, fit it in correctly
            M_sum = 100  # hope this is enough
            slack_inds = pyo.Var(preds_set, domain=pyo.Binary)
            mio_spn.add_component(f"SumSlackIndicators{node.id}", slack_inds)
            if sum_approx == "lower":
                slacking = pyo.Constraint(
                    preds_set,
                    rule=lambda b, pre_id: (
                        b.node_out[node.id]
                        <= b.node_out[pre_id] + np.log(weights[pre_id]) + M_sum * slack_inds[pre_id]
                    ),
                )
            elif sum_approx == "upper":
                slacking = pyo.Constraint(
                    preds_set,
                    rule=lambda b, pre_id: (
                        b.node_out[node.id]
                        <= b.node_out[pre_id]
                        + (  # approximate by the bound on logsumexp
                            np.log(weights[pre_id] * n_preds)
                            if weights[pre_id] * n_preds < 1
                            else 0  # or by using the fact it is a mixture
                        )
                        + M_sum * slack_inds[pre_id]
                    ),
                )
                opposite_slacking = pyo.Constraint(
                    preds_set,
                    rule=lambda b, pre_id: (
                        b.node_out[node.id]
                        >= b.node_out[pre_id]
                        + (  # approximate by the bound on logsumexp
                            np.log(weights[pre_id] * n_preds)
                            if weights[pre_id] * n_preds < 1
                            else 0  # or by using the fact it is a mixture
                        )
                        - M_sum * slack_inds[pre_id]
                    ),
                )
                mio_spn.add_component(f"SumOppSlackConstr{node.id}", opposite_slacking)
            elif sum_approx == "piecewise":
                max_value = pyo.Var(within=pyo.Reals)
                mio_spn.add_component(f"SumMax{node.id}", max_value)

                slacking = pyo.Constraint(
                    preds_set,
                    rule=lambda b, pre_id: (
                        max_value
                        <= b.node_out[pre_id] + np.log(weights[pre_id]) + M_sum * slack_inds[pre_id]
                    ),
                )

                logsumexp_block = pyo.Block()
                mio_spn.add_component(f"LSE_{node.id}", logsumexp_block)

                sub_vars = pyo.Var(preds_set, bounds=(-np.inf, 0))
                mio_spn.add_component(f"sub{node.id}", sub_vars)
                subtraction = pyo.Constraint(
                    preds_set,
                    rule=lambda b, pre_id: sub_vars[pre_id]
                    == b.node_out[pre_id] + np.log(weights[pre_id]) - max_value,
                )
                mio_spn.add_component(f"MaxSub{node.id}", subtraction)
                lse = logsumexp_approximation_mip(logsumexp_block, sub_vars, **kwargs)
                lse_out = pyo.Constraint(expr=mio_spn.node_out[node.id] == max_value + lse)
                mio_spn.add_component(f"LSE_node_out{node.id}", lse_out)
            else:
                raise ValueError('sum_approx must be one of ["upper", "lower", "piecewise"]')
            mio_spn.add_component(f"SumSlackConstr{node.id}", slacking)
            one_tight = pyo.Constraint(expr=sum(slack_inds[i] for i in preds_set) == n_preds - 1)
            mio_spn.add_component(f"SumTightConstr{node.id}", one_tight)

            # implemented using SOS1 constraints, see here: https://www.gurobi.com/documentation/current/refman/general_constraints.html
            # slacks = pyo.Var(preds_set, domain=pyo.NonNegativeReals)
            # mio_spn.add_component(f"SumSlackVars{node.id}", slacks)
            # if sum_approx == "lower":
            #     slacking = pyo.Constraint(
            #         preds_set,
            #         rule=lambda b, pre_id: (
            #             b.node_out[node.id]
            #             == b.node_out[pre_id] + np.log(weights[pre_id]) + slacks[pre_id]
            #         ),
            #     )
            # elif sum_approx == "upper":
            #     slacking = pyo.Constraint(
            #         preds_set,
            #         rule=lambda b, pre_id: (
            #             b.node_out[node.id]
            #             == b.node_out[pre_id]
            #             + (  # approximate by the bound on logsumexp
            #                 np.log(weights[pre_id] * n_preds)
            #                 if weights[pre_id] * n_preds < 1
            #                 else 0  # or by using the fact it is a mixture
            #             )
            #             + slacks[pre_id]
            #         ),
            #     )
            # else:
            #     raise ValueError('sum_approx must be one of ["upper", "lower"]')
            # mio_spn.add_component(f"SumSlackConstr{node.id}", slacking)

            # indicators = pyo.Var(preds_set, domain=pyo.Binary)
            # mio_spn.add_component(f"SumIndicators{node.id}", indicators)
            # indicating = pyo.Constraint(
            #     rule=lambda b: (
            #         sum(b.component(f"SumIndicators{node.id}")[i] for i in preds_set)
            #         == 1
            #     )
            # )
            # mio_spn.add_component(f"SumIndicatorConstr{node.id}", indicating)

            # sos = pyo.SOSConstraint(
            #     preds_set,
            #     rule=lambda b, pred: [
            #         b.component(f"SumIndicators{node.id}")[pred],
            #         b.component(f"SumSlackVars{node.id}")[pred],
            #     ],
            #     sos=1,
            # )
            # mio_spn.add_component(f"SumSosConstr{node.id}", sos)

    return mio_spn.node_out
