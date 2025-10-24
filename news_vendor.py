import logging

from typing import Any
from SPN.SPN import SPN
from SPN.spn_enc import encode_spn

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from scipy.stats import norm, expon
from pyomo.opt import SolverStatus, TerminationCondition

logger = logging.getLogger(__name__)

MIO_EPS = 1e-4

# TODO Add costs and multiple news - maximize profit?
PROBLEMS = [
    {"name":"Newsvendor1", "kwargs":{
        "demand": {"distribution":"normal", "mean": 100, "std": 20},
    }}
]

def _sample_demand(n_samples: int, distribution: str, mean: float, std: float | None = None, seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)
    if distribution == "normal":
        if std is not None:
            demands = norm.rvs(loc=mean, scale=std, size=n_samples)
        else:
            demands = norm.rvs(loc=mean, size=n_samples)
    elif distribution == "exponential":
        demands = expon.rvs(scale=mean, size=n_samples)
    else:
        raise ValueError(f"Unkown distribution: \"{distribution}\"")

    return demands

def solve_model(name:str, demand: dict[str, Any], chance_variant: str, n_samples: int, p: float, seed:int | None = None, spn: SPN | None = None):
    demands = _sample_demand(n_samples=n_samples, seed=seed, **demand)

    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.NonNegativeReals)  # order quantity

    if chance_variant in ["sample_robust", "sample_average"]:
        model.nsamples = pyo.Set(initialize=range(n_samples))

        # Pyomo does not support min/max in expressions, so use constraints and auxiliary variables
        model.sold = pyo.Var(model.nsamples)
        model.leftover = pyo.Var(model.nsamples, domain=pyo.NonNegativeReals)

        # cover demand
        if chance_variant == "sample_robust":
            model.cover_demand = pyo.Constraint(model.nsamples, rule=lambda m, i: m.x >= demands[i])
            # model.covered = pyo.Param(model.nsamples, initialize=1)
        elif chance_variant == "sample_average":
            model.covered = pyo.Var(model.nsamples, domain=pyo.Binary)
            model.covers_demand = pyo.Constraint(model.nsamples, rule=lambda m, i: demands[i] - m.x <= (1 - m.covered[i]) * demands.max())
            model.covers_demand2 = pyo.Constraint(model.nsamples, rule=lambda m, i: m.x - demands[i] <= m.covered[i] * demands.max())
            model.cover_demand = pyo.Constraint(expr=sum([model.covered[i] for i in model.nsamples]) >= p*n_samples)

        # minimize what is left over
        model.leftover_max = pyo.Constraint(model.nsamples, rule=lambda m, i: m.leftover[i] >= m.x - demands[i])

        model.obj = pyo.Objective(expr=sum(model.leftover[i] for i in model.nsamples) / n_samples, sense=pyo.minimize)
    elif chance_variant == "SPN":
        # TODO make this (a TPM variant) a separate functions - better handling of p, better specificaion of expected out value...
        model.spn = pyo.Block()
        spn_outputs = encode_spn(
            spn,
            model.spn,
            # [None] + [model.x] + [0],
            [None] + [model.x] + [1],
            # [model.x] + [1],
            mio_epsilon=MIO_EPS,
            # sum_approx="upper",
            # sum_approx="lower",
            sum_approx="piecewise",
        )
        model.cover_demand = pyo.Constraint(expr=spn_outputs[spn.out_node_id] >= np.log(p))
        # model.cover_demand = pyo.Constraint(expr=spn_outputs[spn.out_node_id] <= np.log(p))
        model.leftover = pyo.Var(domain=pyo.NonNegativeReals)
        model.leftover_max = pyo.Constraint(expr=model.leftover >= model.x - np.mean(demands))

        model.obj = pyo.Objective(expr=model.leftover, sense=pyo.minimize)
        # model.obj = pyo.Objective(expr=model.leftover / np.mean(demands) - spn_outputs[spn.out_node_id] / np.abs(np.log(p)), sense=pyo.minimize)
    else:
        raise ValueError(f"Unknown variant: \"{chance_variant}\"")

    solver = pyo.SolverFactory('gurobi')
    result = solver.solve(model, tee=False, load_solutions=False)

    if result.solver.status == SolverStatus.ok and result.solver.termination_condition == TerminationCondition.optimal:
        model.solutions.load_from(result)
        if chance_variant == "SPN":
            print([spn_outputs[spn.out_node_id].value])
            print([model.leftover.value])
        # print([model.leftover[i].value for i in model.nsamples])
        return model.x.value
    elif result.solver.termination_condition in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]:
        logger.warning(f"Infeasible model {name}")
        write_iis(model, "IIS.ilp", solver="gurobi")
        return None
    else:
        logger.warning(f"Failed to solve model {name}")
        return None

def eval_solution(x: float, demand: dict[str, Any], n_samples:int=100_000, seed:int=0):
    demands = _sample_demand(n_samples=n_samples, seed=seed, **demand)
    return np.mean(np.maximum(x - demands, 0)), np.mean(demands > x)

def get_training_samples(demand: dict[str, Any], n_samples:int=1_000, seed:int | None = None):
    demands = _sample_demand(n_samples=n_samples, seed=seed, **demand)
    l, u = demands.min(), demands.max()
    # x = np.random.uniform(l, u, size=n_samples)
    x = np.linspace(l, u, 1000)#n_samples)
    res = demands.reshape(-1, 1) <= x.reshape(1, -1)
    grid_d, grid_x = np.meshgrid(demands, x, indexing='ij')
    samples = (grid_d.reshape(-1,1), grid_x.reshape(-1,1), res.reshape(-1,1))
    p_x = 1/(u - l)
    return samples, p_x
