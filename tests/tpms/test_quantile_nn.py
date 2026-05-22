from unittest.mock import patch

import numpy as np
import pyomo.environ as pyo
import pytest
import torch

from stochopt.problem.newsvendor import NewsvendorProblem
from stochopt.quantile.quantile_nn import PinballLoss, QuantileNN


@pytest.fixture
def dummy_problem():
    demand_params = {"mean": [10.0, 15.0], "std": [2.0, 3.0]}
    costs = np.array([5.0, 5.0])
    prices = np.array([10.0, 10.0])

    prob = NewsvendorProblem(
        n_products=2,
        costs=costs,
        prices=prices,
        demand_dist="normal",
        demand_params=demand_params,
        x_density_type="uniform",
    )
    return prob


def test_pinball_loss():
    # Loss for predictions and targets: target - pred = error
    # L_q = max(q * error, (q - 1) * error)

    # Target = 1.0, Pred = 0.0 -> Error = 1.0 (positive error)
    # For q = 0.05: max(0.05 * 1.0, -0.95 * 1.0) = 0.05
    preds_pos = torch.tensor([0.0])
    targets_pos = torch.tensor([1.0])
    loss_pos = PinballLoss(q=0.05)(preds_pos, targets_pos)
    assert torch.isclose(loss_pos, torch.tensor(0.05))

    # Target = -1.0, Pred = 0.0 -> Error = -1.0 (negative error)
    # For q = 0.05: max(0.05 * -1.0, -0.95 * -1.0) = max(-0.05, 0.95) = 0.95
    preds_neg = torch.tensor([0.0])
    targets_neg = torch.tensor([-1.0])
    loss_neg = PinballLoss(q=0.05)(preds_neg, targets_neg)
    assert torch.isclose(loss_neg, torch.tensor(0.95))


@patch("mlflow.log_metric")
def test_hidden_size_factors(mock_log_metric, dummy_problem):
    """Test that hidden_size_factors correctly configure the network architecture."""
    qnn = QuantileNN()
    train_samples = np.array([[10.0, 15.0]])
    hidden_size_factors = [1.0, 2.0, 0.5]

    qnn.train(
        dummy_problem,
        train_samples,
        epochs=1,
        batch_size=1,
        val_size=2,
        hidden_size_factors=hidden_size_factors,
        min_hidden_size=2,
        max_hidden_size=100,
    )

    assert qnn.model is not None
    layers = list(qnn.model.network.children())

    # 2 inputs, factor 1.0 -> 2 hidden
    assert layers[0].out_features == 2
    # factor 2.0 -> 4 hidden
    assert layers[2].out_features == 4
    # factor 0.5 -> 1 hidden, but min_hidden_size=2 -> 2 hidden
    assert layers[4].out_features == 2


@patch("mlflow.log_metric")
def test_quantile_nn_predict(mock_log_metric, dummy_problem):
    """Test that QuantileNN predicts the quantile of satisfaction margin."""
    qnn = QuantileNN()
    train_samples = np.array([[10.0, 15.0], [9.0, 14.0]])

    qnn.train(
        dummy_problem,
        train_samples,
        epochs=2,
        batch_size=2,
        val_size=2,
    )

    x_test = np.array([[12.0, 17.0], [8.0, 13.0]])
    preds = qnn.predict_quantile(x_test)
    assert preds.shape == (2,)


@patch("mlflow.log_metric")
def test_quantile_nn_encode(mock_log_metric, dummy_problem):
    """Test OMLT encoding into Pyomo block."""
    qnn = QuantileNN()
    train_samples = np.array([[10.0, 15.0]])

    qnn.train(
        dummy_problem,
        train_samples,
        epochs=1,
        batch_size=2,
        val_size=2,
    )

    model = pyo.ConcreteModel()
    model.x = pyo.Var(range(2), domain=pyo.Reals)
    model.nn_block = pyo.Block()

    inputs = [model.x[i] for i in range(2)]
    margin_output = qnn.encode(model.nn_block, inputs)

    assert margin_output is not None
    assert hasattr(model.nn_block, "nn")
