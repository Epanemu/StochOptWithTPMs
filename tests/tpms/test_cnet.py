import numpy as np
import pytest

from stochopt.data.DataHandler import DataHandler
from stochopt.tpms.CNet.cnet_learning import learn_cnet_tree
from stochopt.tpms.cnet_tpm import CNetTPM


@pytest.fixture
def synthetic_binary_data():
    np.random.seed(42)
    N = 500
    # X0 -> X1
    x0 = np.random.randint(0, 2, N).astype(np.float64)
    x1 = np.array(
        [np.random.choice([0, 1], p=[0.9, 0.1] if v == 0 else [0.1, 0.9]) for v in x0],
        dtype=np.float64,
    )
    # X2 independent
    x2 = np.random.randint(0, 2, N).astype(np.float64)

    data = np.stack([x0, x1, x2], axis=1)
    feat_names = ["x0", "x1", "x2"]
    categ_map = {"x0": [0, 1], "x1": [0, 1], "x2": [0, 1]}

    return data, feat_names, categ_map


def test_cnet_learning_basic(synthetic_binary_data):
    data, feat_names, categ_map = synthetic_binary_data
    dh = DataHandler(data, feature_names=feat_names, categ_map=categ_map)

    cnet = learn_cnet_tree(dh, data, min_instances_slice=20)
    assert cnet is not None

    # Check inference
    test_x = np.array([0, 0, 0])
    lp = cnet.log_inference(test_x)
    assert isinstance(lp, float)
    assert lp <= 0


def test_cnet_tpm_integration(synthetic_binary_data):
    data, feat_names, categ_map = synthetic_binary_data
    dh = DataHandler(data, feature_names=feat_names, categ_map=categ_map)

    tpm = CNetTPM(dh)
    tpm.train(data, n_bins=2)

    # Test log_probability
    test_row = data[0]
    lp = tpm.log_probability(test_row)
    assert lp <= 0

    # Let's test the encoding (which involves marginalization)
    import pyomo.environ as pyo

    model = pyo.ConcreteModel()
    model.b = pyo.Block()
    # Inputs for [x0, x1, None]
    inputs = [0, 0, None]

    # We need to set up the tpm._discretized_data which happens during train()
    # Then encode should work
    lp_var = tpm.encode(model.b, inputs)
    assert isinstance(lp_var, pyo.Var)

    # Test marginalized log_probability
    # Note: Marginalized CNet training happens on the fly in some versions or needs pre-training
    # In cnet_tpm.py, log_probability handles marginalization if nones are present
    marg_sample = np.array([0, 0, None])
    # For CNetTPM, it seems we need to have _kept_indices set up if we use nones?
    lp_marg = tpm.log_probability(marg_sample)
    assert isinstance(lp_marg, float)
    assert lp_marg <= 0


def test_cnet_discretization_logic():
    # Test continuous data discretization
    np.random.seed(42)
    data = np.random.normal(0, 1, (100, 2))
    feat_names = ["c0", "c1"]
    dh = DataHandler(data, feature_names=feat_names)

    tpm = CNetTPM(dh)
    tpm.train(data, n_bins=5, discretization_method="quantile")

    assert len(tpm.discretization_info) == 2
    assert tpm.discretization_info[0]["n_bins"] == 5

    # Check if discretized data is within bounds
    d_data = tpm._discretized_data
    assert np.all(d_data >= 0)
    assert np.all(d_data < 5)
