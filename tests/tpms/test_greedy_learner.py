import numpy as np
import pytest

from stochopt.data.DataHandler import DataHandler
from stochopt.tpms.TreeTPM.learning import GreedyTopDownLearner
from stochopt.tpms.TreeTPM.nodes import DecisionNode, LeafNode
from stochopt.tpms.TreeTPM.tree_tpm import TreeTPM


@pytest.fixture
def simple_split_data():
    np.random.seed(42)
    # One feature with two clusters: [0, 1] and [10, 11]
    x_part1 = np.random.uniform(0, 1, (50, 1))
    x_part2 = np.random.uniform(10, 11, (50, 1))
    x = np.concatenate([x_part1, x_part2], axis=0)

    # Another feature that is constant
    y = np.zeros((100, 1))

    data = np.concatenate([x, y], axis=1)
    feat_names = ["x", "y"]
    return data, feat_names


def test_greedy_learner_basic(simple_split_data):
    data, feat_names = simple_split_data
    dh = DataHandler(data, feature_names=feat_names)

    learner = GreedyTopDownLearner(
        data_handler=dh, min_samples=10, max_depth=2, val_ratio=0.2
    )

    root = learner.learn(data)
    assert root is not None
    # Since there are two clear clusters in x, it should split at least once
    assert isinstance(root, DecisionNode)
    assert root.split_var == 0  # Split on x


def test_treetpm_greedy_train(simple_split_data):
    data, feat_names = simple_split_data
    dh = DataHandler(data, feature_names=feat_names)

    tpm = TreeTPM(dh)
    tpm.train_greedy_top_down(data, min_samples=10, max_depth=2)

    assert tpm.root is not None

    # Test log_probability
    test_val = np.array([0.5, 0.0])
    lp = tpm.log_probability(test_val)
    assert lp <= 0

    # Test point far from data (should have lower probability if smoothed, or -inf)
    far_val = np.array([5.0, 0.0])
    lp_far = tpm.log_probability(far_val)
    # Since it's between clusters, it depends on how bins are picked
    # But it should be valid
    assert lp_far <= lp


def test_categorical_split():
    np.random.seed(42)
    # Categorical feature with two groups: {0, 1} and {2, 3}
    x = np.random.choice([0, 1, 2, 3], size=(100, 1))
    # Correlation with another variable to make split beneficial
    y = np.array(
        [
            np.random.normal(0, 1) if val < 2 else np.random.normal(10, 1)
            for val in x.flatten()
        ]
    ).reshape(-1, 1)

    data = np.concatenate([x, y], axis=1)
    feat_names = ["cat", "cont"]
    categ_map: dict[int | str, list[int | str]] = {"cat": [0, 1, 2, 3]}

    dh = DataHandler(data, feature_names=feat_names, categ_map=categ_map)

    learner = GreedyTopDownLearner(
        data_handler=dh,
        min_samples=10,
        max_depth=2,
        max_branches=2,  # Force grouping if it were more values, but here we test basic split
    )

    root = learner.learn(data)
    assert isinstance(root, DecisionNode)
    assert root.split_var == 0  # Split on categorical x
    assert root.feature_type == "categorical"


def test_make_leaf_boundaries():
    np.random.seed(42)
    data = np.random.uniform(0, 1, (20, 2))
    feat_names = ["x1", "x2"]
    dh = DataHandler(data, feature_names=feat_names)

    learner = GreedyTopDownLearner(dh)
    # bounds are (min, max)
    bounds = [(0, 1), (0, 1)]
    leaf = learner._make_leaf(data, bounds)

    assert isinstance(leaf, LeafNode)
    assert leaf.histogram is not None

    # Test inference inside and outside bounds
    assert leaf.log_inference(np.array([0.5, 0.5])) > -100
    # Outside bounds might return MIN_LOG_PROB or similar
    # In JointHistogram.log_inference, it uses np.digitize which handles values outside range
    # but the logic depends on whether edges were extended.
    # learning.py:429-430 adds tiny margins.
