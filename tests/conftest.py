"""
Pytest configuration and shared fixtures for StochOpt tests.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def small_config():
    """Minimal configuration for fast tests."""
    config = OmegaConf.create(
        {
            "seed": 42,
            "risk_level": 0.05,
            "solver": "appsi_highs",  # HiGHS solver
            "samples": {
                "train": 10,
                "opt": 5,
                "test": 20,
                "validation": 15,
                "train_decisions": 10,
            },
            "mlflow": {
                "tracking_uri": "sqlite:///test_mlflow.db",
                "experiment_name": "test_experiment",
            },
            "problem": {
                "_target_": "stochopt.problem.newsvendor.NewsvendorProblem",
                "n_products": 1,
                "costs": [1.0],
                "prices": [2.0],
                "demand_dist": "normal",
                "demand_params": {"mean": [100.0], "std": [20.0]},
                "density_type": "uniform",
                "correlated": False,
            },
            "method": {"name": "robust"},
            "time_limit": 60,  # 1 minute
        }
    )
    return config


@pytest.fixture
def newsvendor_config():
    """Configuration for newsvendor problem tests."""
    return OmegaConf.create(
        {
            "_target_": "stochopt.problem.newsvendor.NewsvendorProblem",
            "n_products": 2,
            "costs": [1.0, 1.5],
            "prices": [2.0, 3.0],
            "demand_dist": "normal",
            "demand_params": {"mean": [50.0, 75.0], "std": [10.0, 15.0]},
            "density_type": "uniform",
        }
    )


@pytest.fixture
def newsvendor_problem(newsvendor_config):
    """Create a newsvendor problem instance."""
    return instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")


@pytest.fixture
def sample_demands():
    """Generate sample demand data."""
    np.random.seed(42)
    return np.random.normal(loc=100, scale=20, size=(10, 2))


@pytest.fixture
def sample_orders():
    """Generate sample order data."""
    np.random.seed(42)
    return np.random.uniform(low=50, high=150, size=(10, 1))


@pytest.fixture
def temp_mlflow_dir(test_data_dir):
    """Temporary MLflow tracking directory."""
    mlflow_dir = test_data_dir / "mlruns"
    mlflow_dir.mkdir(exist_ok=True)
    return str(mlflow_dir)


@pytest.fixture(autouse=True)
def cleanup_mlflow():
    """Clean up MLflow after each test."""
    yield
    # MLflow cleanup is handled by temp directories
    pass


@pytest.fixture
def test_config_path(tmp_path):
    """Create a temporary config file."""
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    return config_dir
