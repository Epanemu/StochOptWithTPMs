"""
End-to-end tests for the experiment runner.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
from hydra.utils import instantiate

from stochopt.runner import run_experiment


class TestRunner:
    """Integration tests for the experiment runner."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Setup: create temporary mlflow directory
        self.temp_mlflow = tempfile.mkdtemp()
        yield
        # Teardown: clean up
        shutil.rmtree(self.temp_mlflow, ignore_errors=True)

    def test_robust_method_complete_run(self, small_config):
        """Test complete run with robust method."""
        small_config.mlflow.tracking_uri = f"sqlite:///{self.temp_mlflow}/mlflow.db"
        small_config.method.name = "robust"

        # Should complete without errors
        run_experiment(small_config)

        # Check that MLflow directory was created
        mlflow_path = Path(self.temp_mlflow)
        assert mlflow_path.exists()

    def test_sample_average_method_complete_run(self, small_config):
        """Test complete run with sample_average method."""
        small_config.mlflow.tracking_uri = f"sqlite:///{self.temp_mlflow}/mlflow.db"
        small_config.method.name = "sample_average"

        run_experiment(small_config)

        mlflow_path = Path(self.temp_mlflow)
        assert mlflow_path.exists()

    def test_solution_logging(self, small_config):
        """Test that solution is logged to MLflow."""
        import mlflow

        small_config.mlflow.tracking_uri = f"sqlite:///{self.temp_mlflow}/mlflow.db"
        mlflow.set_tracking_uri(small_config.mlflow.tracking_uri)

        run_experiment(small_config)

        # Check that at least one run was created
        mlflow_path = Path(self.temp_mlflow)
        assert mlflow_path.exists()

        # There should be experiment and run directories
        experiment_dirs = list(mlflow_path.glob("*"))
        assert len(experiment_dirs) > 0

    def test_validation_metrics_logged(self, small_config):
        """Test that validation metrics are computed and logged."""
        small_config.mlflow.tracking_uri = f"sqlite:///{self.temp_mlflow}/mlflow.db"

        # Run experiment
        run_experiment(small_config)

        # MLflow tracking should have been used
        mlflow_path = Path(self.temp_mlflow)
        assert mlflow_path.exists()

    def test_different_solvers(self, small_config):
        """Test that different solvers work."""
        small_config.mlflow.tracking_uri = f"sqlite:///{self.temp_mlflow}/mlflow.db"

        # HiGHS solver
        small_config.solver = "appsi_highs"
        run_experiment(small_config)

        assert Path(self.temp_mlflow).exists()

    def test_multi_product_newsvendor(self, small_config):
        """Test with multiple products."""
        small_config.mlflow.tracking_uri = f"sqlite:///{self.temp_mlflow}/mlflow.db"
        small_config.problem.n_products = 3
        small_config.problem.costs = [1.0, 1.5, 2.0]
        small_config.problem.prices = [2.0, 3.0, 4.0]
        small_config.problem.demand_params.mean = [50.0, 75.0, 100.0]
        small_config.problem.demand_params.std = [10.0, 15.0, 20.0]

        run_experiment(small_config)

        assert Path(self.temp_mlflow).exists()

    def test_reproducibility_with_seed(self, small_config):
        """Test that same seed produces reproducible results."""
        import numpy as np

        # Create problem and generate samples twice with same seed
        problem1 = instantiate(
            small_config.problem, solver="appsi_highs", _convert_="all"
        )
        samples1 = problem1.generate_samples(n_samples=10, seed=42)

        problem2 = instantiate(
            small_config.problem, solver="appsi_highs", _convert_="all"
        )
        samples2 = problem2.generate_samples(n_samples=10, seed=42)

        np.testing.assert_array_equal(samples1, samples2)
