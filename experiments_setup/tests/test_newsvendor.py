"""
Unit tests for the newsvendor problem implementation.
"""
import pytest
import numpy as np
from hydra.utils import instantiate


class TestNewsvendorProblem:
    """Test cases for NewsvendorProblem class."""

    def test_init(self, newsvendor_config):
        """Test problem initialization."""
        problem = instantiate(newsvendor_config, solver="appsi_highs")
        assert problem.n_products == 2
        assert len(problem.costs) == 2
        assert len(problem.prices) == 2
        assert problem.demand_dist == "normal"
        assert problem.x_density_type == "uniform"

    def test_generate_samples_normal(self, newsvendor_problem):
        """Test sample generation with normal distribution."""
        samples = newsvendor_problem.generate_samples(n_samples=100, seed=42)

        assert samples.shape == (100, 2)
        assert np.all(np.isfinite(samples))

        # Check approximate statistics
        means = np.mean(samples, axis=0)
        stds = np.std(samples, axis=0)
        assert np.allclose(means, [50.0, 75.0], atol=5.0)
        assert np.allclose(stds, [10.0, 15.0], atol=3.0)

    def test_generate_samples_exponential(self, newsvendor_config):
        """Test sample generation with exponential distribution."""
        newsvendor_config.demand_dist = "exponential"
        problem = instantiate(newsvendor_config, solver="appsi_highs")

        samples = problem.generate_samples(n_samples=100, seed=42)

        assert samples.shape == (100, 2)
        assert np.all(samples >= 0)  # Exponential is non-negative

    def test_generate_decision_samples(self, newsvendor_problem, sample_demands):
        """Test decision variable sample generation."""
        x_samples = newsvendor_problem.generate_decision_samples(
            n_samples=10,
            seed=42,
            xi=sample_demands[:, :2]  # Use first 2 columns for 2 products
        )

        assert x_samples.shape == (10, 2)
        assert np.all(x_samples >= 0)  # Order quantities are non-negative

    def test_compute_satisfaction(self, newsvendor_problem):
        """Test satisfaction computation."""
        # Create test data
        demands = np.array([[50, 60], [70, 80]])
        orders = np.array([[60, 70], [65, 75]])  # First satisfies, second doesn't

        satisfaction = newsvendor_problem.compute_satisfaction(demands, orders)

        assert satisfaction.shape == (2, 1)
        assert satisfaction[0, 0] == True  # Both products satisfied
        assert satisfaction[1, 0] == False  # Second product not satisfied

    def test_check_satisfaction(self, newsvendor_problem):
        """Test satisfaction checking for a solution."""
        # Create test scenarios
        scenarios = np.array([[50, 60], [70, 80], [55, 65]])
        solution = np.array([75, 85])  # Should satisfy all

        satisfied = newsvendor_problem.check_satisfaction(solution, scenarios)

        assert satisfied.shape == (3,)
        assert np.all(satisfied == True)  # All scenarios satisfied

    def test_get_feature_names(self, newsvendor_problem):
        """Test feature name generation."""
        xi_names, x_names, sat_name = newsvendor_problem.get_feature_names()

        assert len(xi_names) == 2
        assert len(x_names) == 2
        assert sat_name == "sat"
        assert xi_names == ["demand_0", "demand_1"]
        assert x_names == ["order_0", "order_1"]

    def test_generate_tpm_data(self, newsvendor_problem):
        """Test TPM data generation."""
        train_samples = newsvendor_problem.generate_samples(n_samples=20, seed=42)

        tpm_data, feat_names = newsvendor_problem.generate_tpm_data(n_decisions=20, train_samples=train_samples, seed=42)

        # Check shape: [demands (2), orders (2), sat (1)]
        assert tpm_data.shape == (400, 5)
        assert len(feat_names) == 5
        assert feat_names[-1] == "sat"

        # Check satisfaction column is binary
        assert np.all(np.isin(tpm_data[:, -1], [0.0, 1.0]))

    def test_build_model_robust(self, newsvendor_problem):
        """Test model building with robust method."""
        scenarios = newsvendor_problem.generate_samples(n_samples=5, seed=42)

        model = newsvendor_problem.build_model(
            method="robust",
            scenarios=scenarios,
            risk_level=0.05
        )

        assert model is not None
        assert hasattr(model, 'x')
        assert hasattr(model, 'obj')
        assert hasattr(model, 'robust_constr')

    def test_build_model_sample_average(self, newsvendor_problem):
        """Test model building with sample average method."""
        scenarios = newsvendor_problem.generate_samples(n_samples=10, seed=42)

        model = newsvendor_problem.build_model(
            method="sample_average",
            scenarios=scenarios,
            risk_level=0.05
        )

        assert model is not None
        assert hasattr(model, 'x')
        assert hasattr(model, 'y')
        assert hasattr(model, 'chance_constr')
        assert hasattr(model, 'prob_constr')

    def test_solve_robust_method(self, newsvendor_problem):
        """Test solving with robust method."""
        scenarios = newsvendor_problem.generate_samples(n_samples=5, seed=42)

        newsvendor_problem.build_model(
            method="robust",
            scenarios=scenarios
        )

        result = newsvendor_problem.solve()

        assert result is not None
        assert "status" in result

    def test_get_solution(self, newsvendor_problem):
        """Test extracting solution from solved model."""
        scenarios = newsvendor_problem.generate_samples(n_samples=5, seed=42)

        newsvendor_problem.build_model(
            method="robust",
            scenarios=scenarios
        )
        newsvendor_problem.solve()

        solution = newsvendor_problem.get_solution()

        assert solution.shape == (2,)
        assert np.all(solution >= 0)  # Non-negative order quantities


class TestReproducibility:
    """Test reproducibility of data generation with seeds."""

    def test_generate_samples_normal_reproducibility(self, newsvendor_config):
        """Test that normal distribution samples are reproducible with same seed."""
        newsvendor_config.demand_dist = "normal"
        problem1 = instantiate(newsvendor_config, solver="appsi_highs")
        problem2 = instantiate(newsvendor_config, solver="appsi_highs")

        samples1 = problem1.generate_samples(n_samples=20, seed=42)
        samples2 = problem2.generate_samples(n_samples=20, seed=42)

        np.testing.assert_array_equal(samples1, samples2)

    def test_generate_samples_normal_different_seeds(self, newsvendor_config):
        """Test that different seeds produce different samples."""
        newsvendor_config.demand_dist = "normal"
        problem = instantiate(newsvendor_config, solver="appsi_highs")

        samples1 = problem.generate_samples(n_samples=20, seed=42)
        samples2 = problem.generate_samples(n_samples=20, seed=100)

        assert not np.array_equal(samples1, samples2)

    def test_generate_samples_exponential_reproducibility(self, newsvendor_config):
        """Test that exponential distribution samples are reproducible with same seed."""
        newsvendor_config.demand_dist = "exponential"
        problem1 = instantiate(newsvendor_config, solver="appsi_highs")
        problem2 = instantiate(newsvendor_config, solver="appsi_highs")

        samples1 = problem1.generate_samples(n_samples=20, seed=42)
        samples2 = problem2.generate_samples(n_samples=20, seed=42)

        np.testing.assert_array_equal(samples1, samples2)

    def test_generate_samples_exponential_different_seeds(self, newsvendor_config):
        """Test that different seeds produce different exponential samples."""
        newsvendor_config.demand_dist = "exponential"
        problem = instantiate(newsvendor_config, solver="appsi_highs")

        samples1 = problem.generate_samples(n_samples=20, seed=42)
        samples2 = problem.generate_samples(n_samples=20, seed=100)

        assert not np.array_equal(samples1, samples2)

    def test_decision_samples_with_demands_reproducibility(self, newsvendor_problem):
        """Test that decision samples are reproducible when demands are provided."""
        # Generate demand samples
        demands = newsvendor_problem.generate_samples(n_samples=15, seed=10)

        # Generate decision samples twice with same seed and demands
        x_samples1 = newsvendor_problem.generate_decision_samples(
            n_samples=15, seed=42, demands=demands
        )
        x_samples2 = newsvendor_problem.generate_decision_samples(
            n_samples=15, seed=42, demands=demands
        )

        np.testing.assert_array_equal(x_samples1, x_samples2)

    def test_decision_samples_without_demands_reproducibility(self, newsvendor_problem):
        """Test that decision samples are reproducible without demands."""
        x_samples1 = newsvendor_problem.generate_decision_samples(n_samples=15, seed=42)
        x_samples2 = newsvendor_problem.generate_decision_samples(n_samples=15, seed=42)

        np.testing.assert_array_equal(x_samples1, x_samples2)

    def test_decision_samples_different_seeds(self, newsvendor_problem):
        """Test that different seeds produce different decision samples."""
        demands = newsvendor_problem.generate_samples(n_samples=15, seed=10)

        x_samples1 = newsvendor_problem.generate_decision_samples(
            n_samples=15, seed=42, demands=demands
        )
        x_samples2 = newsvendor_problem.generate_decision_samples(
            n_samples=15, seed=100, demands=demands
        )

        assert not np.array_equal(x_samples1, x_samples2)

    def test_decision_samples_different_demand_bounds(self, newsvendor_problem):
        """Test that different demand bounds affect decision sample range."""
        demands1 = newsvendor_problem.generate_samples(n_samples=15, seed=10)
        demands2 = newsvendor_problem.generate_samples(n_samples=15, seed=20)

        x_samples1 = newsvendor_problem.generate_decision_samples(
            n_samples=15, seed=42, demands=demands1
        )
        x_samples2 = newsvendor_problem.generate_decision_samples(
            n_samples=15, seed=42, demands=demands2
        )
        # Same seed but different demands should produce different samples
        assert not np.array_equal(x_samples1, x_samples2)

        x_samples1 = newsvendor_problem.generate_decision_samples(
            n_samples=15, seed=42
        )
        x_samples2 = newsvendor_problem.generate_decision_samples(
            n_samples=15, seed=42
        )
        # When demands are not provided, should produce same samples
        assert np.array_equal(x_samples1, x_samples2)

    def test_tpm_data_generation_reproducibility(self, newsvendor_problem):
        """Test that TPM data generation is reproducible."""
        train_samples = newsvendor_problem.generate_samples(n_samples=20, seed=10)

        tpm_data1, feat_names1 = newsvendor_problem.generate_tpm_data(n_decisions=20, train_samples=train_samples, seed=42)
        tpm_data2, feat_names2 = newsvendor_problem.generate_tpm_data(n_decisions=20, train_samples=train_samples, seed=42)

        np.testing.assert_array_equal(tpm_data1, tpm_data2)
        assert feat_names1 == feat_names2

    def test_multi_product_reproducibility(self, newsvendor_config):
        """Test reproducibility with multiple products and distributions."""
        newsvendor_config.n_products = 3
        newsvendor_config.costs = [1.0, 1.5, 2.0]
        newsvendor_config.prices = [2.0, 3.0, 4.0]
        newsvendor_config.demand_params.mean = [50.0, 75.0, 100.0]
        newsvendor_config.demand_params.std = [10.0, 15.0, 20.0]
        newsvendor_config.demand_dist = "normal"

        problem1 = instantiate(newsvendor_config, solver="appsi_highs")
        problem2 = instantiate(newsvendor_config, solver="appsi_highs")

        samples1 = problem1.generate_samples(n_samples=20, seed=42)
        samples2 = problem2.generate_samples(n_samples=20, seed=42)

        np.testing.assert_array_equal(samples1, samples2)

        tpm_data1, feat_names1 = problem1.generate_tpm_data(n_decisions=20, train_samples=samples1, seed=42)
        tpm_data2, feat_names2 = problem2.generate_tpm_data(n_decisions=20, train_samples=samples2, seed=42)

        np.testing.assert_array_equal(tpm_data1, tpm_data2)
        assert feat_names1 == feat_names2


class TestDensityConfiguration:
    """Test density type configuration."""

    def test_density_type_uniform(self, newsvendor_config):
        """Test uniform density type setting."""
        newsvendor_config.x_density = "uniform"
        problem = instantiate(newsvendor_config)
        assert problem.x_density_type == "uniform"

    def test_density_type_default(self, newsvendor_config):
        """Test default density type when not specified."""
        # Remove x_density if present
        if "x_density" in newsvendor_config:
            del newsvendor_config.x_density
        problem = instantiate(newsvendor_config)
        assert problem.x_density_type == "uniform"
