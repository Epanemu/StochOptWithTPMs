"""
Integration tests for different optimization methods.
"""

from typing import Dict, List

import numpy as np
from hydra.utils import instantiate


class TestOptimizationMethods:
    """Integration tests for optimization methods."""

    def test_robust_method_small(self, newsvendor_config):
        """Test robust method with small problem."""
        problem = instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")

        # Generate small scenario set
        scenarios = problem.generate_samples(n_samples=5, seed=42)

        # Build and solve model
        problem.build_model(method="robust", scenarios=scenarios)
        problem.solve()

        # Verify solution exists
        solution = problem.get_solution()
        assert solution.shape == (2,)
        assert np.all(solution >= 0)

        # Solution should satisfy all scenarios (robust)
        satisfied = problem.check_satisfaction(solution, scenarios)
        assert np.all(satisfied), "Robust solution should satisfy all scenarios"

    def test_sample_average_method(self, newsvendor_config):
        """Test sample average approximation method."""
        problem = instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")

        scenarios = problem.generate_samples(n_samples=50, seed=42)

        problem.build_model(
            method="sample_average",
            scenarios=scenarios,
            risk_level=0.1,  # Allow 10% violation
        )
        problem.solve()

        solution = problem.get_solution()
        assert solution.shape == (2,)

        # Check that satisfaction rate is at least (1 - risk_level)
        satisfied = problem.check_satisfaction(solution, scenarios)
        satisfaction_rate = np.mean(satisfied)
        assert (
            satisfaction_rate >= 0.9
        ), f"Satisfaction rate {satisfaction_rate} too low"

    def test_single_product_problem(self, newsvendor_config):
        """Test with single product."""
        newsvendor_config.n_products = 1
        newsvendor_config.costs = [1.0]
        newsvendor_config.prices = [2.0]
        newsvendor_config.demand_params.mean = [100.0]
        newsvendor_config.demand_params.std = [20.0]

        problem = instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")
        scenarios = problem.generate_samples(n_samples=5, seed=42)

        problem.build_model(method="robust", scenarios=scenarios)
        problem.solve()
        solution = problem.get_solution()

        assert solution.shape == (1,)
        assert solution[0] >= np.max(scenarios)  # Should cover all demand

    def test_validation_workflow(self, newsvendor_config):
        """Test complete validation workflow."""
        problem = instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")

        # Training samples
        train_samples = problem.generate_samples(n_samples=20, seed=42)
        opt_samples = train_samples[:5]

        # Build and solve
        problem.build_model(method="robust", scenarios=opt_samples)
        problem.solve()
        solution = problem.get_solution()

        # Validation on new samples
        val_samples = problem.generate_samples(n_samples=50, seed=100)
        val_satisfied = problem.check_satisfaction(solution, val_samples)
        val_rate = np.mean(val_satisfied)

        # Robust solution should satisfy most validation samples
        assert val_rate >= 0.7, f"Validation satisfaction too low: {val_rate}"


class TestTPMOptimization:
    """Test optimization using TPM (SPN) approach."""

    def test_spn_method_basic(self, newsvendor_config):
        """Test basic TPM method with SPN."""
        from stochopt.data.DataHandler import DataHandler
        from stochopt.tpms.spn_tpm import SpnTPM

        problem = instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")

        # Generate training data
        train_samples = problem.generate_samples(n_samples=50, seed=42)
        tpm_data, feat_names = problem.generate_tpm_data(
            n_decisions=50, train_samples=train_samples, seed=42
        )

        # Setup DataHandler
        categ_map: Dict[int | str, List[int | str]] = {"sat": [0, 1]}
        data_handler = DataHandler(
            tpm_data, feature_names=feat_names, categ_map=categ_map
        )

        # Train TPM
        tpm = SpnTPM(data_handler=data_handler)
        tpm.train(tpm_data, min_instances_slice=10, n_clusters=2)

        # Build and solve model with TPM
        model = problem.build_model(
            method="tpm", tpm=tpm, data_handler=data_handler, risk_level=0.1
        )

        assert model is not None
        assert hasattr(model, "tpm_block")
        assert hasattr(model, "chance_constr")

        problem.solve()
        solution = problem.get_solution()

        assert solution.shape == (2,)
        assert np.all(solution >= 0)

    def test_cnet_method_basic(self, newsvendor_config):
        """Test basic TPM method with CNet."""
        from stochopt.data.DataHandler import DataHandler
        from stochopt.tpms.cnet_tpm import CNetTPM

        problem = instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")

        # Generate training data
        train_samples = problem.generate_samples(n_samples=50, seed=42)
        tpm_data, feat_names = problem.generate_tpm_data(
            n_decisions=50, train_samples=train_samples, seed=42
        )

        # Setup DataHandler
        categ_map: Dict[int | str, List[int | str]] = {"sat": [0, 1]}
        data_handler = DataHandler(
            tpm_data, feature_names=feat_names, categ_map=categ_map
        )

        # Train TPM
        tpm = CNetTPM(data_handler=data_handler)
        tpm.train(tpm_data, min_instances_slice=10, n_clusters=2)

        # Build and solve model with TPM
        model = problem.build_model(
            method="tpm", tpm=tpm, data_handler=data_handler, risk_level=0.1
        )

        assert model is not None
        assert hasattr(model, "tpm_block")
        assert hasattr(model, "chance_constr")

        problem.solve()
        solution = problem.get_solution()

        assert solution.shape == (2,)
        assert np.all(solution >= 0)

    def test_tpm_with_different_risk_levels(self, newsvendor_config):
        """Test TPM with different risk levels."""
        from stochopt.data.DataHandler import DataHandler
        from stochopt.tpms.spn_tpm import SpnTPM

        newsvendor_config.n_products = 1
        newsvendor_config.costs = [1.0]
        newsvendor_config.prices = [2.0]
        newsvendor_config.demand_params.mean = [100.0]
        newsvendor_config.demand_params.std = [20.0]

        problem = instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")

        # Generate and train TPM
        train_samples = problem.generate_samples(n_samples=80, seed=42)
        tpm_data, feat_names = problem.generate_tpm_data(
            n_decisions=80, train_samples=train_samples, seed=42
        )
        categ_map: Dict[int | str, List[int | str]] = {"sat": [0, 1]}
        data_handler = DataHandler(
            tpm_data, feature_names=feat_names, categ_map=categ_map
        )

        tpm = SpnTPM(data_handler=data_handler)
        tpm.train(tpm_data, min_instances_slice=15, n_clusters=2)

        # Test different risk levels
        solutions = {}
        for risk_level in [0.10, 0.25]:
            problem.build_model(
                method="tpm", tpm=tpm, data_handler=data_handler, risk_level=risk_level
            )
            problem.solve()
            solutions[risk_level] = problem.get_solution()

        # All solutions should be valid
        for risk_level, sol in solutions.items():
            assert sol.shape == (1,)
            assert np.all(
                sol >= 0
            ), f"Solution for risk {risk_level} has negative values"


class TestTPMDataGeneration:
    """Test TPM data generation workflow."""

    def test_tpm_data_structure(self, newsvendor_config):
        """Test that TPM data has correct structure."""
        problem = instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")

        train_samples = problem.generate_samples(n_samples=30, seed=42)
        tpm_data, feat_names = problem.generate_tpm_data(
            n_decisions=30, train_samples=train_samples, seed=42
        )

        # Verify structure
        n_products = newsvendor_config.n_products
        expected_features = n_products * 2 + 1  # demands + orders + sat

        assert tpm_data.shape[1] == expected_features
        assert len(feat_names) == expected_features

        # Verify feature names
        xi_names, x_names, sat_name = problem.get_feature_names()
        expected_names = xi_names + x_names + [sat_name]
        assert feat_names == expected_names

    def test_tpm_data_satisfaction_distribution(self, newsvendor_config):
        """Test that TPM data has balanced satisfaction."""
        problem = instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")

        train_samples = problem.generate_samples(n_samples=100, seed=42)
        tpm_data, _ = problem.generate_tpm_data(
            n_decisions=100, train_samples=train_samples, seed=42
        )

        # Check satisfaction distribution
        sat_column = tpm_data[:, -1]
        sat_rate = np.mean(sat_column)

        # With uniform sampling, should have reasonable mix
        assert 0.2 <= sat_rate <= 0.8, f"Satisfaction rate too skewed: {sat_rate}"

    def test_tpm_data_reproducibility(self, newsvendor_config):
        """Test that TPM data generation is reproducible with seed."""
        problem = instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")

        train_samples = problem.generate_samples(n_samples=20, seed=42)

        # Generate twice with same seed
        tpm_data1, _ = problem.generate_tpm_data(
            n_decisions=20, train_samples=train_samples, seed=42
        )
        tpm_data2, _ = problem.generate_tpm_data(
            n_decisions=20, train_samples=train_samples, seed=42
        )

        np.testing.assert_array_equal(tpm_data1, tpm_data2)
