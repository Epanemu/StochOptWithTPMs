
import pytest
import numpy as np
from hydra.utils import instantiate

class TestTPMPairing:
    """Test pairing logic in generate_tpm_data."""

    def test_pairing_correctness(self, newsvendor_config):
        """Test that xi and x are paired correctly 1-to-1."""
        problem = instantiate(newsvendor_config)

        # Create known train_samples (xi)
        # 2 samples, 2 products
        train_samples = np.array([
            [10.0, 20.0],
            [30.0, 40.0]
        ])

        # Mock generate_decision_samples to return known x
        # We want to verify that row i of xi is paired with row i of x
        known_x = np.array([
            [100.0, 200.0],
            [300.0, 400.0],
            [500.0, 600.0]
        ])

        # Monkey patch generate_decision_samples
        original_gen = problem.generate_decision_samples
        problem.generate_decision_samples = lambda n, seed=None, xi=None: known_x[:n]

        try:
            # n_decisions=2, train_samples=2. Total 4 pairs.
            tpm_data, feat_names = problem.generate_tpm_data(n_decisions=2, train_samples=train_samples)

            # Check shape: 2*2 samples, 2+2+1 columns
            assert tpm_data.shape == (4, 5)

            # Expected pairing:
            # (xi1, x1), (xi1, x2), (xi2, x1), (xi2, x2)

            # Check xi columns (first 2)
            assert np.allclose(tpm_data[0, :2], train_samples[0]) # xi1
            assert np.allclose(tpm_data[1, :2], train_samples[0]) # xi1
            assert np.allclose(tpm_data[2, :2], train_samples[1]) # xi2
            assert np.allclose(tpm_data[3, :2], train_samples[1]) # xi2

            # Check x columns (next 2)
            assert np.allclose(tpm_data[0, 2:4], known_x[0]) # x1
            assert np.allclose(tpm_data[1, 2:4], known_x[1]) # x2
            assert np.allclose(tpm_data[2, 2:4], known_x[0]) # x1
            assert np.allclose(tpm_data[3, 2:4], known_x[1]) # x2

        finally:
            # Restore method
            problem.generate_decision_samples = original_gen
