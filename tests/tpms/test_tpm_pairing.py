import numpy as np
from hydra.utils import instantiate


class TestTPMPairing:
    """Test pairing logic in generate_tpm_data."""

    def test_pairing_cartesian(self, newsvendor_config):
        """Test that xi and x are paired as Cartesian product."""
        problem = instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")

        # 2 samples, 2 products
        train_samples = np.array([[10.0, 20.0], [30.0, 40.0]])
        # Mock generate_decision_samples
        known_x = np.array([[100.0, 200.0], [300.0, 400.0]])

        original_gen = problem.generate_decision_samples
        # n_decisions=2
        problem.generate_decision_samples = lambda n, seed=None, xi=None: known_x[:n]

        try:
            tpm_data, feat_names = problem.generate_tpm_data(
                n_decisions=2, train_samples=train_samples, cartesian_product=True
            )

            # Check shape: 2*2 samples, 2+2+1 columns
            assert tpm_data.shape == (4, 5)

            # Expected pairing: (xi1, x1), (xi1, x2), (xi2, x1), (xi2, x2)
            assert np.allclose(tpm_data[0, :2], train_samples[0])
            assert np.allclose(tpm_data[1, :2], train_samples[0])
            assert np.allclose(tpm_data[2, :2], train_samples[1])
            assert np.allclose(tpm_data[3, :2], train_samples[1])

            assert np.allclose(tpm_data[0, 2:4], known_x[0])
            assert np.allclose(tpm_data[1, 2:4], known_x[1])
            assert np.allclose(tpm_data[2, 2:4], known_x[0])
            assert np.allclose(tpm_data[3, 2:4], known_x[1])

        finally:
            problem.generate_decision_samples = original_gen

    def test_pairing_plain(self, newsvendor_config):
        """Test that xi and x are paired 1-to-1."""
        problem = instantiate(newsvendor_config, solver="appsi_highs", _convert_="all")

        # 5 train samples
        train_samples = np.array(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]
        )
        n_dec = 10
        known_x = np.random.rand(n_dec, 2)

        original_gen = problem.generate_decision_samples
        problem.generate_decision_samples = lambda n, seed=None, xi=None: known_x[:n]

        try:
            tpm_data, feat_names = problem.generate_tpm_data(
                n_decisions=n_dec,
                train_samples=train_samples,
                cartesian_product=False,
                seed=42,
            )

            # Check shape: n_dec samples
            assert tpm_data.shape == (n_dec, 5)

            # Check that each xi in tpm_data is one of the train_samples
            for i in range(n_dec):
                row_xi = tpm_data[i, :2]
                assert any(np.allclose(row_xi, ts) for ts in train_samples)

            # Check x columns match known_x
            assert np.allclose(tpm_data[:, 2:4], known_x)

        finally:
            problem.generate_decision_samples = original_gen
