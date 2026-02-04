import numpy as np


def log_sum_exp(a: float, b: float, w_a: float, w_b: float) -> float:
    """
    Compute log(w_a * exp(a) + w_b * exp(b)) safely in log-space.

    Args:
        a: float
            First log-value.
        b: float
            Second log-value.
        w_a: float
            Weight for the first value.
        w_b: float
            Weight for the second value.

    Returns:
        float: The log-sum-exp result.
    """
    if w_a <= 0:
        return float(b + np.log(max(1e-12, w_b)))
    if w_b <= 0:
        return float(a + np.log(max(1e-12, w_a)))

    la = a + np.log(w_a)
    lb = b + np.log(w_b)

    m = max(la, lb)
    if m == -np.inf:
        return float(-np.inf)
    return float(m + np.log(np.exp(la - m) + np.exp(lb - m)))


MIN_LOG_PROB = -100.0
