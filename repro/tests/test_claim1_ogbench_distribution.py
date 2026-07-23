import numpy as np

from claim1_ogbench_distribution import nll_numpy, reconstruct_distribution_numpy


def test_numpy_reconstruction_is_normalized_and_monotone():
    rng = np.random.default_rng(7)
    logit0 = rng.normal(size=32)
    logits = rng.normal(size=(32, 12))
    probabilities, _hazards, _survival_start, survival_end = (
        reconstruct_distribution_numpy(logit0, logits)
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(probabilities >= 0.0)
    assert np.all(np.diff(survival_end, axis=1) <= 1e-15)


def test_numpy_nll_known_constant_hazard_cases():
    bins = np.array([0, 1, 2, 4], dtype=np.int32)
    p = 0.2
    logit = np.log(p / (1.0 - p))
    logit0 = np.full(4, logit)
    logits = np.full((4, 3), logit)
    is_event = np.array([1, 1, 1, 0], dtype=float)
    tau = np.array([0, 1, 3, 1], dtype=np.int32)
    censor = np.array([0, 1, 3, 2], dtype=np.int32)
    observed = nll_numpy(bins, logit0, logits, is_event, tau, censor)
    expected_likelihood = np.array(
        [p, (1 - p) ** 2 * p, (1 - p) ** 3 * p, (1 - p) ** 3]
    )
    np.testing.assert_allclose(observed, -np.log(expected_likelihood), atol=1e-12)
