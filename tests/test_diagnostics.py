from __future__ import annotations

import numpy as np

from us_bvar.diagnostics import _array_chain_metrics, chain_diagnostic


def test_vectorized_array_metrics_match_scalar_reference() -> None:
    rng = np.random.default_rng(204)
    values = rng.normal(size=(48, 17))
    chain_ids = np.repeat(np.arange(2), 24)

    vector_r_hat, vector_ess = _array_chain_metrics(values, chain_ids)
    scalar = [chain_diagnostic(values[:, index], chain_ids) for index in range(values.shape[1])]
    scalar_r_hat = np.asarray([item["r_hat"] for item in scalar])
    scalar_ess = np.asarray([item["effective_sample_size"] for item in scalar])

    assert np.allclose(vector_r_hat, scalar_r_hat, rtol=1e-12, atol=1e-12)
    assert np.allclose(vector_ess, scalar_ess, rtol=1e-12, atol=1e-12)


def test_vectorized_metrics_handle_constant_dimensions_without_warnings() -> None:
    values = np.ones((40, 3))
    chain_ids = np.repeat(np.arange(2), 20)

    r_hat, effective_size = _array_chain_metrics(values, chain_ids)

    assert np.array_equal(r_hat, np.ones(3))
    assert np.array_equal(effective_size, np.full(3, 40.0))
