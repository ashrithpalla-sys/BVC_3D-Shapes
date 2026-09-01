import numpy as np

from stoolgen.geometry import generate_stool, sample_parameters


def test_stool_is_reproducible_and_normalized():
    first_rng = np.random.default_rng(4)
    params = sample_parameters(first_rng)
    first, labels = generate_stool(params, 256, first_rng)
    second_rng = np.random.default_rng(4)
    second_params = sample_parameters(second_rng)
    second, _ = generate_stool(second_params, 256, second_rng)
    np.testing.assert_allclose(first, second)
    assert first.shape == (256, 3)
    assert labels.shape == (256,)
    assert np.abs(first).max() < 1.5

