import numpy as np

from marsim.demand.functions import _calculate_mean_demand, generate_samples_numba


def test_calculate_mean_demand():
    # Test normal case
    assert np.isclose(_calculate_mean_demand(10.0, 100.0, 0.1, 0.0), 26.894142136)

    # Test with zero price
    assert np.isclose(_calculate_mean_demand(0.0, 100.0, 0.1, 0.0), 50.0)

    # Test with large values
    assert _calculate_mean_demand(1000.0, 100.0, 0.1, 0.0) < 1e-40

    # Test with different parameter combinations
    assert np.isclose(_calculate_mean_demand(5.0, 200.0, 0.2, 1.0), 23.8405844)


def test_generate_samples_numba_shape():
    prices = np.array([1.0, 2.0, 3.0, 4.0])
    samples = generate_samples_numba(prices, 100.0, 0.1, 0.0, 0.2, 10)
    assert samples.shape == (4, 10)


def test_generate_samples_numba_non_negative():
    prices = np.array([1.0, 5.0, 10.0])
    samples = generate_samples_numba(prices, 100.0, 0.1, 0.0, 0.2, 100)
    assert np.all(samples >= 0)


def test_generate_samples_numba_price_sensitivity():
    # Test that demand decreases as price increases (in expectation)
    prices = np.array([1.0, 10.0, 20.0])
    np.random.seed(42)
    samples = generate_samples_numba(prices, 100.0, 0.1, 0.0, 0.1, 1000)

    mean_demands = [np.mean(samples[i]) for i in range(len(prices))]
    assert mean_demands[0] > mean_demands[1] > mean_demands[2]


def test_generate_samples_numba_noise_impact():
    prices = np.array([5.0])
    np.random.seed(42)

    samples_low_noise = generate_samples_numba(prices, 100.0, 0.1, 0.0, 0.1, 1000)
    samples_high_noise = generate_samples_numba(prices, 100.0, 0.1, 0.0, 0.5, 1000)

    assert np.var(samples_high_noise) > np.var(samples_low_noise)


def test_generate_samples_numba_parameters():
    prices = np.array([3.0])
    np.random.seed(42)

    # Higher alpha should lead to higher demand
    samples_low_alpha = generate_samples_numba(prices, 50.0, 0.1, 0.0, 0.2, 1000)
    samples_high_alpha = generate_samples_numba(prices, 150.0, 0.1, 0.0, 0.2, 1000)
    assert np.mean(samples_high_alpha) > np.mean(samples_low_alpha)

    # Higher beta should lead to lower demand (more price sensitive)
    samples_low_beta = generate_samples_numba(prices, 100.0, 0.05, 0.0, 0.2, 1000)
    samples_high_beta = generate_samples_numba(prices, 100.0, 0.2, 0.0, 0.2, 1000)
    assert np.mean(samples_low_beta) > np.mean(samples_high_beta)

    # Higher gamma should shift the curve and lead to lower demand
    samples_low_gamma = generate_samples_numba(prices, 100.0, 0.1, 0.0, 0.2, 1000)
    samples_high_gamma = generate_samples_numba(prices, 100.0, 0.1, 1.0, 0.2, 1000)
    assert np.mean(samples_low_gamma) > np.mean(samples_high_gamma)


def test_generate_samples_numba_rounding():
    prices = np.array([2.0])
    samples = generate_samples_numba(prices, 100.0, 0.1, 0.0, 0.2, 100)

    # Check that all values are integers
    assert np.all(np.equal(np.mod(samples, 1), 0))
