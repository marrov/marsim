from typing import Protocol

import numba
import numpy as np


class SamplingFunctionProtocol(Protocol):
    def __call__(self, size: int, scale: float) -> np.ndarray: ...


@numba.jit(nopython=True)
def _calculate_mean_demand(
    price: float, alpha: float, beta: float, gamma: float
) -> float:
    return alpha / (1 + np.exp(beta * price + gamma))


@numba.jit(nopython=True)
def generate_samples_numba(
    prices: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    noise_scale: float,
    n_samples: int,
) -> np.ndarray:
    n_points = len(prices)
    results = np.zeros((n_points, n_samples))
    for i in range(n_points):
        mean_demand = _calculate_mean_demand(prices[i], alpha, beta, gamma)
        base_samples = np.random.poisson(mean_demand, size=n_samples)
        noise = np.exp(np.random.normal(0, noise_scale, size=n_samples))
        results[i] = np.round(base_samples * noise, decimals=0)
    return results
