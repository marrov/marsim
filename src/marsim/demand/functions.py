from typing import Protocol

import numba
import numpy as np


class SamplingFunctionProtocol(Protocol):
    def __call__(
        self,
        prices: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        noise_scale: float,
        n_samples: int,
    ) -> np.ndarray: ...


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
    """
    Generate demand samples based on the given parameters.
    This function uses a Poisson distribution to generate the base demand
    and adds log-normal noise to it.

    Parameters
    ----------
    prices : np.ndarray
        Array of price points for which to generate demand samples
    alpha : float
        Maximum demand parameter (scaling factor for the demand function)
    beta : float
        Price sensitivity parameter (higher values indicate greater sensitivity)
    gamma : float
        Offset parameter that shifts the demand curve horizontally
    noise_scale : float
        Standard deviation of the log-normal noise distribution
    n_samples : int
        Number of demand samples to generate for each price point

    Returns
    -------
    np.ndarray
        2D array of shape (len(prices), n_samples) containing the generated demand
        samples for each price point
    """
    n_points = len(prices)
    results = np.zeros((n_points, n_samples))
    for i in range(n_points):
        mean_demand = _calculate_mean_demand(prices[i], alpha, beta, gamma)
        base_samples = np.random.poisson(mean_demand, size=n_samples)
        noise = np.exp(np.random.normal(0, noise_scale, size=n_samples))
        results[i] = np.round(base_samples * noise, decimals=0)
    return results
