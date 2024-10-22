from typing import Protocol

import numpy as np
from scipy import stats


class DemandGenerator(Protocol):
    def sample(self, price: float, n_samples: int) -> np.ndarray:
        pass

    def generate_demand_curve(
        self, min_price: float, max_price: float, n_points: int, n_samples: int
    ) -> np.ndarray:
        pass


class DemandGeneratorScipy:
    def __init__(
        self,
        alpha: float = 100,
        beta: float = 0.1,
        gamma: float = 0,
        noise_std: float = 0.1,
    ):
        """
        Initialize the demand generator with parameters for the logistic function
        and additional noise.

        Parameters
        ----------
        alpha : float, optional
            Market size parameter (controls curve height), by default 100
        beta : float, optional
            Controls slope of elastic region, by default 0.1
        gamma : float, optional
            Controls location of inflection point, by default 0
        noise_std : float, optional
            Standard deviation of the multiplicative noise, by default 0.1
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.noise_std = noise_std

    def _calculate_mean_demand(self, price: float) -> float:
        """
        Calculate expected demand using logistic function.

        Parameters
        ----------
        price : float
            Price point to calculate demand for

        Returns
        -------
        float
            Expected demand
        """
        return self.alpha / (1 + np.exp(self.beta * price + self.gamma))

    def _generate_noise(self, size: int = 1) -> np.ndarray:
        """
        Generate multiplicative noise using a log-normal distribution.
        This ensures noise is always positive and multiplicative.

        Parameters
        ----------
        size : int, optional
            Number of noise samples to generate, by default 1

        Returns
        -------
        np.ndarray
            Array of noise samples
        """
        return np.exp(stats.norm.rvs(loc=0, scale=self.noise_std, size=size))

    def sample(self, price: float, n_samples: int = 10) -> np.ndarray:
        """
        Generate demand samples for a given price with added noise.

        Parameters
        ----------
        price : float
            Price point to generate demand for
        n_samples : int, optional
            Number of samples to generate, by default 1

        Returns
        -------
        np.ndarray
            Array of demand samples
        """
        mean_demand = self._calculate_mean_demand(price)
        base_samples = stats.poisson.rvs(mu=mean_demand, size=n_samples)
        noise = self._generate_noise(size=n_samples)
        return np.round(base_samples * noise).astype(int)

    def generate_demand_curve(
        self, min_price: float, max_price: float, n_points: int, n_samples: int = 10
    ) -> np.ndarray:
        """
        Generate a demand curve with given price range and number of points.

        Parameters
        ----------
        min_price : float
            Minimum price point for demand curve
        max_price : float
            Maximum price point for demand curve
        n_points : int
            Number of points to generate
        n_samples : int, optional
            Number of samples to generate for each price point (default is 10)

        Returns
        -------
        np.ndarray
            Array of demand samples
        """
        prices = np.linspace(min_price, max_price, n_points)
        sampling_func = np.vectorize(lambda price: self.sample(price, n_samples))
        demand_curve = np.array([sampling_func(prices) for prices in prices])
        return demand_curve
