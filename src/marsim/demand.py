from typing import Protocol
import numpy as np
import numba
from dataclasses import dataclass
import pandas as pd
from matplotlib import pyplot as plt


# Define return type for demand curve
@dataclass
class DemandCurve:
    """Container for demand curve data with built-in plotting capabilities."""

    prices: np.ndarray
    demands: np.ndarray  # Shape: (n_prices, n_samples)

    @property
    def mean_demand(self) -> np.ndarray:
        """Calculate mean demand across samples for each price point."""
        return np.mean(self.demands, axis=1)

    @property
    def std_demand(self) -> np.ndarray:
        """Calculate standard deviation of demand across samples for each price point."""
        return np.std(self.demands, axis=1)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert demand curve data to a pandas DataFrame."""
        df = pd.DataFrame(
            {
                "price": self.prices,
                "mean_demand": self.mean_demand,
                "std_demand": self.std_demand,
            }
        )
        # Add individual sample columns
        for i in range(self.demands.shape[1]):
            df[f"sample_{i}"] = self.demands[:, i]
        return df

    def plot(self, show_samples: bool = True, show_std: bool = True):
        """
        Plot the demand curve with optional samples and standard deviation.

        Parameters
        ----------
        show_samples : bool, optional
            Whether to plot individual samples, by default True
        show_std : bool, optional
            Whether to plot standard deviation band, by default True
        """
        plt.figure(figsize=(10, 6))

        # Plot mean demand
        plt.plot(self.prices, self.mean_demand, "b-", label="Mean Demand", linewidth=2)

        if show_std:
            # Plot standard deviation band
            plt.fill_between(
                self.prices,
                self.mean_demand - self.std_demand,
                self.mean_demand + self.std_demand,
                alpha=0.2,
                color="b",
                label="±1 std",
            )

        if show_samples:
            # Plot individual samples
            for i in range(self.demands.shape[1]):
                plt.plot(self.prices, self.demands[:, i], "b.", alpha=0.1, markersize=2)

        plt.xlabel("Price")
        plt.ylabel("Demand")
        plt.title("Demand Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()


class DemandGenerator(Protocol):
    def sample(self, price: float, n_samples: int) -> np.ndarray:
        pass

    def generate_demand_curve(
        self, min_price: float, max_price: float, n_points: int, n_samples: int
    ) -> DemandCurve:
        pass


# Numba-optimized functions for core calculations
@numba.jit(nopython=True)
def _calculate_mean_demand(
    price: float, alpha: float, beta: float, gamma: float
) -> float:
    return alpha / (1 + np.exp(beta * price + gamma))


@numba.jit(nopython=True)
def _generate_samples(
    prices: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    noise_std: float,
    n_samples: int,
) -> np.ndarray:
    n_points = len(prices)
    results = np.zeros((n_points, n_samples))

    for i in range(n_points):
        mean_demand = _calculate_mean_demand(prices[i], alpha, beta, gamma)
        # Generate Poisson samples
        base_samples = np.random.poisson(mean_demand, size=n_samples)
        # Generate log-normal noise
        noise = np.exp(np.random.normal(0, noise_std, size=n_samples))
        # Combine and store results
        results[i] = np.round(base_samples * noise)

    return results


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

        # Set random seed for numba functions
        np.random.seed(42)

    def sample(self, price: float, n_samples: int = 10) -> np.ndarray:
        """
        Generate demand samples for a given price with added noise.

        Parameters
        ----------
        price : float
            Price point to generate demand for
        n_samples : int, optional
            Number of samples to generate, by default 10

        Returns
        -------
        np.ndarray
            Array of demand samples
        """
        return _generate_samples(
            np.array([price]),
            self.alpha,
            self.beta,
            self.gamma,
            self.noise_std,
            n_samples,
        )[0]

    def generate_demand_curve(
        self, min_price: float, max_price: float, n_points: int, n_samples: int = 10
    ) -> DemandCurve:
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
        DemandCurve
            Object containing prices and demands with plotting capabilities
        """
        prices = np.linspace(min_price, max_price, n_points)
        demands = _generate_samples(
            prices, self.alpha, self.beta, self.gamma, self.noise_std, n_samples
        )

        return DemandCurve(prices=prices, demands=demands)
