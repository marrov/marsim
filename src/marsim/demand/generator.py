from typing import Protocol

import numpy as np

from .curve import DemandCurve
from .functions import _generate_samples


class DemandGenerator(Protocol):
    def sample(self, price: float, n_samples: int) -> np.ndarray:
        pass

    def generate_demand_curve(
        self, min_price: float, max_price: float, n_points: int, n_samples: int
    ) -> DemandCurve:
        pass


class DemandGeneratorScipy:
    def __init__(
        self,
        alpha: float = 100,
        beta: float = 0.1,
        gamma: float = 0,
        noise_std: float = 0.1,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.noise_std = noise_std
        np.random.seed(42)

    def sample(self, price: float, n_samples: int = 10) -> np.ndarray:
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
        prices = np.linspace(min_price, max_price, n_points)
        demands = _generate_samples(
            prices, self.alpha, self.beta, self.gamma, self.noise_std, n_samples
        )
        return DemandCurve(prices=prices, demands=demands)
