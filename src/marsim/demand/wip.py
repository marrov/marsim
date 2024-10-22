# Work in progres for a more composable design
from abc import ABC, abstractmethod
import numpy as np


class MeanDemandModel(ABC):
    @abstractmethod
    def calculate(self, price: float) -> float:
        pass


class LogisticDemandModel(MeanDemandModel):
    def __init__(self, alpha: float, beta: float, gamma: float):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def calculate(self, price: float) -> float:
        return self.alpha / (1 + np.exp(self.beta * price + self.gamma))


class LinearDemandModel(MeanDemandModel):
    def __init__(self, slope: float, intercept: float):
        self.slope = slope
        self.intercept = intercept

    def calculate(self, price: float) -> float:
        return max(0, self.intercept + self.slope * price)


class BaseDistribution(ABC):
    @abstractmethod
    def sample(self, loc: float, size: int) -> np.ndarray:
        pass


class PoissonDistribution(BaseDistribution):
    def sample(self, loc: float, size: int) -> np.ndarray:
        return np.random.poisson(loc, size=size)


class NormalDistribution(BaseDistribution):
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def sample(self, loc: float, size: int) -> np.ndarray:
        return np.random.normal(loc, self.scale, size=size)


class NoiseModel(ABC):
    @abstractmethod
    def apply(self, samples: np.ndarray) -> np.ndarray:
        pass


class LogNormalNoise(NoiseModel):
    def __init__(self, scale: float):
        self.scale = scale

    def apply(self, samples: np.ndarray) -> np.ndarray:
        noise = np.exp(np.random.normal(0, self.scale, size=len(samples)))
        return samples * noise


class ModularDemandGenerator:
    def __init__(
        self,
        demand_model: MeanDemandModel,
        distribution: BaseDistribution,
        noise_model: NoiseModel | None = None,
    ):
        self.demand_model = demand_model
        self.distribution = distribution
        self.noise_model = noise_model

    def generate_samples(self, prices: np.ndarray, n_samples: int) -> np.ndarray:
        n_points = len(prices)
        results = np.zeros((n_points, n_samples))

        for i, price in enumerate(prices):
            mean_demand = self.demand_model.calculate(price)
            samples = self.distribution.sample(mean_demand, n_samples)

            if self.noise_model is not None:
                samples = self.noise_model.apply(samples)

            results[i] = np.round(samples, decimals=0)

        return results


# Example usage:
demand_model = LogisticDemandModel(alpha=50, beta=1, gamma=-10)
distribution = PoissonDistribution()
noise_model = LogNormalNoise(scale=0.001)

generator = ModularDemandGenerator(
    demand_model=demand_model, distribution=distribution, noise_model=noise_model
)
