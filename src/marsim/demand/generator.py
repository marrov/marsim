import numpy as np

from .curve import DemandCurve
from .functions import SamplingFunctionProtocol, generate_samples_numba


class DemandGenerator:
    def __init__(
        self,
        alpha: float = 100,
        beta: float = 1,
        gamma: float = 10,
        noise_scale: float = 0.001,
        sampling_function: SamplingFunctionProtocol = generate_samples_numba,
    ):
        """
        Initialize the demand generator class

        Parameters
        ----------
        alpha : float, optional
            Market size, by default 100
        beta : float, optional
            Slope of the elastic region, by default 1
        gamma : float, optional
            Location of the inflection point, by default 10
        noise_scale : float, optional
            Scale of the noise, by default 0.001
        sampling_function : SamplingFunctionProtocol, optional
            Function that generate the sampled demands, by default _generate_samples
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = -1 * gamma
        self.noise_scale = noise_scale
        self.sampling_function = sampling_function
        np.random.seed(42)

    def generate_demand_curve(
        self, min_price: float, max_price: float, n_points: int, n_samples: int = 10
    ) -> DemandCurve:
        prices = np.linspace(min_price, max_price, n_points)
        demands = self.sampling_function(
            prices=prices,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            noise_scale=self.noise_scale,
            n_samples=n_samples,
        )
        return DemandCurve(prices=prices, demands=demands)
