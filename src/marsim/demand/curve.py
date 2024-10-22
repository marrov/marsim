from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
        for i in range(self.demands.shape[1]):
            df[f"sample_{i}"] = self.demands[:, i]
        return df

    def plot(self, show_samples: bool = True, show_std: bool = True):
        """Plot the demand curve with optional samples and standard deviation."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.prices, self.mean_demand, "bo-", label="Mean Demand", linewidth=2)
        if show_std:
            plt.fill_between(
                self.prices,
                self.mean_demand - self.std_demand,
                self.mean_demand + self.std_demand,
                alpha=0.2,
                color="b",
                label="±1 std",
            )
        if show_samples:
            for i in range(self.demands.shape[1]):
                plt.plot(self.prices, self.demands[:, i], "b.", alpha=0.3, markersize=3)
        plt.xlabel("Price")
        plt.ylabel("Demand")
        plt.title("Demand Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
