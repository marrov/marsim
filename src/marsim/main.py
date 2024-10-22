# %% Imports

from demand import DemandGeneratorScipy

# %% Initialize demand generator

generator = DemandGeneratorScipy(alpha=100, beta=1, gamma=-10, noise_std=0.001)

# %% Generate demand

# Generate demand curve
demand_curve = generator.generate_demand_curve(
    min_price=0, max_price=20, n_points=50, n_samples=100
)

# %% Plot demand curve

demand_curve.plot()

# %%
