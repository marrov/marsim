# %% Imports

from demand import DemandGenerator

# %% Initialize demand generator

generator = DemandGenerator(alpha=50, beta=1, gamma=10, noise_scale=0.001)

# %% Generate demand

demand_curve = generator.generate_demand_curve(
    min_price=0, max_price=20, n_points=50, n_samples=10
)

# %% Plot demand curve

demand_curve.plot()

# %% Print out as a dataframe

demand_curve.to_df()

# %%
