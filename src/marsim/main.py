# %% Imports

from demand import DemandGeneratorScipy

# %% Initialize demand generator

alpha = 100
beta = 1
gamma = -10
noise = 0.001
demand_generator = DemandGeneratorScipy(alpha, beta, gamma, noise)

# %% Generate demand

demand = demand_generator.generate_demand_curve(min_price=0, max_price=10, n_points=10)

# %% Plot demand curve
