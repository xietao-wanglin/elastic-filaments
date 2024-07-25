import numpy as np
import pandas as pd
from tqdm import tqdm

from simulation import Simulation

"""
This scripts generates a large dataset with many simulations,
each with different values of eta.
"""
if __name__ == "__main__":

    n_etas = 100
    seed = 3
    np.random.seed(seed)
    eta_range = np.random.uniform(10000, 20000, n_etas)
    data_all = []
    print('Generating dataset... \n')
    for eta in tqdm(eta_range):
        sim = Simulation(timesteps=1000,
                         n_points=50,
                         max_time=1, osc_freq=1,
                         length=1, eta=eta)
        sim.run()
        y, x, t, _ = sim.get_data()
        for j, t_j in enumerate(t):
            for i, x_i in enumerate(x):
                list_data = [x_i, eta, t_j, y[j][i]]
                data_all.append(list_data)

    df = pd.DataFrame(data_all, columns=['x', 'eta', 't', 'y'])
    df.to_parquet(f'./data/test_{seed}.parquet')
    df_sub = df.sample(n=10000, random_state=seed)            
    df_sub.to_parquet(f'./data/test_sub_{seed}.parquet')