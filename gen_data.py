import numpy as np
from tqdm import tqdm
import pandas as pd

from simulation import Simulation

if __name__ == "__main__":

    n_etas = 100
    seed = 0
    np.random.seed(seed)
    eta_range = np.random.uniform(0, 10, n_etas)
    data_all = []
    print('Generating dataset... \n')
    for eta in tqdm(eta_range):
        sim = Simulation(timesteps=100,
                         n_points=200,
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
