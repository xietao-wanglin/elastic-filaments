import numpy as np

from simulation import Simulation


if __name__ == "__main__":

    eta = 40
    sim = Simulation(timesteps=1000,
                     n_points=200,
                     max_time=1, osc_freq=1,
                     length=1.5, eta=eta)
    
    sim.run(verbose=True)
    sim.save_data(filename=f'filament_1.5_{eta}')
