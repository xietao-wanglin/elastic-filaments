import numpy as np

from simulation import Simulation


if __name__ == "__main__":

    sim = Simulation(timesteps=10000,
                     n_points=2000,
                     max_time=1, osc_freq=4,
                     length=1, eta=1,)
    
    sim.run(verbose=True)
    sim.create_video()
