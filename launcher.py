import numpy as np

from simulation import Simulation


if __name__ == "__main__":

    sim = Simulation(timesteps=1000,
                     n_points=50,
                     max_time=1, osc_freq=1,
                     length=1, eta=1,)
    
    sim.run(verbose=True)
    sim.create_video()
