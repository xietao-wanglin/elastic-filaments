import numpy as np

from simulation import Simulation


if __name__ == "__main__":

    eta = 1
    sim = Simulation(timesteps=10000,
                    n_points=200,
                    max_time=1, osc_freq=1,
                    length=1, eta=eta)
    
    sim.run(verbose=True)
    sim.create_video()
