import numpy as np

from simulation import Simulation


if __name__ == "__main__":

    eta = 0.1
    sim = Simulation(timesteps=1000,
                    n_points=200,
                    max_time=10, osc_freq=1,
                    length=1, eta=eta)
    
    sim.run(verbose=True)
    sim.create_video()
