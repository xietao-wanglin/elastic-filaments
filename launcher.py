from simulation import Simulation

if __name__ == "__main__":

    sim = Simulation(timesteps=100,
                     n_points=30,
                     max_time=10, osc_freq=1,
                     length=1, eta=1)
    
    sim.run(verbose=True)
    sim.create_video()
