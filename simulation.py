import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_banded
from typing import Optional
from tqdm import tqdm

class Simulation:

    def __init__(self, timesteps: int,
                n_points: int, 
                max_time: float, 
                eta: Optional[float] = 1.0, 
                osc_freq: Optional[float] = 1.0, 
                length_multiplier: Optional[int] = 1) -> None:
        
        self.timesteps = timesteps
        self.n_points = n_points
        self.max_time = max_time
        self.eta = eta
        self.osc_freq = osc_freq
        self.length = length_multiplier*((eta/osc_freq)**(1/4))
        self.dt = self.max_time/(self.timesteps-1)
        self.dx = self.length/(self.n_points-1)
        self.y = np.zeros((self.timesteps, self.n_points), dtype=np.cdouble)

    def next_t(self, prev, iteration):

        al = self.eta*self.dt/(self.dx**4)
        Ab = np.zeros((7, self.n_points))
        Ab[0, 4:] = al
        Ab[1, 3:-1] = -4*al
        Ab[2, 2:-2] = 6*al + 1
        Ab[3, 1:-3] = -4*al
        Ab[4, :-4] = al
        
        Ab[2, 0] = 1
        
        Ab[3, 0] = -(11/6)/self.dx
        Ab[2, 1] = +(3)/self.dx
        Ab[1, 2] = -(9/6)/self.dx
        Ab[0, 3] = +(1/3)/self.dx

        Ab[1, -1] = 2
        Ab[2, -2] = -5
        Ab[3, -3] = 4
        Ab[4, -4] = -1

        Ab[2, -1] = 5/2
        Ab[3, -2] = -9
        Ab[4, -3] = 12
        Ab[5, -4] = -7
        Ab[6, -5] = 3/2
        b = (prev).copy()
        b[0] = 0
        b[1] = np.exp(1j*self.osc_freq*self.dt*iteration)

        b[-1] = 0
        b[-2] = 0
        sol = solve_banded((4,2), Ab, b)
        return sol
    
    def run(self):
        print('Solving BVP... \n')
        for iteration in tqdm(range(1, self.timesteps)):
            self.y[iteration] = self.next_t(self.y[iteration-1], iteration)

    def create_video(self):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-0.2, self.length + 0.2)
        ax.set_ylim(-1, 1)
        ax.grid()
        x = np.linspace(0, self.length, self.n_points) # Grid
        line, = ax.plot(x, self.y[0])
        point = ax.scatter(0, 0, marker='x')
        frame_mult = int(self.timesteps/100)
        def update(fn):
            line.set_data(x, self.y[frame_mult*fn])
            point.set_label(f'Time: {frame_mult*fn*self.dt:.4f}')
            ax.legend(loc='upper left')

        animation = FuncAnimation(fig, update, interval=50, frames=100)
        plt.show()

if __name__ == "__main__":

    sim = Simulation(timesteps=10000,
                     n_points=200,
                     max_time=10, osc_freq=4,
                     length_multiplier=10)
    
    sim.run()
    sim.create_video()