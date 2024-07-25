from typing import Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_banded
from tqdm import tqdm

class Simulation:

    def __init__(self, timesteps: int,
                n_points: int, 
                max_time: Optional[float] = 1.0, 
                eta: Optional[float] = 1.0, 
                osc_freq: Optional[float] = 1.0, 
                length: Optional[float] = 1.0, 
                left_bc: Optional[str] = 'clamped',
                right_bc: Optional[str] = 'free',
                ic: Optional[Callable] = lambda _: 0) -> None:
        """
        Initializes the Simulation class.

        Parameters:
        timesteps (int): number of time steps.
        n_points (int): number of spatial points.
        max_time (float, optional): total simulation time, default is 1.0.
        eta (float, optional): eta, default is 1.0.
        osc_freq (float, optional): oscillation frequency, default is 1.0.
        length (float, optional): length of the domain, default is 1.0.
        left_bc (str, optional): type of boundary condition at x=0, default is 'clamped'.
        right_bc (str, optional): type of boundary condition at x=L, default is 'free'.
        ic (Callabel, optional): the initial condition function, default is 0.
        """
        self.timesteps = timesteps
        self.n_points = n_points
        self.max_time = max_time
        self.eta = eta
        self.osc_freq = osc_freq
        self.length = length
        self.dt = self.max_time/(self.timesteps-1)
        self.dx = self.length/(self.n_points-1)
        self.y = np.zeros((self.timesteps, self.n_points), dtype=np.cdouble)
        self.left_bc = left_bc
        self.right_bc = right_bc

        x = np.linspace(0, self.length, self.n_points)
        self.y[0] = ic(x)

    def next_t(self, prev, iteration):

        al = self.dt/(self.eta*self.dx**4)
        b = (prev).copy()
        Ab = np.zeros((9, self.n_points))
        Ab[2, 4:] = al
        Ab[3, 3:-1] = -4*al
        Ab[4, 2:-2] = 6*al + 1
        Ab[5, 1:-3] = -4*al
        Ab[6, :-4] = al

        if self.left_bc == 'clamped':

            Ab[4, 0] = 1
            
            Ab[5, 0] = -(11/6)/self.dx
            Ab[4, 1] = +(3)/self.dx
            Ab[3, 2] = -(9/6)/self.dx
            Ab[2, 3] = +(1/3)/self.dx

            b[0] = 0
            b[1] = np.cos(self.osc_freq*self.dt*iteration)

        elif self.left_bc == 'free':

            Ab[4, 0] = -5/2
            Ab[3, 1] = 9
            Ab[2, 2] = -12
            Ab[1, 3] = 7
            Ab[0, 4] = -3/2
            
            Ab[5, 0] = +(2)/self.dx
            Ab[4, 1] = -(5)/self.dx
            Ab[3, 2] = +(4)/self.dx
            Ab[2, 3] = -(1)/self.dx

            b[0] = 0
            b[1] = 0

        else: raise ValueError(f'Unknown boundary condition on left side: {self.left_bc}')

        Ab[3, -1] = 2
        Ab[4, -2] = -5
        Ab[5, -3] = 4
        Ab[6, -4] = -1

        Ab[4, -1] = 5/2
        Ab[5, -2] = -9
        Ab[6, -3] = 12
        Ab[7, -4] = -7
        Ab[8, -5] = 3/2

        b[-1] = 0
        b[-2] = 0
        sol = solve_banded((4,4), Ab, b)
        return sol
    
    def run(self, verbose=False):
        """
        Runs the simulation.

        Parameters:
        verbose (bool, optional): If True, prints progress. Default is False.
        """
        if verbose:
            print('Solving BVP... \n')
            for iteration in tqdm(range(1, self.timesteps)):
                self.y[iteration] = self.next_t(self.y[iteration-1], iteration)
        else:
            for iteration in range(1, self.timesteps):
                self.y[iteration] = self.next_t(self.y[iteration-1], iteration)

    def create_video(self):
        """
        Creates a video of the simulation.
        """
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.length)
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

    def get_data(self):
        """
        Returns a tuple with the data.
        """
        y_real = np.real(self.y)
        x = np.linspace(0, self.length, self.n_points)
        t = np.linspace(0, self.max_time, self.timesteps)
        return y_real, x, t, self.eta

    def save_data(self, filename: Optional[str] = None):
        """
        Creates a .parquet file with the data.
        
        Parameters:
        filename (str, optional): sets a fixed filename.
        """
        if filename is None:
            filename = f'eta_{self.eta}'
        
        y_real, x, t, _ = self.get_data()
        df_data = []
        for j, t_j in enumerate(t):
            for i, x_i in enumerate(x):
                list_data = [x_i, self.eta, t_j, y_real[j][i]]
                df_data.append(list_data)
        
        df = pd.DataFrame(df_data, columns=['x', 'eta', 't', 'y'])
        df.to_parquet(f'./data/{filename}.parquet')
