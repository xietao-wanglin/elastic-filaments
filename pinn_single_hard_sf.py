import os
os.environ["DDE_BACKEND"] = "pytorch" # Export Enviromental variable to use PyTorch

import deepxde as dde
import numpy as np
import pandas as pd
from deepxde.backend import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
eta = 1

def pde(x, y):
    """
    PDE residual definition
    x[:, 0:1] - x-variable
    x[:, 1:2] - t-variable
    """
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return (
        dy_t
        + eta*dy_xxxx
    )

def dy(x, y):
    dy_x = dde.grad.jacobian(y, x, j=0)
    return dy_x

def ddy(x, y):
    return dde.grad.hessian(y, x, i=0, j=0)

def dddy(x, y):
    dy_xx = dde.grad.jacobian(y, x, j=0)
    dy_xxx = dde.grad.hessian(dy_xx, x, j=0)
    return dy_xxx

def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

def output_transform(x, y):
    return x[:, 0:1]*x[:, 1:2]*y

def feature_transform_sfourier(x):
    features = torch.cat([x,
                         torch.sin(np.pi*x), torch.cos(np.pi*x),
                         torch.sin(2*np.pi*x), torch.cos(2*np.pi*x),
                         torch.sin(3*np.pi*x), torch.cos(3*np.pi*x),
                         torch.sin(4*np.pi*x), torch.cos(4*np.pi*x),
                         torch.sin(5*np.pi*x), torch.cos(5*np.pi*x),], dim=1)
    return features

if __name__ == "__main__":

    print('Initialisation... HARD - SF with x\n')

    results_folder = f'fixed_hardbcs_sf_{eta}_withx'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # One set is 1000 iterations
    sets_adam = 40
    sets_lbfgs = 5

    geom = dde.geometry.Interval(0, 1) # X 
    timedomain = dde.geometry.TimeDomain(0, 10) # T
    geomtime = dde.geometry.GeometryXTime(geom, timedomain) # X x T

    bc2 = dde.icbc.OperatorBC(geomtime, lambda x, y, _: dy(x, y) - torch.cos(x[:, 1:2]), boundary_l)
    bc3 = dde.icbc.OperatorBC(geomtime, lambda x, y, _: ddy(x, y), boundary_r)
    bc4 = dde.icbc.OperatorBC(geomtime, lambda x, y, _: dddy(x, y), boundary_r)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc2, bc3, bc4],
        num_domain=6000,
        num_boundary=2000,
        num_test=10000,
    )

    net = dde.nn.FNN([22] + [20] * 4 + [1], "tanh", "Glorot uniform")
    net.apply_feature_transform(feature_transform_sfourier)
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)

    total_sets = sets_adam + sets_lbfgs

    print('Training Adam optimiser... \n')
    model.compile("adam", lr=0.001)    

    for s in range(sets_adam):
        losshistory, train_state = model.train(iterations=1000, model_save_path=f'./{results_folder}/model')

    print('Training L-BFGS optimiser... \n')
    for s in range(sets_lbfgs):

        dde.optimizers.set_LBFGS_options(maxiter=1000)
        model.compile("L-BFGS")
        losshistory, train_state = model.train(iterations=1000, model_save_path=f'./{results_folder}/model')

    print('Saving data... \n')
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=f'./{results_folder}')
