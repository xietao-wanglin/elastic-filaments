import sys
import os
os.environ["DDE_BACKEND"] = "pytorch" # Export Enviromental variable to use PyTorch

import deepxde as dde
import numpy as np
import pandas as pd
from deepxde.backend import torch
job_id = int(sys.argv[1])
dde.config.set_random_seed(job_id)
np.random.seed(job_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(job_id)

def pde(x, y):
    """
    PDE residual definition
    x[:, 0:1] - x-variable
    x[:, 1:2] - eta
    x[:, 2:3] - t-variable
    """
    dy_t = dde.grad.jacobian(y, x, i=0, j=2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return (
        dy_t
        + x[:, 1:2]*dy_xxxx
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
    return x[:, 0:1]*x[:, 2:3]*y

B = torch.normal(mean=0, std=1, size=(128, 2), device=device)*1
def feature_transform_random(x):
    x1 = x[:, 0:1].reshape(-1, 1)
    eta = x[:, 1:2].reshape(-1, 1)
    t = x[:, 2:3].reshape(-1, 1)
    
    x = x[:, 1:3]
    s = torch.sin((x @ B.T))
    c = torch.cos((x @ B.T))
    features = torch.cat([s, c, x1, eta, t], dim=1)
    return features

if __name__ == "__main__":

    print('Initialisation... RF \n')

    results_folder = f'run_rf_{job_id}'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # One set is 1000 iterations
    sets_adam = 40
    sets_lbfgs = 5

    geom = dde.geometry.Rectangle([0, 0.0001], [1, 10]) # X x [\eta]
    timedomain = dde.geometry.TimeDomain(0, 10) # T
    geomtime = dde.geometry.GeometryXTime(geom, timedomain) # X x [\eta] x T

    bc2 = dde.icbc.OperatorBC(geomtime, lambda x, y, _: dy(x, y) - torch.cos(x[:, 2:3]), boundary_l)
    bc3 = dde.icbc.OperatorBC(geomtime, lambda x, y, _: ddy(x, y), boundary_r)
    bc4 = dde.icbc.OperatorBC(geomtime, lambda x, y, _: dddy(x, y), boundary_r)

    # Collection points
    n_data = 6000
    x_data = np.random.uniform(0, 1, n_data)
    eta_data = np.exp(np.random.uniform(np.log(0.0001), np.log(10), n_data))
    t_data = np.random.uniform(0, 10, n_data)
    data_anchors = np.column_stack((x_data, eta_data, t_data))

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc2, bc3, bc4],
        num_domain=0,
        num_boundary=2000,
        num_test=10000,
        anchors=data_anchors
    )

    net = dde.nn.FNN([256 + 3] + [20] * 4 + [1], "tanh", "Glorot uniform")
    net.apply_feature_transform(feature_transform_random)
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
    np.save(f'./{results_folder}/B_matrix.npy', B.numpy())
