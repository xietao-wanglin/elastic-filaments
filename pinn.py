import os
os.environ["DDE_BACKEND"] = "pytorch" # Export Enviromental variable to use PyTorch

import deepxde as dde
import numpy as np
import pandas as pd
from deepxde.backend import torch

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
        x[:, 1:2]*dy_t
        + dy_xxxx
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

def load_data(dataset_path):
    """
    Reads parquet files and formats data.
    """
    data = pd.read_parquet(dataset_path)
    y_data = data.pop('y').to_numpy().reshape(-1, 1)
    x_data = data.to_numpy()
    return x_data, y_data

if __name__ == "__main__":

    print('Initialisation... \n')

    # One set is 1000 iterations
    sets_adam = 30
    sets_lbfgs = 5

    x_train, y_train = load_data('./data/train.parquet')
    x_test, y_test = load_data('./data/test.parquet')

    geom = dde.geometry.Rectangle([0, 1e-5], [1, 10000]) # X x [\eta]
    timedomain = dde.geometry.TimeDomain(0, 1) # T
    geomtime = dde.geometry.GeometryXTime(geom, timedomain) # X x [\eta] x T

    bc1 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_l)
    bc2 = dde.icbc.OperatorBC(geomtime, lambda x, y, _: dy(x, y) - torch.cos(x[:, 2:3]), boundary_l)
    bc3 = dde.icbc.OperatorBC(geomtime, lambda x, y, _: ddy(x, y), boundary_r)
    bc4 = dde.icbc.OperatorBC(geomtime, lambda x, y, _: dddy(x, y), boundary_r)

    ic = dde.icbc.IC(
        geomtime,
        lambda x: 0,
        lambda _, on_initial: on_initial,
    )

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc1, bc2, bc3, bc4, ic],
        num_domain=6000,
        num_boundary=1500,
        num_initial=1500,
        num_test=6000,
    )

    net = dde.nn.FNN([3] + [48] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    errors_train = np.zeros(sets_adam + sets_lbfgs)
    errors_test = np.zeros(sets_adam + sets_lbfgs)

    print('Training Adam optimiser... \n')
    for s in range(sets_adam):
        model.compile("adam", lr=1e-3)
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        errors_train[s] = dde.metrics.l2_relative_error(y_train, y_pred_train)
        errors_test[s] = dde.metrics.l2_relative_error(y_train, y_pred_test)
        print(f'L2 rel. errors. Prediction: {errors_train[s]}. Extrapolation: {errors_test[s]}')
        losshistory, train_state = model.train(iterations=1000, model_save_path='./model/model')

    print('Training L-BFGS optimiser... \n')
    for s in range(sets_lbfgs):

        dde.optimizers.set_LBFGS_options(maxiter=1000)
        model.compile("L-BFGS")
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        errors_train[s+sets_adam] = dde.metrics.l2_relative_error(y_train, y_pred_train)
        errors_test[s+sets_adam] = dde.metrics.l2_relative_error(y_train, y_pred_test)
        losshistory, train_state = model.train(iterations=1000, model_save_path='./model/model')

    print('Saving data... \n')
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir='./data')

    np.save('/data/errors_train.npy', errors_train)
    np.save('/data/errors_test.npy', errors_test)
