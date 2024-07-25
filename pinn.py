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
        dy_t
        + x[:, 1:2]*dy_xxxx
    )

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

    x_test, y_test = load_data('./data/test_0.parquet')

    geom = dde.geometry.Rectangle([0, 0], [1, 10]) # X x [\eta]
    timedomain = dde.geometry.TimeDomain(0, 1) # T
    geomtime = dde.geometry.GeometryXTime(geom, timedomain) # X x [\eta] x T

    bc1 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_l)
    bc2 = dde.icbc.NeumannBC(geomtime, lambda x: np.cos(x[:, 2:3]), boundary_l)
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

    checker = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", save_better_only=True, period=1000
    )   

    net = dde.nn.FNN([3] + [48] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3)
    sets = 30
    errors = np.zeros(sets)
    for s in range(sets):
        y_pred = model.predict(x_test)
        errors[s] = dde.metrics.l2_relative_error(y_test, y_pred)
        print(errors[s])
        model.train(iterations=1000, callbacks=[checker])

    dde.optimizers.set_LBFGS_options(maxiter=3000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train(iterations=3000)
    
    dde.saveplot(losshistory, train_state, isplot=True)
