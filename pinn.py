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
        + torch.exp(x[:, 1:2])*dy_xxxx
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

def output_transform(x, y):
    return x[:, 0:1]*x[:, 2:3]*y

if __name__ == "__main__":

    print('Initialisation... \n')

    results_folder = 'log_data2'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # One set is 1000 iterations
    sets_adam = 40
    sets_lbfgs = 5
    sets_adam2 = 0

    x_train, y_train = load_data('./data/train_longtime.parquet')
    x_test, y_test = load_data('./data/test_longtime.parquet')

    geom = dde.geometry.Rectangle([0, np.log(0.0001)], [1, np.log(10)]) # X x [\eta]
    timedomain = dde.geometry.TimeDomain(0, 10) # T
    geomtime = dde.geometry.GeometryXTime(geom, timedomain) # X x [\eta] x T

    bc1 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_l) # Not used
    bc2 = dde.icbc.OperatorBC(geomtime, lambda x, y, _: dy(x, y) - torch.cos(x[:, 2:3]), boundary_l)
    bc3 = dde.icbc.OperatorBC(geomtime, lambda x, y, _: ddy(x, y), boundary_r)
    bc4 = dde.icbc.OperatorBC(geomtime, lambda x, y, _: dddy(x, y), boundary_r)

    ic = dde.icbc.IC(
        geomtime,
        lambda x: 0,
        lambda _, on_initial: on_initial,
    ) # Not used

    # Collection points
    n_data = 6000
    x_data = np.random.uniform(0, 1, n_data)
    eta_data = np.exp(np.random.uniform(np.log(0.0001), np.log(10), n_data))
    t_data = np.random.uniform(0, 10, n_data)
    data = np.column_stack((x_data, eta_data, t_data))

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc2, bc3, bc4],
        num_domain=6000,
        num_boundary=2000,
        num_test=10000,
    )

    net = dde.nn.FNN([3] + [20] * 4 + [1], "tanh", "Glorot uniform")
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)

    total_sets = sets_adam + sets_lbfgs + sets_adam2
    errors_train = np.zeros(total_sets)
    errors_test = np.zeros(total_sets)

    # model.compile("L-BFGS")
    # model.restore(save_path = f"./run10/model-45000.pt") # Restore previous weights

    print('Training Adam optimiser... \n')
    model.compile("adam", lr=0.001)    

    for s in range(sets_adam):
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        errors_train[s] = dde.metrics.l2_relative_error(y_train, y_pred_train)
        errors_test[s] = dde.metrics.l2_relative_error(y_test, y_pred_test)
        print(f'L2 rel. errors. Prediction: {errors_train[s]}. Extrapolation: {errors_test[s]}')
        losshistory, train_state = model.train(iterations=1000, model_save_path=f'./{results_folder}/model')

    print('Training L-BFGS optimiser... \n')
    for s in range(sets_lbfgs):

        dde.optimizers.set_LBFGS_options(maxiter=1000)
        model.compile("L-BFGS")
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        errors_train[s+sets_adam] = dde.metrics.l2_relative_error(y_train, y_pred_train)
        errors_test[s+sets_adam] = dde.metrics.l2_relative_error(y_test, y_pred_test)
        losshistory, train_state = model.train(iterations=1000, model_save_path=f'./{results_folder}/model')
    
    print('Training Adam (2) optimiser... \n')
    model.compile("adam", lr=1e-3)
    for s in range(sets_adam2):
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        errors_train[s+sets_adam+sets_lbfgs] = dde.metrics.l2_relative_error(y_train, y_pred_train)
        errors_test[s+sets_adam+sets_lbfgs] = dde.metrics.l2_relative_error(y_test, y_pred_test)
        losshistory, train_state = model.train(iterations=1000, model_save_path=f'./{results_folder}/model')

    print('Saving data... \n')
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=f'./{results_folder}')

    np.save(f'./{results_folder}/errors_train.npy', errors_train)
    np.save(f'./{results_folder}/errors_test.npy', errors_test)
