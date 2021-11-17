
from mbrl.models.gaussian_mlp import GaussianMLP
from numpy.lib.npyio import load
import torch
import numpy as np
from Modelling.Models.BNN import BNN

from configs.model_config import model_cfg
from configs.train_config import train_cfg

from util import labeled_data, model_from_cfg, train_data_from_array, create_test_data, load_model, monte_carlo_mean_var
from Modelling.train import  train

import matplotlib.pyplot as plt


def test_train(model_type):
    _, _, x_train, y_train = create_test_data()
    train_data = labeled_data(x_train, y_train)
    model_cfg.input_size = train_data["x"].shape[1]
    model_cfg.output_size = train_data["y"].shape[1]

    print("Input: ",model_cfg.input_size)
    print("Output: ", model_cfg.output_size)

    model = model_from_cfg(model_cfg, model_type)
    log_dir = f"Logs/{model_type}_sine_test"

    train(model, train_data, train_cfg, model_cfg, log_dir=log_dir)

def test_load(load_dir):
    model = load_model(load_dir)
    test_sine_model(model)

def test_sine_model(model_wrapper):

    num_points = 10000
    x_data = np.linspace(-12, 12, num_points)
    y_data = np.sin(x_data)

    x_tensor = torch.from_numpy(x_data).unsqueeze(1).float().to(model_wrapper.device)
    try:
        x_tensor = model_wrapper.input_normalizer.normalize(x_tensor)
    except:
        print("No normalizer in model")
    
    model = model_wrapper.model

    with torch.no_grad():
        if isinstance(model, BNN):
            y_pred, y_pred_var = monte_carlo_mean_var(model, x_tensor)
            y_pred = y_pred[..., 0]
            y_pred_var = y_pred_var[..., 0]
            y_var = y_pred_var
        elif isinstance(model, GaussianMLP):
            y_pred, y_pred_logvar = model(x_tensor)
            y_pred = y_pred[..., 0]
            y_pred_logvar = y_pred_logvar[..., 0]
            y_var = y_pred_logvar.exp()
        else:
            print("No available model of type BNN or PNN")
            return
    
    y_var_epi = y_pred.var(dim=0).cpu().numpy()
    y_pred = y_pred.mean(dim=0).cpu().numpy()
    y_var_ale = y_var.mean(dim=0).cpu().numpy()

    var = y_var_epi + y_var_ale
    y_std = np.sqrt(var)
    plt.figure(figsize=(16, 8))
    plt.plot(x_data, y_data, 'r')
    plt.plot(x_data, y_pred, 'b-', markersize=4)
    plt.fill_between(x_data, y_pred, y_pred + 2 * y_std, color='b', alpha=0.2)
    plt.fill_between(x_data, y_pred - 2 * y_std, y_pred, color='b', alpha=0.2)
    plt.axis([-12, 12, -2.5, 2.5])
    plt.legend(["Test data", "Train data", "Model prediction", "Pred uncertainty"])
    plt.title("Model prediction results")
    plt.show()

if __name__ == "__main__":
    
    test_train("BNN")
    test_train("PNN")

    test_load("Logs/BNN_sine_test")
    test_load("Logs/PNN_sine_test")

    '''
    idx_list: 
    t       0
    Force   1,  2,  3
    Torque  4,  5,  6
    Pos     7,  8,  9
    Vel     10, 11, 12
    Ang.vel 13, 14, 15
    '''

