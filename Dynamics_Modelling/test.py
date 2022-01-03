
from mbrl.models.gaussian_mlp import GaussianMLP
from numpy.lib.npyio import load, save
import torch
import numpy as np
from Modelling.Models.BNN import BNN

from configs.model_config import model_cfg
from configs.train_config import train_cfg

from util import labeled_data, model_from_cfg, train_data_from_array, create_test_data, load_model, monte_carlo_mean_var, seed_everything
from Modelling.train import  train
from evaluator import DatasetEvaluator

import matplotlib.pyplot as plt


def test_train(model_type):
    _, _, x_train, y_train = create_test_data()
    train_data = labeled_data(x_train, y_train)
    model_cfg.input_size = train_data["x"].shape[1]
    model_cfg.output_size = train_data["y"].shape[1]
    if model_type == "BNN":
        train_cfg.normalize = False
    train_cfg.target_is_delta = False

    print("Input: ",model_cfg.input_size)
    print("Output: ", model_cfg.output_size)


    model = model_from_cfg(model_cfg, model_type)
    log_dir = f"Logs/{model_type}_sine_test_final"

    train(model, train_data, train_cfg, model_cfg, log_dir=log_dir)

def test_load(load_dir):
    model = load_model(load_dir)
    test_sine_model(model, save_dir=load_dir)

def test_sine_model(model_wrapper, save_dir = None):

    num_points = 10000
    x_data = np.linspace(-12, 12, num_points)
    y_data = np.sin(x_data)

    x_data, y_data, x_train, y_train = create_test_data()
    
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
    plt.plot(x_train, y_train, '.', markersize=0.9)
    plt.fill_between(x_data, y_pred, y_pred + 2 * y_std, color='b', alpha=0.2)
    plt.fill_between(x_data, y_pred - 2 * y_std, y_pred, color='b', alpha=0.2)
    plt.axis([-12, 12, -2.5, 2.5])
    plt.legend(["Test data", "Train data", "Model prediction", "Pred uncertainty"])
    plt.title("Model prediction results")
    plt.xlabel("Data intput x")
    plt.ylabel("Data / Model output [sin(x)]")
    if save_dir is not None:
        plt.savefig(save_dir + "prediction_results.svg", format = "svg")
    plt.show()

def test_evaulator(model_dir, data_dir= None, out_dir = None):
    
    if data_dir is None:
        data_dir = model_dir
    
    if out_dir is None:
        out_dir = model_dir + "/evalutaion"
    
    evaluator = DatasetEvaluator(model_dir, data_dir, out_dir)
    evaluator.run()


def test_BNN():

    ensemble = BNN(1, 1, torch.device("cpu"), 1, 1, hid_size=5)
    # ensemble = GaussianMLP(1,1,torch.device("cpu"), 1, 1, hid_size=5)
    # ensemble.MOPED_()
    print(list(ensemble.named_parameters()))
    # ensemble.MOPED_()
    print(list(ensemble.named_parameters()))

    
    optim = torch.optim.Adam(ensemble.parameters())
    
    x = torch.tensor([[1.]])
    y = torch.sin(x)
    print(x,y)
    print(x.shape, y.shape)

    pre = list(ensemble.named_parameters())

    print(pre)
    # pred_x = ensemble(x)
    for _ in range(1000):
        loss,_ = ensemble.loss(x,y)
        # print(loss)
        loss.backward()
        optim.step()
    print('-'*50)
    print(list(ensemble.named_parameters()))
    print(ensemble(x), y)
    

if __name__ == "__main__":
    seed_everything(42)

    # test_BNN()
    test_train("BNN")
    test_load("Logs/BNN_sine_test_final")
    # test_evaulator("Logs/PNN_sine_test")

    # test_train("BNN")
    # test_load("Logs/BNN_sine_test2")
    # test_evaulator("Logs/BNN_sine_test")

    x_data, y_data, x_train, y_train = create_test_data()
    plt.figure(figsize=(16, 8))
    plt.plot(x_data, y_data, x_train, y_train, '.', x_val, y_val, 'o', markersize=4)
    plt.show()


    '''
    idx_list: 
    t       0
    Force   1,  2,  3
    Torque  4,  5,  6
    Pos     7,  8,  9
    Vel     10, 11, 12
    Ang.vel 13, 14, 15
    '''

