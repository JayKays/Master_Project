

from dotmap import DotMap
from configs.default import cfg
from Modelling.train import train
import mbrl.models as models
import torch
import numpy as np

from Modelling.Models.BNN import BNN

def data(num_samples = 100):

    # x_data = np.linspace(-12, 12, num_samples)
    # y_data = np.sin(x_data)

    x_data = np.linspace(-12, 12, 10000)
    y_data = np.sin(x_data)

    train_size = 2000
    val_size = 200
    x_train = np.zeros(2 * train_size)
    y_train = np.zeros(2 * train_size)
    x_val = np.zeros(2 * val_size)
    y_val = np.zeros(2 * val_size)

    # Half with lower noise
    train_val_idx_1 = np.random.choice(list(range(1200, 3500)), 
                                    size=train_size + val_size, 
                                    replace=False)
    mag = 0.05
    x_train[:train_size] = x_data[train_val_idx_1[:train_size]]
    y_train[:train_size] = y_data[train_val_idx_1[:train_size]] + mag * np.random.randn(train_size)
    x_val[:val_size] = x_data[train_val_idx_1[train_size:]]
    y_val[:val_size] = y_data[train_val_idx_1[train_size:]] + mag * np.random.randn(val_size)

    # Half with higher noise
    train_val_idx_2 = np.random.choice(list(range(6500, 8800)), 
                                    size=train_size + val_size, 
                                    replace=False)
    mag = 0.20
    x_train[train_size:] = x_data[train_val_idx_2[:train_size]]
    y_train[train_size:] = y_data[train_val_idx_2[:train_size]] + mag * np.random.randn(train_size)
    x_val[val_size:] = x_data[train_val_idx_2[train_size:]]
    y_val[val_size:] = y_data[train_val_idx_2[train_size:]] + mag * np.random.randn(val_size)

    return DotMap({'x':x_train, 'y': y_train})


def main():
    
    cfg.num_epochs = 50

    device = torch.device("cuda:0") if torch.cuda.is_available() \
        else torch.device('cpu')
    train_data = np.load("Dynamics_Modelling/Data_preperation/test_data.npy").T

    print(train_data.shape)

    ensemble = BNN(
        train_data.shape[1], # input size
        train_data.shape[1], # output size
        device, 
        num_layers=3,
        hid_size=64, 
        activation_fn_cfg={"_target_": "torch.nn.ReLU"}, 
        ensemble_size=2,
        propagation_method = 'expectation'
    )

    train(ensemble, train_data, cfg)



if __name__ == "__main__":
    main()
