

from dotmap import DotMap
from configs.default import cfg
from Modelling.train import train
import mbrl.models as models
import torch
import numpy as np

from Modelling.Models.BNN import BNN


def data(file_path, state_idx, act_idx):

    data_arr = np.loadt(file_path)
    
    state_obs = data_arr[state_idx]
    
    if act_idx is not None:
        actions = data_arr[act_idx]

        return state_obs, actions
    
    return state_obs, None


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
