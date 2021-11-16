
from omegaconf import OmegaConf
import torch

"""
Configs for each of the network ensemble models
"""

model_cfg_dict = {
    #Input/output size must be inherited from training data
    'input_size': '???',
    'output_size': '???', 
    'device': 'cuda:0' if not torch.cuda.is_available() else 'cpu',
    'num_layers': 3,
    'propagation_method': None,

    #BNN spesific parameters
    'BNN': {
        'hid_size': 200,
        'activation_fn_cfg' : {"_target_": "torch.nn.SiLU", 'negative_slope': 0.01},
        "ensemble_size": 5,
        'prior_sigma': (1, 0.01),
        'prior_pi': 0.8,
        'freeze': False
    },

    #PNN spesific parameters
    'PNN': {
        'hid_size': 200,
        'activation_fn_cfg' : {"_target_": "torch.nn.LeakyReLu", 'negative_slope': 0.01},
        'ensemble_size': 5,
        'deterministic': False
    }
}

model_cfg = OmegaConf.create(model_cfg_dict)