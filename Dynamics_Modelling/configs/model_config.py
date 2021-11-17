
from omegaconf import OmegaConf
import torch

"""
Configs for each of the network ensemble models
"""

model_cfg_dict = {
    #Input/output size must be inherited from training data
    'input_size': '???',
    'output_size': '???', 
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'num_layers': 3,
    'propagation_method': None,
    # 'activation_fn_cfg' : {"_target_": "torch.nn.LeakyReLU", 'negative_slope': 0.01},
    'activation_fn_cfg': {"_target_": "torch.nn.SiLU"},
    'hid_size': 150,
    'ensemble_size': 5,
    #BNN spesific parameters
    'BNN': {
        'prior_sigma': (1, 0.01),
        'prior_pi': 0.8,
        'freeze': False
    },

    #PNN spesific parameters
    'PNN': {
        'deterministic': False
    }   
}
model_cfg = OmegaConf.create(model_cfg_dict)
print(f"Model config device: {model_cfg.device}")